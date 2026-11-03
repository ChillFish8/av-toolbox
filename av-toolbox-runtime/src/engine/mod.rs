mod aggregate;
mod channel;
mod disperse;
mod freedom;
mod passthrough;
mod plan;
mod region;
mod source;
mod spawn;
#[cfg(test)]
pub(crate) mod tests;
mod worker;
mod wrap;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use av_toolbox_core::{ArtefactSink, PipelineRegistry};
use av_toolbox_gpu::{Backend, FrameBudget, FrameContext};
use av_toolbox_graph::{
    ArtefactSinkNode,
    CompiledGraph,
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    NodeId,
    Region,
    RegionId,
    Routing,
    SolvedFormats,
    StageBox,
};

#[cfg(test)]
pub(crate) use self::aggregate::run_aggregate;
use self::channel::{DownstreamLink, Slotted};
use self::freedom::FirstIdle;
pub(crate) use self::freedom::WorkerFreedom;
use self::passthrough::run_passthrough;
pub(crate) use self::plan::FRAME_UNITS_IN_FLIGHT;
#[cfg(test)]
pub(crate) use self::plan::HEADROOM_ITEMS_PER_WORKER;
use self::plan::{
    CHUNK_RESULTS_IN_FLIGHT,
    bound_slots,
    build_region_edges,
    default_source_share,
    holds_nothing_in_frame_region,
    inherited_budgets,
    is_source_region,
    region_is_supported,
    source_budget,
};
pub(crate) use self::region::{AggregateSide, WorkerSlot};
use self::region::{DisperseSide, ExchangedRegion, PassthroughRegion, PreparedRegion, SourceRegion};
use self::source::run_source;
use self::spawn::{Outcome, spawn, spawn_exchanged};
#[cfg(test)]
pub(crate) use self::worker::run_worker;
use self::wrap::{SinkTally, wrap_head_source, wrap_sinks};
use crate::artefacts::{ArtefactCollector, write_artefacts};
use crate::profile::Profiler;
use crate::worker::RegionRunner;
use crate::{Aggregate, Cancel, Disperse, NodeTiming, Profiling, Progress, RunError};

/// What one run of an [Engine] is allowed to keep in flight.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// The total frame memory every source may hold outstanding at once, shared out evenly
    /// between them. `None` derives one from the plan itself.
    pub frame_budget: Option<usize>,
    /// How long a source may wait for room on its share before the run fails rather than hangs.
    /// `None` leaves [FrameBudget](av_toolbox_gpu::FrameBudget) its own default.
    pub stall_deadline: Option<Duration>,
    /// How many frames every source may produce before it ends. `None` lets each run to its end.
    pub frame_limit: Option<u64>,
    /// How many items reach the sinks before [RunReport::counted_wall] starts counting.
    pub warmup: u64,
    /// Times every stage call when set.
    pub profiling: Option<Profiling>,
}

/// What one run of an [Engine] produced.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// How many items sinks consumed, counting one for every input slot they were given.
    ///
    /// A sink reading two inputs therefore counts two for each step it runs, since two items
    /// genuinely reached it.
    pub frames: u64,
    /// How many files the artefact sinks wrote, zero for a cancelled run.
    pub artefacts: usize,
    /// Whether the run was stopped before its sources ran out.
    pub cancelled: bool,
    /// How many items a stopped run was still holding, and so never gave to a worker.
    ///
    /// This counts what a region had in its own hands, not what was still travelling between
    /// two of them. A run that was not cancelled never reports any, since losing items without
    /// being asked to stop fails the run instead.
    pub dropped: usize,
    /// How many records the sinks received.
    pub records: u64,
    /// How long the run took from start to finish.
    pub wall: Duration,
    /// How long the run took after [Settings::warmup] items had reached the sinks.
    ///
    /// Zero when the warmup was never reached, and the whole run when there was none.
    pub counted_wall: Duration,
    /// Every node's counted calls in node order, empty unless the run was profiled.
    pub timings: Vec<NodeTiming>,
}

/// A graph, planned and instantiated into threads ready to run.
///
/// [Engine::build] does everything that can fail on its own, instantiating every worker's
/// stages and wiring the channels between them. [Engine::run] only spawns the threads that
/// wiring describes and waits for them to finish.
pub struct Engine {
    prepared: Vec<PreparedRegion>,
    tally: Arc<SinkTally>,
    progress: Progress,
    dropped: Arc<AtomicUsize>,
    cancel: Arc<Cancel>,
    profiler: Option<Arc<Profiler>>,
    artefacts: Arc<ArtefactCollector>,
    /// The artefact sinks the graph declared, lined up with `artefact_writers`.
    artefact_sinks: Vec<ArtefactSinkNode>,
    artefact_writers: Vec<Box<dyn ArtefactSink>>,
}

impl Engine {
    /// Instantiates every worker a planned graph needs, against a real backend, and wires
    /// them together.
    ///
    /// This is `Engine::build_with` with the one closure a real run always wants: build a
    /// worker's stages by asking `registry` and `backend` to instantiate the region for real.
    pub fn build(
        graph: &CompiledGraph,
        registry: &PipelineRegistry,
        backend: &Backend,
        formats: &SolvedFormats,
        settings: Settings,
    ) -> Result<Engine, Diagnostics> {
        let held = graph.held_frames(registry, formats)?;
        let writers = graph.build_artefact_sinks(registry)?;

        let engine = Self::build_with(
            graph,
            formats,
            &held,
            settings,
            |region, _worker| graph.instantiate_region(registry, backend, formats, region),
            || backend.frame_context(),
        )?;

        Ok(engine.writing_artefacts_to(writers))
    }

    /// Instantiates every worker a planned graph needs, however `instantiate` and
    /// `make_context` produce them, and wires them together.
    ///
    /// A backend is only ever needed to build a stage. Everything from here on is threads,
    /// channels and instances that are already `Send`, so a test can supply its own closures
    /// building fixtures directly, exactly as [crate::worker::region]'s own tests build a
    /// [RegionRunner] by hand, and exercise this whole topology without a device.
    ///
    /// `held` is how many frames each node may hold back, indexed by node, as
    /// [CompiledGraph::held_frames](av_toolbox_graph::CompiledGraph::held_frames) gives it.
    ///
    /// `graph` must already carry an [ExecutionPlan](av_toolbox_graph::ExecutionPlan). Every
    /// region is built in isolation, so one region's stages failing to build does not stop
    /// another's from being attempted, and every failure is reported together.
    pub(crate) fn build_with<Instantiate, MakeContext>(
        graph: &CompiledGraph,
        formats: &SolvedFormats,
        held: &[usize],
        settings: Settings,
        mut instantiate: Instantiate,
        mut make_context: MakeContext,
    ) -> Result<Engine, Diagnostics>
    where
        Instantiate: FnMut(&Region, usize) -> Result<Vec<(NodeId, StageBox)>, Diagnostics>,
        MakeContext: FnMut() -> FrameContext,
    {
        let plan = graph
            .execution_plan()
            .expect("a graph must be planned before an engine can be built from it");
        let regions = plan.regions();

        let source_workers: usize = regions
            .iter()
            .filter(|region| is_source_region(graph, region))
            .map(Region::width)
            .sum();

        let total_budget = settings
            .frame_budget
            .unwrap_or_else(|| default_source_share(graph, formats, regions, held) * source_workers);
        let source_share = total_budget.checked_div(source_workers).unwrap_or(0);

        let mut diagnostics = Diagnostics::new();
        let tally = Arc::new(SinkTally::new(settings.warmup));
        let progress = Progress::new(graph.sinks());
        let dropped = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(Cancel::new());
        let profiler = settings.profiling.clone().map(Profiler::new).map(Arc::new);
        let artefacts = Arc::new(ArtefactCollector::new(graph));

        let mut built_regions = Vec::with_capacity(regions.len());
        for region in regions {
            let supported = region_is_supported(graph, region, &mut diagnostics);
            let holds_nothing = holds_nothing_in_frame_region(graph, region, held, &mut diagnostics);
            if !supported || !holds_nothing {
                built_regions.push(Vec::new());
                continue;
            }

            let width = if region.has_disperse() { region.width() } else { 1 };
            let mut workers = Vec::with_capacity(width);
            let mut refused = false;

            for worker in 0..width {
                match instantiate(region, worker) {
                    Ok(mut nodes) => {
                        // Every worker refuses alike, so the first refusal is reported alone.
                        if !refused {
                            refused = !announce_wanted(graph, &artefacts, &mut nodes, &mut diagnostics);
                        }

                        workers.push(nodes);
                    },
                    Err(region_diagnostics) => {
                        for diagnostic in region_diagnostics.into_inner() {
                            diagnostics.push(diagnostic);
                        }
                    },
                }
            }

            built_regions.push(workers);
        }

        if diagnostics.has_errors() {
            return Err(diagnostics);
        }

        // Every producer feeding one consumer region shares one channel, cloned once per edge
        // and tagged with the slot that edge fills, so the consumer gathers a whole bundle from
        // one receiver instead of racing one thread per slot into it.
        let mut inbound: HashMap<RegionId, Receiver<Slotted>> = HashMap::new();
        let mut outbound: HashMap<NodeId, Vec<DownstreamLink>> = HashMap::new();

        for region in regions {
            let head = graph.node(region.head());
            if head.inputs().is_empty() {
                continue;
            }

            let (sender, receiver) = mpsc::channel();
            for edge in head.inputs() {
                outbound
                    .entry(edge.source())
                    .or_default()
                    .push((edge.slot(), sender.clone()));
            }

            inbound.insert(region.id(), receiver);
        }

        // Regions are listed in topological order, so a source's budget is always made before any
        // region inheriting it is wired. An inheriting region charges that budget without waiting
        // on it, so only the source is held back and nothing after it can deadlock on its room.
        let inherited = inherited_budgets(graph, regions);
        let mut source_budgets: HashMap<RegionId, Arc<FrameBudget>> = HashMap::new();

        let mut prepared = Vec::with_capacity(regions.len());
        for (region, workers) in regions.iter().zip(built_regions) {
            let downstream: Vec<DownstreamLink> = region
                .exits()
                .iter()
                .flat_map(|exit| outbound.remove(exit).unwrap_or_default())
                .collect();

            if !region.has_disperse() {
                let nodes = workers
                    .into_iter()
                    .next()
                    .expect("a region always instantiates at least one worker");
                debug_assert_eq!(
                    nodes.first().map(|(id, _)| *id),
                    Some(region.head()),
                    "a region's stages must list its head first"
                );
                let nodes = wrap_sinks(nodes, tally.clone(), &progress);

                if is_source_region(graph, region) {
                    let budget = source_budget(source_share, settings.stall_deadline);
                    let released = budget.clone();
                    cancel.on_stop(move || released.release());

                    if let Some(profiler) = &profiler {
                        budget.set_waits(profiler.clone());
                    }

                    source_budgets.insert(region.id(), budget.clone());

                    let context = make_context().with_budget(budget.clone());
                    let exhausted = Arc::new(AtomicBool::new(false));
                    let nodes = wrap_head_source(
                        nodes,
                        exhausted.clone(),
                        settings.frame_limit,
                        progress.read_counter(),
                    );
                    let edges = build_region_edges(graph, region);
                    let runner = RegionRunner::new(nodes, edges)
                        .routed_on(region.route())
                        .profiled(profiler.clone())
                        .collecting(artefacts.clone());

                    prepared.push(PreparedRegion::Source(SourceRegion {
                        runner,
                        context,
                        budget,
                        exhausted,
                        downstream,
                    }));
                } else {
                    let inherited_budget = inherited
                        .get(&region.id())
                        .and_then(|source| source_budgets.get(source));
                    let context = match inherited_budget {
                        Some(budget) => make_context().with_charged_budget(budget.clone()),
                        None => make_context(),
                    };
                    let edges = build_region_edges(graph, region);
                    let links = graph.node(region.head()).inputs().len();
                    let runner = RegionRunner::new(nodes, edges)
                        .fed_by(links)
                        .routed_on(region.route())
                        .profiled(profiler.clone())
                        .collecting(artefacts.clone());
                    let upstream = inbound
                        .remove(&region.id())
                        .expect("a region that reads no input is refused before anything is wired");

                    prepared.push(PreparedRegion::Passthrough(PassthroughRegion {
                        runner,
                        context,
                        upstream,
                        downstream,
                    }));
                }

                continue;
            }

            let width = workers.len();
            let routing = region.routing();
            let authority = match routing {
                Routing::ByChunk => {
                    let slot = region.chunk_authority().expect(
                        "a chunk routed disperse with no chunk authority is refused before anything is wired",
                    );
                    Some(slot)
                },
                Routing::ByFrame => None,
            };
            let bound = bound_slots(graph, region.head());
            let slots: Vec<usize> = bound
                .iter()
                .enumerate()
                .filter(|&(_, &is_bound)| is_bound)
                .map(|(slot, _)| slot)
                .collect();
            // How many units the dispatch policy hands a worker before its aggregate drains one.
            let units_in_flight = match routing {
                Routing::ByChunk => 1,
                Routing::ByFrame => FRAME_UNITS_IN_FLIGHT,
            };

            // How many results a worker may produce ahead of its aggregate before it has to wait.
            // A chunk routed worker holds one chunk, but a later chunk only drains once every one
            // before it has, so it needs room for many of its frames rather than one.
            let results_in_flight = match routing {
                Routing::ByChunk => CHUNK_RESULTS_IN_FLIGHT,
                Routing::ByFrame => FRAME_UNITS_IN_FLIGHT,
            };

            let freedom = Arc::new(WorkerFreedom::new(width, units_in_flight));
            let woken = freedom.clone();
            cancel.on_stop(move || woken.wake_everyone());

            let dispatch = FirstIdle {
                freedom: freedom.clone(),
            };
            let disperse = match authority {
                Some(authority) => Disperse::new(&bound, authority, dispatch).with_route(region.route()),
                None => Disperse::by_frame(&bound, dispatch),
            };
            let upstream = inbound
                .remove(&region.id())
                .expect("a region with a disperse reads at least one input");

            let (assignment_sender, assignment_receiver) = mpsc::channel();

            let mut work_senders = Vec::with_capacity(width);
            let mut work_receivers = Vec::with_capacity(width);
            for _ in 0..width {
                let (work_sender, work_receiver) = mpsc::channel();
                work_senders.push(work_sender);
                work_receivers.push(work_receiver);
            }

            let mut worker_slots = Vec::with_capacity(width);
            let mut result_receivers = Vec::with_capacity(width);
            for (nodes, work_receiver) in workers.into_iter().zip(work_receivers) {
                debug_assert_eq!(
                    nodes.first().map(|(id, _)| *id),
                    Some(region.head()),
                    "a region's stages must list its head first"
                );
                let nodes = wrap_sinks(nodes, tally.clone(), &progress);
                let edges = build_region_edges(graph, region);
                let runner = RegionRunner::new(nodes, edges)
                    .routed_on(region.route())
                    .profiled(profiler.clone())
                    .collecting(artefacts.clone());
                let context = make_context();
                let (result_sender, result_receiver) = mpsc::sync_channel(results_in_flight);

                result_receivers.push(result_receiver);
                worker_slots.push(WorkerSlot {
                    runner,
                    context,
                    work_receiver,
                    result_sender,
                });
            }

            let aggregate = region.has_aggregate().then(|| AggregateSide {
                aggregate: Aggregate::new(),
                assignment_receiver,
            });

            prepared.push(PreparedRegion::Exchanged(ExchangedRegion {
                disperse: DisperseSide {
                    disperse,
                    upstream,
                    slots,
                    freedom: freedom.clone(),
                    work_senders,
                    assignment_sender,
                },
                freedom,
                aggregate,
                result_receivers,
                workers: worker_slots,
                downstream,
            }));
        }

        Ok(Engine {
            prepared,
            tally,
            progress,
            dropped,
            cancel,
            profiler,
            artefacts,
            artefact_sinks: graph.artefact_sinks().to_vec(),
            artefact_writers: Vec::new(),
        })
    }

    /// The same engine, writing the artefacts its graph's artefact sinks select through `writers`.
    ///
    /// `writers` is what [CompiledGraph::build_artefact_sinks](av_toolbox_graph::CompiledGraph::build_artefact_sinks)
    /// built from the same graph, in the same order.
    pub fn writing_artefacts_to(mut self, writers: Vec<Box<dyn ArtefactSink>>) -> Self {
        assert_eq!(
            writers.len(),
            self.artefact_sinks.len(),
            "every artefact sink the graph declares needs exactly one writer"
        );

        self.artefact_writers = writers;
        self
    }

    /// The same engine, collecting and merging artefacts as usual but writing none of them.
    pub fn discarding_artefacts(mut self) -> Self {
        self.artefact_writers = Vec::new();
        self
    }

    /// The signal that stops this run, wherever its threads have got to.
    ///
    /// Stopping is not an error. The run winds down, every thread joins, and
    /// [Engine::run] answers with a report saying it was cancelled and what it was still holding.
    pub fn cancel_handle(&self) -> Arc<Cancel> {
        self.cancel.clone()
    }

    /// A live view of this run, readable from any thread while it goes.
    pub fn progress(&self) -> Progress {
        self.progress.clone()
    }

    /// Spawns every worker, disperse and aggregate thread the plan needs, and waits for them
    /// all to finish.
    ///
    /// The first failure any thread reports stops the run and is the one returned. Stopping
    /// reaches a thread wherever it is parked: a channel by its senders being dropped as the
    /// threads holding them exit, the frame budget by being released, and a wait for a free
    /// worker by being woken. A thread that panics has its panic caught where it happened and
    /// turned into an error for the run, so the join loop below always runs to the end and every
    /// thread is joined.
    ///
    /// A run nobody cancelled that still ends holding items it never gave to a worker fails as
    /// well. Those items went nowhere, and answering with a report a caller has to read the
    /// fields of would let a pipeline lose frames and still look like it succeeded.
    ///
    /// An action set with [Progress::on_first_output] that has not run by the end is dropped then,
    /// however the run ended.
    pub fn run(self) -> Result<RunReport, RunError> {
        let progress = self.progress.clone();
        let _first_output = FirstOutputGuard(progress);

        self.run_threads()
    }

    fn run_threads(self) -> Result<RunReport, RunError> {
        let Engine {
            prepared,
            tally,
            progress,
            dropped,
            cancel,
            profiler,
            artefacts: collector,
            artefact_sinks,
            artefact_writers,
        } = self;

        let started = Instant::now();
        progress.start();
        let outcome = Arc::new(Outcome {
            cancel: cancel.clone(),
            error: Mutex::new(None),
        });

        let mut handles: Vec<JoinHandle<()>> = Vec::new();
        for region in prepared {
            match region {
                PreparedRegion::Source(region) => {
                    let cancel = cancel.clone();
                    spawn(&mut handles, &outcome, move || run_source(region, &cancel));
                },
                PreparedRegion::Passthrough(region) => {
                    spawn(&mut handles, &outcome, move || run_passthrough(region));
                },
                PreparedRegion::Exchanged(region) => {
                    spawn_exchanged(region, &mut handles, &outcome, &cancel, &dropped);
                },
            }
        }

        for handle in handles {
            if handle.join().is_err() {
                outcome.report(RunError::Panicked {
                    message: "a thread panicked as it was being joined".to_string(),
                });
            }
        }

        let ended = Instant::now();

        if let Some(error) = outcome.take_error() {
            return Err(error);
        }

        let cancelled = cancel.is_stopped();
        let dropped = dropped.load(Ordering::SeqCst);
        if dropped > 0 && !cancelled {
            return Err(RunError::DroppedItems { dropped });
        }

        let artefacts = match cancelled {
            true => 0,
            false => write_artefacts(&collector, &artefact_sinks, artefact_writers)?,
        };

        let wall = ended.duration_since(started);
        let counted_wall = tally.counted_wall(started, ended);

        let timings = match &profiler {
            Some(profiler) => profiler.timings(),
            None => Vec::new(),
        };

        Ok(RunReport {
            frames: tally.frames(),
            artefacts,
            cancelled,
            dropped,
            records: tally.records(),
            wall,
            counted_wall,
            timings,
        })
    }
}

/// Drops a run's first-output action when the run returns or panics.
struct FirstOutputGuard(Progress);

impl Drop for FirstOutputGuard {
    fn drop(&mut self) {
        self.0.drop_first_output();
    }
}

/// Tells each of a worker's nodes what is wanted of it, reporting every node that refuses.
///
/// Answers whether every node accepted.
fn announce_wanted(
    graph: &CompiledGraph,
    artefacts: &ArtefactCollector,
    nodes: &mut [(NodeId, StageBox)],
    diagnostics: &mut Diagnostics,
) -> bool {
    let mut accepted = true;

    for (id, stage) in nodes.iter_mut() {
        let Err(error) = artefacts.announce(*id, stage) else {
            continue;
        };

        let node = graph.node(*id);
        let path = node.path();
        let message = format!("{path} cannot produce the artefacts wanted of it: {error}");
        let diagnostic = Diagnostic::error(DiagnosticCode::StageBuildFailed, path, node.span(), message);

        diagnostics.push(diagnostic);
        accepted = false;
    }

    accepted
}
