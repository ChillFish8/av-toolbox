mod aggregate;
mod channel;
mod disperse;
mod freedom;
mod passthrough;
mod plan;
mod region;
mod source;
mod spawn;
mod worker;
mod wrap;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, FrameContext};
use av_toolbox_graph::{
    CompiledGraph,
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
    bound_slots,
    build_region_edges,
    default_source_share,
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
use self::wrap::{wrap_head_source, wrap_sinks};
use crate::worker::RegionRunner;
use crate::{Aggregate, Cancel, Disperse, RunError, RunWarning};

/// What one run of an [Engine] is allowed to keep in flight.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// The total frame memory every source may hold outstanding at once, shared out evenly
    /// between them. `None` derives one from the plan itself.
    pub frame_budget: Option<usize>,
    /// How long a source may wait for room on its share before the run fails rather than hangs.
    /// `None` leaves [FrameBudget] its own default.
    ///
    /// [FrameBudget]: av_toolbox_gpu::FrameBudget
    pub stall_deadline: Option<Duration>,
}

/// What one run of an [Engine] produced.
#[derive(Debug, Clone)]
pub struct RunReport {
    /// How many items sinks consumed, counting one for every input slot they were given.
    ///
    /// A sink reading two inputs therefore counts two for each step it runs, since two items
    /// genuinely reached it.
    pub frames: u64,
    /// Whether the run was stopped before its sources ran out.
    pub cancelled: bool,
    /// How many items a stopped run was still holding, and so never gave to a worker.
    ///
    /// This counts what a region had in its own hands, not what was still travelling between
    /// two of them. A run that was not cancelled never reports any, since losing items without
    /// being asked to stop fails the run instead.
    pub dropped: usize,
    /// Things the run noticed that did not stop it.
    pub warnings: Vec<RunWarning>,
}

/// A graph, planned and instantiated into threads ready to run.
///
/// [Engine::build] does everything that can fail on its own, instantiating every worker's
/// stages and wiring the channels between them. [Engine::run] only spawns the threads that
/// wiring describes and waits for them to finish.
pub struct Engine {
    prepared: Vec<PreparedRegion>,
    frames: Arc<AtomicU64>,
    dropped: Arc<AtomicUsize>,
    warnings: Arc<Mutex<Vec<RunWarning>>>,
    cancel: Arc<Cancel>,
}

impl Engine {
    /// Instantiates every worker a planned graph needs, against a real backend, and wires
    /// them together.
    ///
    /// This is [Engine::build_with] with the one closure a real run always wants: build a
    /// worker's stages by asking `registry` and `backend` to instantiate the region for real.
    pub fn build(
        graph: &CompiledGraph,
        registry: &PipelineRegistry,
        backend: &Backend,
        formats: &SolvedFormats,
        settings: Settings,
    ) -> Result<Engine, Diagnostics> {
        Self::build_with(
            graph,
            formats,
            settings,
            |region, _worker| graph.instantiate_region(registry, backend, formats, region),
            || backend.frame_context(),
        )
    }

    /// Instantiates every worker a planned graph needs, however `instantiate` and
    /// `make_context` produce them, and wires them together.
    ///
    /// A backend is only ever needed to build a stage. Everything from here on is threads,
    /// channels and instances that are already `Send`, so a test can supply its own closures
    /// building fixtures directly, exactly as [crate::worker::region]'s own tests build a
    /// [RegionRunner] by hand, and exercise this whole topology without a device.
    ///
    /// `graph` must already carry an [ExecutionPlan](av_toolbox_graph::ExecutionPlan). Every
    /// region is built in isolation, so one region's stages failing to build does not stop
    /// another's from being attempted, and every failure is reported together.
    pub(crate) fn build_with<Instantiate, MakeContext>(
        graph: &CompiledGraph,
        formats: &SolvedFormats,
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
            .unwrap_or_else(|| default_source_share(graph, formats, regions) * source_workers);
        let source_share = total_budget.checked_div(source_workers).unwrap_or(0);

        let mut diagnostics = Diagnostics::new();
        let frames = Arc::new(AtomicU64::new(0));
        let dropped = Arc::new(AtomicUsize::new(0));
        let warnings = Arc::new(Mutex::new(Vec::new()));
        let cancel = Arc::new(Cancel::new());

        let mut built_regions = Vec::with_capacity(regions.len());
        for region in regions {
            if !region_is_supported(graph, region, &mut diagnostics) {
                built_regions.push(Vec::new());
                continue;
            }

            let width = if region.has_disperse() { region.width() } else { 1 };
            let mut workers = Vec::with_capacity(width);

            for worker in 0..width {
                match instantiate(region, worker) {
                    Ok(nodes) => workers.push(nodes),
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
                let nodes = wrap_sinks(nodes, frames.clone());

                if is_source_region(graph, region) {
                    let budget = source_budget(source_share, settings.stall_deadline);
                    let released = budget.clone();
                    cancel.on_stop(move || released.release());

                    let context = make_context().with_budget(budget.clone());
                    let exhausted = Arc::new(AtomicBool::new(false));
                    let nodes = wrap_head_source(nodes, exhausted.clone());
                    let runner = RegionRunner::new(nodes, build_region_edges(graph, region));

                    prepared.push(PreparedRegion::Source(SourceRegion {
                        runner,
                        context,
                        budget,
                        exhausted,
                        downstream,
                    }));
                } else {
                    let context = make_context();
                    let runner = RegionRunner::new(nodes, build_region_edges(graph, region));
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
            // One local sizes both how many units the dispatch policy hands a worker and how deep
            // that worker's results channel is. They are one number because both describe the
            // same thing, how far a worker may run ahead of its aggregate before it has to wait.
            let capacity = match routing {
                Routing::ByChunk => 1,
                Routing::ByFrame => FRAME_UNITS_IN_FLIGHT,
            };

            let freedom = Arc::new(WorkerFreedom::new(width, capacity));
            let woken = freedom.clone();
            cancel.on_stop(move || woken.wake_everyone());

            let dispatch = FirstIdle {
                freedom: freedom.clone(),
            };
            let disperse = match authority {
                Some(authority) => Disperse::new(&bound, authority, dispatch),
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
                let nodes = wrap_sinks(nodes, frames.clone());
                let runner = RegionRunner::new(nodes, build_region_edges(graph, region));
                let context = make_context();
                let (result_sender, result_receiver) = mpsc::sync_channel(capacity);

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
                    authority,
                    width,
                    head: region.head(),
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
            frames,
            dropped,
            warnings,
            cancel,
        })
    }

    /// The signal that stops this run, wherever its threads have got to.
    ///
    /// Stopping is not an error. The run winds down, every thread joins, and
    /// [Engine::run] answers with a report saying it was cancelled and what it was still holding.
    pub fn cancel_handle(&self) -> Arc<Cancel> {
        self.cancel.clone()
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
    pub fn run(self) -> Result<RunReport, RunError> {
        let Engine {
            prepared,
            frames,
            dropped,
            warnings,
            cancel,
        } = self;

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
                    spawn_exchanged(region, &mut handles, &outcome, &cancel, &dropped, &warnings);
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

        if let Some(error) = outcome.take_error() {
            return Err(error);
        }

        let cancelled = cancel.is_stopped();
        let dropped = dropped.load(Ordering::SeqCst);
        if dropped > 0 && !cancelled {
            return Err(RunError::DroppedItems { dropped });
        }

        let warnings = warnings.lock().expect("not poisoned").clone();

        Ok(RunReport {
            frames: frames.load(Ordering::SeqCst),
            cancelled,
            dropped,
            warnings,
        })
    }
}
