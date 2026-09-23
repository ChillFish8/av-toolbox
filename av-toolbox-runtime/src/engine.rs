use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use av_toolbox_core::{Inputs, PipelineRegistry, Sink, Source, StageError, StageInstance, StageKind};
use av_toolbox_gpu::{Backend, Frame, FrameBudget, FrameContext, StreamItem};
use av_toolbox_graph::{
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

use crate::disperse::Dropped;
use crate::worker::{Feed, RegionEdges, RegionRunner};
use crate::{
    Aggregate,
    Assignment,
    Cancel,
    Dispatch,
    Disperse,
    ExchangeError,
    Routed,
    RunError,
    RunWarning,
    WorkItem,
    WorkResult,
};

/// What one run of an [Engine] is allowed to keep in flight.
#[derive(Debug, Clone, Default)]
pub struct Settings {
    /// The total frame memory every source may hold outstanding at once, shared out evenly
    /// between them. `None` derives one from the plan itself.
    pub frame_budget: Option<usize>,
    /// How long a source may wait for room on its share before the run fails rather than hangs.
    /// `None` leaves [FrameBudget] its own default.
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

/// A message crossing the unbounded channel between two regions.
#[derive(Clone)]
pub(crate) enum ExitMessage {
    Item(StreamItem),
    Done,
}

/// One region's message, tagged with the input slot it fills at the consumer.
///
/// A consumer with more than one input is fed by more than one upstream region, each with its
/// own sender cloned onto this one channel, so every message says which of the consumer's
/// slots it belongs to. That is what lets one thread gather a whole bundle from one receiver
/// instead of several threads racing several receivers into the same disperse.
pub(crate) type Slotted = (usize, ExitMessage);

/// Where a region sends what one of its exit nodes produces: the slot it fills at whichever
/// consumer reads that edge, and the channel tagged messages for that slot go out on.
pub(crate) type DownstreamLink = (usize, Sender<Slotted>);

/// A message from a disperse to the aggregate reading its assignments.
pub(crate) enum AggregateSignal {
    Assignment(Assignment),
    Finished,
}

/// Tracks how many units each of a region's workers currently holds, so a disperse can ask for
/// one without blocking and a caller can wait for the answer to change.
///
/// A worker is claimable while it holds fewer units than its capacity. Held means the worker's
/// aggregate has not yet drained the unit, not merely that the worker has sent it.
/// [WorkerFreedom::mark_free] is called from the aggregate thread for exactly that reason.
pub(crate) struct WorkerFreedom {
    /// How many units each worker holds that its aggregate has not drained yet.
    outstanding: Mutex<Vec<usize>>,
    capacity: usize,
    changed: Condvar,
}

impl WorkerFreedom {
    pub(crate) fn new(workers: usize, capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "a worker that may hold nothing can never be given work"
        );

        Self {
            outstanding: Mutex::new(vec![0; workers]),
            capacity,
            changed: Condvar::new(),
        }
    }

    fn workers(&self) -> usize {
        self.outstanding.lock().expect("not poisoned").len()
    }

    /// Claims the least loaded worker with room and charges it one unit, or answers `None` when
    /// every worker is full. Never blocks.
    ///
    /// Ties go to the lowest index. At capacity one this is exactly the first free worker, since
    /// every worker with room ties at zero held units. At a higher capacity it spreads claims
    /// across workers before stacking a second unit onto one that already has one.
    pub(crate) fn try_claim(&self) -> Option<usize> {
        let mut outstanding = self.outstanding.lock().expect("not poisoned");
        let worker = outstanding
            .iter()
            .enumerate()
            .filter(|&(_, held)| *held < self.capacity)
            .min_by_key(|&(_, held)| held)
            .map(|(worker, _)| worker)?;
        outstanding[worker] += 1;
        Some(worker)
    }

    pub(crate) fn mark_free(&self, worker: usize) {
        let mut outstanding = self.outstanding.lock().expect("not poisoned");
        outstanding[worker] = outstanding[worker].saturating_sub(1);
        self.changed.notify_all();
    }

    /// Wakes everyone waiting here, whether or not a worker actually freed up.
    ///
    /// Cancellation registers this, because a thread waiting on workers that have stopped has no
    /// other way out. The notify goes out under the same lock the wait holds, so a waiter that
    /// has checked its condition but not yet slept cannot miss it.
    fn wake_everyone(&self) {
        let held = self.outstanding.lock().expect("not poisoned");
        self.changed.notify_all();
        drop(held);
    }

    /// Blocks until at least one worker has room, or the run stops.
    ///
    /// A stop ends this wait as surely as a worker freeing up does, because a thread waiting on
    /// workers that have stopped has nothing else coming. The wait re-checks its condition every
    /// time it wakes, so a `mark_free` call landing between a failed claim and this call is never
    /// lost, and neither is a stop.
    fn wait_for_a_free_worker(&self, cancel: &Cancel) {
        let outstanding = self.outstanding.lock().expect("not poisoned");
        drop(
            self.changed
                .wait_while(outstanding, |outstanding| {
                    !cancel.is_stopped() && outstanding.iter().all(|held| *held >= self.capacity)
                })
                .expect("not poisoned"),
        );
    }
}

/// Hands a unit to whichever worker holds the fewest units its aggregate has not yet drained.
///
/// A worker stays claimable up to its capacity, so this can still hand out a unit to a worker
/// that is already holding one. Answers `None` rather than blocking when every worker is full,
/// which is exactly what [Disperse::place] needs to defer choosing without stalling the exchange
/// it drives.
struct FirstIdle {
    freedom: Arc<WorkerFreedom>,
}

impl Dispatch for FirstIdle {
    fn choose(&mut self, _unit: u64) -> Option<usize> {
        self.freedom.try_claim()
    }

    fn workers(&self) -> usize {
        self.freedom.workers()
    }
}

/// Wraps a source so the engine can tell, from outside, when it has produced its last frame.
///
/// `Source::next` only ever returns `None` once, per its own contract, so storing that once is
/// enough to know the source will never be called again.
struct WatchedSource {
    inner: Box<dyn Source>,
    exhausted: Arc<AtomicBool>,
}

impl StageInstance for WatchedSource {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl Source for WatchedSource {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        let frame = self.inner.next(context)?;
        if frame.is_none() {
            self.exhausted.store(true, Ordering::SeqCst);
        }

        Ok(frame)
    }
}

/// Wraps a sink so the engine can count what reached it, for [RunReport::frames].
struct CountingSink {
    inner: Box<dyn Sink>,
    frames: Arc<AtomicU64>,
}

impl StageInstance for CountingSink {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let mut slots = Vec::with_capacity(inputs.len());
        let mut seen = 0u64;

        for slot in 0..inputs.len() {
            let item = inputs.take(slot);
            if item.is_some() {
                seen += 1;
            }
            slots.push(item);
        }

        self.frames.fetch_add(seen, Ordering::SeqCst);
        self.inner.consume(context, Inputs::new(slots))
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Replaces the region head's source with a watched one, leaving every other stage alone.
fn wrap_head_source(nodes: Vec<(NodeId, StageBox)>, exhausted: Arc<AtomicBool>) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (id, stage))| {
            if index == 0
                && let StageBox::Source(inner) = stage
            {
                return (
                    id,
                    StageBox::Source(Box::new(WatchedSource {
                        inner,
                        exhausted: exhausted.clone(),
                    })),
                );
            }

            (id, stage)
        })
        .collect()
}

/// Replaces every sink in a region's stages with a counting one.
fn wrap_sinks(nodes: Vec<(NodeId, StageBox)>, frames: Arc<AtomicU64>) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .map(|(id, stage)| match stage {
            StageBox::Sink(inner) => (
                id,
                StageBox::Sink(Box::new(CountingSink {
                    inner,
                    frames: frames.clone(),
                })),
            ),
            other => (id, other),
        })
        .collect()
}

/// Builds one region's internal wiring from the graph it came from.
///
/// Every node after the head reads only from producers already inside the region, so each of its
/// edges names a producer in [Region::nodes] and the slot it fills.
fn build_region_edges(graph: &CompiledGraph, region: &Region) -> RegionEdges {
    let nodes = region.nodes();
    let feeds = nodes[1..]
        .iter()
        .map(|node_id| {
            graph
                .node(*node_id)
                .inputs()
                .iter()
                .map(|edge| Feed {
                    producer: edge.source(),
                    slot: edge.slot(),
                })
                .collect()
        })
        .collect();

    RegionEdges::new(feeds, region.exits().to_vec())
}

/// Which of a head's input slots are bound, sized to the widest slot any edge names.
fn bound_slots(graph: &CompiledGraph, head: NodeId) -> Vec<bool> {
    let edges = graph.node(head).inputs();
    let width = edges.iter().map(|edge| edge.slot() + 1).max().unwrap_or(0);
    let mut bound = vec![false; width];

    for edge in edges {
        bound[edge.slot()] = true;
    }

    bound
}

fn is_source_region(graph: &CompiledGraph, region: &Region) -> bool {
    graph.node(region.head()).kind() == StageKind::Source
}

/// Rejects the region shapes this engine cannot run, reporting each against the node that named
/// it rather than silently doing something other than what was asked.
///
/// A widened source has nothing to route its own output through, so nothing here can shard it
/// across workers without inventing an unspecified policy, and quietly running it at width one
/// would look exactly like the parallelism doing nothing. A region with more than one exit
/// needs one aggregate per exit to keep each of its output streams in order on its own, which
/// this engine does not build, so broadcasting every exit's output to every consumer would
/// silently duplicate frames instead. The last two shapes are ones the wiring below cannot make
/// sense of at all: a region that reads nothing and produces nothing, and a chunk routed disperse
/// whose authority names a slot no input was bound to. A frame routed disperse reads no chunk id,
/// so an authority it never looks at is left alone.
fn region_is_supported(graph: &CompiledGraph, region: &Region, diagnostics: &mut Diagnostics) -> bool {
    let mut supported = true;

    if !region.has_disperse()
        && !is_source_region(graph, region)
        && graph.node(region.head()).inputs().is_empty()
    {
        let head = graph.node(region.head());
        let message = format!(
            "the region headed by {} is not a source and reads no input, so nothing would ever run it",
            head.name(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    let routes_by_chunk = region.routing() == Routing::ByChunk;
    if routes_by_chunk && region.has_disperse() && !authority_is_bound(graph, region) {
        let head = graph.node(region.head());
        let message = format!(
            "{} routes its chunks by an input slot nothing was bound to",
            head.name(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    if is_source_region(graph, region) && region.width() > 1 {
        let node = graph.node(region.head());
        let message = format!(
            "{} asked for {} workers, but a source cannot yet be split across more than one",
            node.name(),
            region.width(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            format!("{}.{}", node.kind(), node.name()),
            node.span(),
            message,
        ));
        supported = false;
    }

    if region.exits().len() > 1 {
        let head = graph.node(region.head());
        let message = format!(
            "the region headed by {} has {} exits, and only one is supported per region",
            head.name(),
            region.exits().len(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    supported
}

/// Whether a region's chunk authority names one of the head's bound input slots.
///
/// [Disperse] routes every bundle by the item in that slot, so a slot outside the head's own
/// inputs, or one no edge fills, leaves it with nothing to read a chunk id from.
fn authority_is_bound(graph: &CompiledGraph, region: &Region) -> bool {
    let Some(authority) = region.chunk_authority() else {
        return false;
    };

    bound_slots(graph, region.head())
        .get(authority)
        .copied()
        .unwrap_or(false)
}

/// The largest stream item the solver resolved anywhere in the plan, in bytes.
///
/// Masks count as well as frames, because a stage allocating one charges it to the same budget a
/// frame is charged to, and a mask is not always the smaller of the two.
fn largest_item_bytes(formats: &SolvedFormats) -> usize {
    let mut largest = 0;

    for index in 0..formats.len() {
        let resolved = formats.get(NodeId::new(index));

        for input in resolved.inputs().iter().flatten() {
            largest = largest.max(input.total_bytes());
        }

        if let Some(output) = resolved.output() {
            largest = largest.max(output.total_bytes());
        }
    }

    largest
}

/// How many items a source may hold outstanding on top of the ones its own region holds.
pub(crate) const HEADROOM_ITEMS_PER_WORKER: usize = 4;

/// How many units a frame routed worker may hold that its aggregate has not drained.
///
/// Matched to the budget's per worker headroom, so a frame routed region at full stretch holds
/// what the budget was sized for rather than something it never counted.
pub(crate) const FRAME_UNITS_IN_FLIGHT: usize = HEADROOM_ITEMS_PER_WORKER;

/// How many nodes the longest source region runs.
///
/// A source region's nodes all share the source's one budgeted context, and
/// [RegionRunner::run] holds every one of their outputs until it returns, so this is how many
/// items that one share has to fit at once.
fn longest_source_region(graph: &CompiledGraph, regions: &[Region]) -> usize {
    regions
        .iter()
        .filter(|region| is_source_region(graph, region))
        .map(|region| region.nodes().len())
        .max()
        .unwrap_or(0)
}

/// The frame memory one source may hold outstanding at once, derived from the plan.
///
/// It covers two things. One item per node in the longest source region, so a region holding
/// every one of its outputs still fits, and four items per worker in the plan on top, so a
/// source can run ahead of the workers reading from it rather than waiting on each item to come
/// all the way back.
fn default_source_share(graph: &CompiledGraph, formats: &SolvedFormats, regions: &[Region]) -> usize {
    let total_workers: usize = regions.iter().map(Region::width).sum();
    let held_at_once = longest_source_region(graph, regions);
    let headroom = HEADROOM_ITEMS_PER_WORKER * total_workers;

    (held_at_once + headroom) * largest_item_bytes(formats)
}

/// One source's budget, holding `share` bytes and giving up on a wait that lasts `deadline`.
fn source_budget(share: usize, deadline: Option<Duration>) -> Arc<FrameBudget> {
    match deadline {
        Some(deadline) => FrameBudget::with_stall_deadline(share, deadline),
        None => FrameBudget::new(share),
    }
}

fn send_to_all(downstream: &[DownstreamLink], message: ExitMessage) {
    for (slot, sender) in downstream {
        sender.send((*slot, message.clone())).ok();
    }
}

fn forward(downstream: &[DownstreamLink], items: Vec<StreamItem>) {
    for item in items {
        send_to_all(downstream, ExitMessage::Item(item));
    }
}

fn send_done(downstream: &[DownstreamLink]) {
    send_to_all(downstream, ExitMessage::Done);
}

fn send_routed(work_senders: &[Sender<WorkItem>], routed: Vec<Routed>) {
    for Routed { worker, item } in routed {
        work_senders[worker].send(item).ok();
    }
}

/// A region with no disperse and no aggregate, headed by a source.
///
/// Nothing feeds it, so it drives itself until [WatchedSource] reports its head is exhausted.
struct SourceRegion {
    runner: RegionRunner,
    context: FrameContext,
    budget: Arc<FrameBudget>,
    exhausted: Arc<AtomicBool>,
    downstream: Vec<DownstreamLink>,
}

/// A region with no disperse and no aggregate, fed by exactly one upstream region.
///
/// The upstream region's own exchange, if it has one, has already put its output back in order,
/// so this region only has to carry it forward. Its one input still arrives tagged with the
/// slot it fills, the same as every other consumer, so it can size its bundle from the tag
/// rather than needing to remember which slot that is.
struct PassthroughRegion {
    runner: RegionRunner,
    context: FrameContext,
    upstream: Receiver<Slotted>,
    downstream: Vec<DownstreamLink>,
}

/// One worker of an [ExchangedRegion], before its thread is spawned.
pub(crate) struct WorkerSlot {
    pub(crate) runner: RegionRunner,
    pub(crate) context: FrameContext,
    pub(crate) work_receiver: Receiver<WorkItem>,
    pub(crate) result_sender: SyncSender<WorkResult>,
}

/// A region routed through a disperse, and merged back through an aggregate.
///
/// This covers every region wider than one worker, and a narrow region whose head pairs more
/// than one input. Either way the disperse decides how many workers matter, not the aggregate.
///
/// Every producer feeding this region's head shares one channel, each message tagged with the
/// slot it fills, so exactly one thread ever touches `disperse`. That alone only removes the
/// data race between producers: `Disperse` still accepts at most one pending item per slot
/// before raising `SlotRanAhead`, so the driving thread cannot simply offer each message the
/// instant it arrives. It keeps its own queue per slot in [drive_disperse], one item deep or
/// more, and only offers once every slot has something waiting. A slot fed by a faster producer
/// then queues up rather than panicking, and how far it may run ahead of a slower one is
/// bounded by the frame budget rather than capped at one.
struct ExchangedRegion {
    disperse: DisperseSide,
    freedom: Arc<WorkerFreedom>,
    aggregate: Option<AggregateSide>,
    result_receivers: Vec<Receiver<WorkResult>>,
    workers: Vec<WorkerSlot>,
    downstream: Vec<DownstreamLink>,
}

/// Everything the one thread merging a region's workers back together owns.
///
/// Only a region above one worker has one. A region of one worker has nothing to merge, so its
/// worker's output goes straight downstream instead, keeping the item indices it already carries.
pub(crate) struct AggregateSide {
    pub(crate) aggregate: Aggregate,
    pub(crate) assignment_receiver: Receiver<AggregateSignal>,
}

/// Everything the one thread driving a region's disperse owns.
struct DisperseSide {
    disperse: Disperse<FirstIdle>,
    upstream: Receiver<Slotted>,
    slots: Vec<usize>,
    freedom: Arc<WorkerFreedom>,
    work_senders: Vec<Sender<WorkItem>>,
    assignment_sender: Sender<AggregateSignal>,
    /// The slot the disperse routes every bundle by, for a region routed by chunk. A frame
    /// routed region reads no chunk id at all, so it has none.
    authority: Option<usize>,
    /// How many workers the region runs, for telling a genuine width-one join from one that
    /// simply has nothing else to spread work across.
    width: usize,
    /// The region's head node, named if a [RunWarning] about missing chunks fires.
    head: NodeId,
}

enum PreparedRegion {
    Source(SourceRegion),
    Passthrough(PassthroughRegion),
    Exchanged(ExchangedRegion),
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

/// The first failure a run's threads reported, and the signal that stops the others.
///
/// The error is recorded before the stop goes out, so a thread the stop wakes cannot report a
/// failure of its own that beats the one which caused it.
struct Outcome {
    cancel: Arc<Cancel>,
    error: Mutex<Option<RunError>>,
}

impl Outcome {
    fn report(&self, error: RunError) {
        {
            let mut first = self.error.lock().expect("not poisoned");
            if first.is_none() {
                *first = Some(error);
            }
        }

        self.cancel.stop();
    }

    fn take_error(&self) -> Option<RunError> {
        self.error.lock().expect("not poisoned").take()
    }
}

/// Spawns one of a run's threads, so whatever it ends in stops the rest of them.
///
/// A panic is caught here, on the thread it happened on, and reported as a failure of the run.
/// Letting it travel to the join instead would panic the caller part way through joining and
/// leave every thread after it unjoined.
fn spawn<Body>(handles: &mut Vec<JoinHandle<()>>, outcome: &Arc<Outcome>, body: Body)
where
    Body: FnOnce() -> Result<(), RunError> + Send + 'static,
{
    let outcome = outcome.clone();

    handles.push(thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(body));

        match result {
            Ok(Ok(())) => {},
            Ok(Err(error)) => outcome.report(error),
            Err(payload) => outcome.report(RunError::Panicked {
                message: panic_message(payload),
            }),
        }
    }));
}

/// Whatever a panic was raised with, as text.
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }

    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    "no message".to_string()
}

fn spawn_exchanged(
    region: ExchangedRegion,
    handles: &mut Vec<JoinHandle<()>>,
    outcome: &Arc<Outcome>,
    cancel: &Arc<Cancel>,
    dropped: &Arc<AtomicUsize>,
    warnings: &Arc<Mutex<Vec<RunWarning>>>,
) {
    let ExchangedRegion {
        disperse,
        freedom,
        aggregate,
        result_receivers,
        workers,
        downstream,
    } = region;

    let disperse_cancel = cancel.clone();
    let disperse_dropped = dropped.clone();
    let disperse_warnings = warnings.clone();
    spawn(handles, outcome, move || {
        drive_disperse(disperse, &disperse_cancel, &disperse_dropped, &disperse_warnings)
    });

    let merge_cancel = cancel.clone();
    spawn(handles, outcome, move || match aggregate {
        Some(side) => run_aggregate(side, result_receivers, &freedom, &merge_cancel, downstream),
        None => run_relay(result_receivers, &freedom, downstream),
    });

    for worker in workers {
        spawn(handles, outcome, move || run_worker(worker));
    }
}

fn run_source(region: SourceRegion, cancel: &Cancel) -> Result<(), RunError> {
    let SourceRegion {
        mut runner,
        mut context,
        budget,
        exhausted,
        downstream,
    } = region;

    runner.start(&mut context)?;

    loop {
        let produced = runner.run(&mut context, Vec::new())?;
        forward(&downstream, produced);
        check_budget(&budget)?;

        if exhausted.load(Ordering::SeqCst) || cancel.is_stopped() {
            break;
        }
    }

    let flushed = runner.end(&mut context)?;
    forward(&downstream, flushed);
    check_budget(&budget)?;
    send_done(&downstream);

    Ok(())
}

/// Fails the run when `budget` gave up waiting for room.
///
/// A source only ever gets room back when a frame it emitted earlier is dropped, so a wait that
/// lasted the whole deadline is one nothing was ever going to end. The run stops naming the limit
/// and what was outstanding, rather than sitting there.
fn check_budget(budget: &FrameBudget) -> Result<(), RunError> {
    match budget.stalled() {
        Some(outstanding) => Err(RunError::BudgetStalled {
            limit: budget.limit(),
            outstanding,
        }),
        None => Ok(()),
    }
}

/// Runs a region that only carries one upstream's output forward.
///
/// Nothing here reads the stop flag. Its one wait is a receive, which ends as soon as the region
/// feeding it stops and drops its sender, and walking out on the flag instead would throw away
/// whatever had already been handed over without anything accounting for it.
fn run_passthrough(region: PassthroughRegion) -> Result<(), RunError> {
    let PassthroughRegion {
        mut runner,
        mut context,
        upstream,
        downstream,
    } = region;

    runner.start(&mut context)?;

    while let Ok((slot, ExitMessage::Item(item))) = upstream.recv() {
        let mut bundle = vec![None; slot + 1];
        bundle[slot] = Some(item);

        let produced = runner.run(&mut context, bundle)?;
        forward(&downstream, produced);
    }

    let flushed = runner.end(&mut context)?;
    forward(&downstream, flushed);
    send_done(&downstream);

    Ok(())
}

/// Drives a region's disperse from its one merged, slot-tagged upstream channel.
///
/// `Disperse::offer` accepts at most one pending item per slot before raising `SlotRanAhead`, so
/// this cannot offer a message the instant it arrives: a slot fed by a faster producer would run
/// straight into that the moment the two producers ran at different speeds. Instead this keeps
/// its own queue per bound slot, pushes every arrival onto the queue for the slot it names, and
/// only offers once every slot's queue has something waiting. It then takes one item from each
/// slot's queue, in slot order, offering each in turn to complete one whole bundle before
/// checking again, and while [Disperse::is_holding] loops [Disperse::place], sending what that
/// returns and retrying when it answers `None`. A fast slot's queue simply grows rather than
/// panicking, so how far it may run ahead of a slow one is bounded by the frame budget rather
/// than capped at one item.
///
/// Owning the disperse outright, on one thread, is what keeps the state machine's own
/// lock-step assumption true: `Disperse` assumes whoever calls `offer` and `place` does so in
/// one sequence, and a second thread offering into the same instance from an independent
/// upstream would have no ordering against this one at all.
///
/// Waiting for a free worker is the one place here a stop has to reach by hand, since every
/// other wait is a receive that ends when its senders do. Once it gives up waiting, whatever is
/// still queued or still held has missed its run, so it is counted into `dropped` rather than
/// going quietly.
///
/// This is also where [RunWarning::DetectorOnWrongBranch] and [RunWarning::NothingChunked] are
/// noticed. Every slot's item is visible here before it is offered, which is the one place both
/// the authority's chunk id and every other slot's can be read from the same bundle. A region
/// with only one slot never runs the branch that could mark another slot chunked, since there is
/// no other slot to read. A frame routed region routes by no slot's chunk id, so nothing there
/// is observed at all, and no warning follows from an authority that was never there to begin
/// with.
fn drive_disperse(
    side: DisperseSide,
    cancel: &Cancel,
    dropped: &AtomicUsize,
    warnings: &Mutex<Vec<RunWarning>>,
) -> Result<(), RunError> {
    let DisperseSide {
        mut disperse,
        upstream,
        slots,
        freedom,
        work_senders,
        assignment_sender,
        authority,
        width,
        head,
    } = side;

    let mut queues: HashMap<usize, VecDeque<StreamItem>> =
        slots.iter().map(|&slot| (slot, VecDeque::new())).collect();
    let mut done: HashSet<usize> = HashSet::new();
    let mut authority_chunked = false;
    let mut other_chunked = false;

    'routing: loop {
        if cancel.is_stopped() {
            break;
        }

        while slots.iter().all(|slot| !queues[slot].is_empty()) {
            for &slot in &slots {
                let item = queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .pop_front()
                    .expect("this slot's queue was just confirmed non-empty");

                if let Some(authority) = authority {
                    note_chunk(
                        slot,
                        authority,
                        item.meta().chunk,
                        &mut authority_chunked,
                        &mut other_chunked,
                    );
                }

                let offered = disperse.offer(slot, item);
                let routed = offered.map_err(refused_offer)?;
                send_routed(&work_senders, routed);

                let placed =
                    drain_holding(&mut disperse, &freedom, cancel, &work_senders, &assignment_sender);
                if !placed {
                    break 'routing;
                }
            }
        }

        if done.len() == slots.len() {
            break;
        }

        match upstream.recv() {
            Ok((slot, ExitMessage::Item(item))) => {
                queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .push_back(item);
            },
            Ok((slot, ExitMessage::Done)) => {
                done.insert(slot);
            },
            Err(_) => break,
        }
    }

    let finished = disperse.finish();
    for assignment in finished.assignments {
        assignment_sender
            .send(AggregateSignal::Assignment(assignment))
            .ok();
    }

    send_routed(&work_senders, finished.routed);
    assignment_sender.send(AggregateSignal::Finished).ok();

    let queued: usize = queues.values().map(VecDeque::len).sum();
    dropped.fetch_add(queued + dropped_items(finished.dropped), Ordering::SeqCst);

    if width > 1 && authority.is_some() && !authority_chunked && !cancel.is_stopped() {
        let warning = if other_chunked {
            RunWarning::DetectorOnWrongBranch { node: head }
        } else {
            RunWarning::NothingChunked { node: head }
        };

        warnings.lock().expect("not poisoned").push(warning);
    }

    Ok(())
}

/// Folds one slot's chunk id into whether the authority, or some other slot, has been seen
/// carrying a chunk above zero anywhere in the run so far.
fn note_chunk(
    slot: usize,
    authority: usize,
    chunk: u64,
    authority_chunked: &mut bool,
    other_chunked: &mut bool,
) {
    if chunk == 0 {
        return;
    }

    if slot == authority {
        *authority_chunked = true;
    } else {
        *other_chunked = true;
    }
}

/// What a refused offer means for the run.
///
/// Two inputs carrying different frames, and a chunk id going backwards, are things the data did,
/// and both fail the run. Every other refusal is about how the disperse was driven, and this
/// driver gathers a whole bundle before it offers any of it, so none of them can happen here.
fn refused_offer(error: ExchangeError) -> RunError {
    match error {
        ExchangeError::OutOfStep { .. } | ExchangeError::ChunkWentBackwards { .. } => {
            RunError::Exchange { source: error }
        },
        other => panic!("a slot only offers once every slot's queue has an item ready: {other}"),
    }
}

/// How many items a bundle the stream ended on was carrying.
fn dropped_items(dropped: Option<Dropped>) -> usize {
    match dropped {
        Some(Dropped::Partial(bundle) | Dropped::Unplaced(bundle)) => {
            bundle.iter().filter(|slot| slot.is_some()).count()
        },
        None => 0,
    }
}

/// Places whatever the disperse is holding, waiting for a worker to free up when it has to.
///
/// Answers false when the run stopped, leaving the bundle held, which tells its caller to stop
/// offering rather than run into a placement that is never coming. The stop is read before every
/// placement rather than only where this waits, so a stopped run never starts one more chunk off
/// on a worker that happens to free up as everything is winding down.
fn drain_holding(
    disperse: &mut Disperse<FirstIdle>,
    freedom: &WorkerFreedom,
    cancel: &Cancel,
    work_senders: &[Sender<WorkItem>],
    assignment_sender: &Sender<AggregateSignal>,
) -> bool {
    while disperse.is_holding() {
        if cancel.is_stopped() {
            return false;
        }

        match disperse.place() {
            Some(placed) => {
                assignment_sender
                    .send(AggregateSignal::Assignment(placed.assignment))
                    .ok();
                send_routed(work_senders, vec![placed.routed]);
            },
            None => freedom.wait_for_a_free_worker(cancel),
        }
    }

    true
}

/// Runs one worker of an exchanged region, one unit at a time.
///
/// Every unit after the first is started by its own first bundle arriving rather than by the
/// unit before it closing, so a worker whose stream ends on a unit close never leaves its
/// stages holding a start they will never see the end of.
///
/// Nothing here reads the stop flag. Everything a worker waits on is a channel: its disperse
/// closes the work channel when it winds down, and its aggregate dropping the other end closes
/// the results one. Walking out part way through a unit instead would leave the aggregate
/// waiting on a close that never comes.
///
/// An aggregate that has already gone ends this the same way an `EndOfStream` does, by leaving
/// the loop, so the flush below still runs. Returning from inside the loop instead would leave
/// every stage in this worker holding a start it never sees the end of.
pub(crate) fn run_worker(worker: WorkerSlot) -> Result<(), RunError> {
    let WorkerSlot {
        mut runner,
        mut context,
        work_receiver,
        result_sender,
    } = worker;

    runner.start(&mut context)?;
    let mut started = true;
    let mut chunk_in_hand = None;

    loop {
        match work_receiver.recv() {
            Ok(WorkItem::Bundle(bundle)) => {
                if !started {
                    runner.start(&mut context)?;
                    started = true;
                }

                chunk_in_hand = bundle_chunk(&bundle);

                let run = runner.run(&mut context, bundle);
                let produced = run.map_err(|error| named_chunk(error, chunk_in_hand))?;
                if send_results(&result_sender, produced).is_err() {
                    break;
                }
            },
            Ok(WorkItem::Unit(unit, bundle)) => {
                if !started {
                    runner.start(&mut context)?;
                    started = true;
                }

                chunk_in_hand = bundle_chunk(&bundle);

                let run = runner.run(&mut context, bundle);
                let produced = run.map_err(|error| named_chunk(error, chunk_in_hand))?;
                if send_results(&result_sender, produced).is_err() {
                    break;
                }

                if result_sender.send(WorkResult::EndOfUnit(unit)).is_err() {
                    break;
                }
            },
            Ok(WorkItem::EndOfChunk(chunk)) => {
                chunk_in_hand = Some(chunk);

                if started {
                    let end = runner.end(&mut context);
                    let flushed = end.map_err(|error| error.in_chunk(chunk))?;
                    started = false;

                    if send_results(&result_sender, flushed).is_err() {
                        break;
                    }
                }

                if result_sender.send(WorkResult::EndOfUnit(chunk)).is_err() {
                    break;
                }
            },
            Ok(WorkItem::EndOfStream) | Err(_) => break,
        }
    }

    if started {
        let end = runner.end(&mut context);
        let flushed = end.map_err(|error| named_chunk(error, chunk_in_hand))?;
        send_results(&result_sender, flushed).ok();
    }

    result_sender.send(WorkResult::EndOfStream).ok();
    Ok(())
}

/// The chunk a bundle belongs to, taken from the first slot carrying an item.
fn bundle_chunk(bundle: &[Option<StreamItem>]) -> Option<u64> {
    let item = bundle.iter().flatten().next()?;

    Some(item.meta().chunk)
}

/// `error`, said to have come from `chunk` when there is one to name.
fn named_chunk(error: RunError, chunk: Option<u64>) -> RunError {
    match chunk {
        Some(chunk) => error.in_chunk(chunk),
        None => error,
    }
}

fn send_results(result_sender: &SyncSender<WorkResult>, items: Vec<StreamItem>) -> Result<(), ()> {
    for item in items {
        result_sender.send(WorkResult::Item(item)).map_err(|_| ())?;
    }

    Ok(())
}

/// Carries one worker's output straight downstream, for a region with no aggregate.
///
/// A region of one worker has nothing to merge back into order, so nothing here renumbers what
/// it produced. Marking the worker free as each unit closes is what lets its disperse hand it
/// the next one, and is the one job this shares with an aggregate.
fn run_relay(
    result_receivers: Vec<Receiver<WorkResult>>,
    freedom: &WorkerFreedom,
    downstream: Vec<DownstreamLink>,
) -> Result<(), RunError> {
    let results = result_receivers
        .into_iter()
        .next()
        .expect("a region without an aggregate runs exactly one worker");

    while let Ok(result) = results.recv() {
        match result {
            WorkResult::Item(item) => forward(&downstream, vec![item]),
            WorkResult::EndOfUnit(_) => freedom.mark_free(0),
            WorkResult::EndOfStream => break,
        }
    }

    send_done(&downstream);
    Ok(())
}

/// Merges what a region's workers produced back into one stream, in assignment order.
///
/// A worker that ends still owing a chunk is only tolerated once the run has stopped, which means
/// something else has already failed and reported it. On a run that is still going it means the
/// exchange itself has lost a chunk, and that is reported rather than quietly finishing with
/// frames missing.
pub(crate) fn run_aggregate(
    side: AggregateSide,
    result_receivers: Vec<Receiver<WorkResult>>,
    freedom: &WorkerFreedom,
    cancel: &Cancel,
    downstream: Vec<DownstreamLink>,
) -> Result<(), RunError> {
    let AggregateSide {
        mut aggregate,
        assignment_receiver,
    } = side;

    loop {
        match aggregate.next_worker() {
            Some(worker) => {
                let Ok(result) = result_receivers[worker].recv() else {
                    break;
                };
                let is_end_of_unit = matches!(result, WorkResult::EndOfUnit(_));

                let accepted = aggregate.accept(worker, result);
                let emitted = match accepted {
                    Ok(emitted) => emitted,
                    // A worker that ends still owing a chunk has either failed or been stopped,
                    // and its own thread reports that. Nothing is left to merge in order, so this
                    // winds down rather than raising a second failure derived from the first.
                    Err(ExchangeError::WorkerEndedOwingUnit { .. }) if cancel.is_stopped() => break,
                    // The stop a failure raises is recorded a moment after the thread that failed
                    // has dropped its senders, so the arm above can lose that race and land here
                    // instead. Reporting rather than panicking leaves the real cause the first
                    // error of the run and this one behind it.
                    Err(error) => return Err(RunError::Exchange { source: error }),
                };

                if is_end_of_unit {
                    freedom.mark_free(worker);
                }

                forward(&downstream, emitted);
            },
            None => {
                if aggregate.is_finished() {
                    break;
                }

                match assignment_receiver.recv() {
                    Ok(AggregateSignal::Assignment(assignment)) => aggregate.assign(assignment),
                    Ok(AggregateSignal::Finished) | Err(_) => aggregate.finish(),
                }
            },
        }
    }

    send_done(&downstream);
    Ok(())
}
