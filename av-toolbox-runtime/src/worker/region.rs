use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use av_toolbox_core::StageError;
use av_toolbox_gpu::{AnyItem, ChunkRoute, FrameContext, MetaRecord, StreamItem, StreamPolicy};
use av_toolbox_graph::{NodeId, StageBox};

use super::node::{deliver_record, end_node, run_node, start_node};
use crate::artefacts::ArtefactCollector;
use crate::chunks::chunk_in;
use crate::flow::Sightings;
use crate::profile::{Counted, Profiler};
use crate::{Flow, RunError};

const STRAY_RECORDS: &str =
    "records can only be made while a stage produces, processes, selects, consumes or ends a stream";

/// One node's output filling one of another node's input slots.
#[derive(Debug, Clone, Copy)]
pub struct Feed {
    pub producer: NodeId,
    pub slot: usize,
}

/// How a region's nodes feed each other.
///
/// `feeds` holds one entry per node after the head, in the same order those nodes appear in
/// [RegionRunner]'s own node list. Each entry lists which producer fills which of that node's
/// input slots. `exits` names the nodes whose output leaves the region. `paths` names each stage
/// node, so what it emits is filed under that name. `chunked` names the nodes given one chunk at a
/// time, which are restarted whenever their inputs move to a new one. `routes` names the layer
/// each node reads its chunk from, and how. `authorities` names, per node, the input slot its chunk is read
/// from, falling back to its lowest filled slot. `frame_routed` marks a region whose output is
/// returned per input frame, so a filter returning no frames is an error.
#[derive(Debug)]
pub struct RegionEdges {
    feeds: Vec<Vec<Feed>>,
    exits: Vec<NodeId>,
    paths: HashMap<NodeId, Arc<str>>,
    chunked: HashSet<NodeId>,
    routes: HashMap<NodeId, ChunkRoute>,
    authorities: HashMap<NodeId, usize>,
    frame_routed: bool,
}

impl RegionEdges {
    pub fn new(feeds: Vec<Vec<Feed>>, exits: Vec<NodeId>) -> Self {
        Self {
            feeds,
            exits,
            paths: HashMap::new(),
            chunked: HashSet::new(),
            routes: HashMap::new(),
            authorities: HashMap::new(),
            frame_routed: false,
        }
    }

    /// The same edges, filing what each named node emits under its path.
    pub fn with_paths(mut self, paths: HashMap<NodeId, Arc<str>>) -> Self {
        self.paths = paths;
        self
    }

    /// The same edges, restarting each of `chunked` whenever its inputs move to a new chunk.
    pub fn with_chunked(mut self, chunked: HashSet<NodeId>) -> Self {
        self.chunked = chunked;
        self
    }

    /// The same edges, reading each node's chunk from the route `routes` names for it.
    pub fn with_routes(mut self, routes: HashMap<NodeId, ChunkRoute>) -> Self {
        self.routes = routes;
        self
    }

    /// The same edges, reading each node's chunk from the input slot `authorities` names for it.
    pub fn with_authorities(mut self, authorities: HashMap<NodeId, usize>) -> Self {
        self.authorities = authorities;
        self
    }

    /// The same edges, for a region whose frames are each handed back on their own.
    pub fn frame_routed(mut self) -> Self {
        self.frame_routed = true;
        self
    }
}

/// Runs one region's nodes, in topological order, on one thread.
///
/// The head is the only node whose inputs come from outside the region, so [RegionRunner::run]
/// hands it the bundle directly. Every other node's inputs are assembled from other nodes already
/// run in this same list, following the region's edges.
///
/// Every node after the head keeps one queue per feed, holding what its producers emitted until
/// every feed has an item ready, so a node whose producers lag each other still sees matching frames.
///
/// A record keeps its place among the items around it. A node passes one on once it has taken every
/// item that arrived ahead of it, and a meta sink reads it instead of passing it on.
pub struct RegionRunner {
    nodes: Vec<(NodeId, StageBox)>,
    edges: RegionEdges,
    positions: HashMap<NodeId, usize>,
    inboxes: Vec<Inbox>,
    offered: Offered,
    last_chunks: Vec<Option<u64>>,
    route: Option<ChunkRoute>,
    profiler: Option<Arc<Profiler>>,
    artefacts: Option<Arc<ArtefactCollector>>,
}

impl RegionRunner {
    pub fn new(nodes: Vec<(NodeId, StageBox)>, edges: RegionEdges) -> Self {
        let positions = nodes
            .iter()
            .enumerate()
            .map(|(position, (id, _))| (*id, position))
            .collect();
        let inboxes = edges.feeds.iter().map(|feeds| Inbox::new(feeds.len())).collect();
        let offered = Offered::fed_by(1);

        let last_chunks = vec![None; nodes.len()];

        Self {
            nodes,
            edges,
            positions,
            inboxes,
            offered,
            last_chunks,
            route: None,
            profiler: None,
            artefacts: None,
        }
    }

    /// The route the region was planned to route its chunks on.
    pub(crate) fn route(&self) -> Option<ChunkRoute> {
        self.route
    }

    /// The same runner, routing its chunks on `route`.
    pub(crate) fn routed_on(mut self, route: Option<ChunkRoute>) -> Self {
        self.route = route;
        self
    }

    /// The same runner, offered records from outside by `links` upstream edges.
    ///
    /// Only a region fed by several links checks what it is offered for repeats, since one link never
    /// carries a record twice. A region is fed by one link unless told otherwise.
    pub(crate) fn fed_by(mut self, links: usize) -> Self {
        self.offered = Offered::fed_by(links);
        self
    }

    /// The same runner, timing every stage call through `profiler` when there is one.
    pub(crate) fn profiled(mut self, profiler: Option<Arc<Profiler>>) -> Self {
        self.profiler = profiler;
        self
    }

    /// The same runner, handing what its nodes build to `artefacts`.
    pub(crate) fn collecting(mut self, artefacts: Arc<ArtefactCollector>) -> Self {
        self.artefacts = Some(artefacts);
        self
    }

    /// Signals the start of a chunk to every node in the region.
    pub fn start(&mut self, context: &mut FrameContext) -> Result<(), RunError> {
        self.last_chunks.fill(None);

        for (id, stage) in &mut self.nodes {
            start_node(*id, stage, context)?;
            refuse_stray_records(*id, context)?;
        }

        Ok(())
    }

    /// Holds records that reached the region from outside.
    ///
    /// They leave through the head, ahead of its next output, or after its end flush. A region fed by
    /// several links leaves out any record it was already offered since it last ended.
    pub fn offer_records(&mut self, records: Vec<MetaRecord>) {
        for record in records {
            self.offered.offer(record);
        }
    }

    /// Runs the region once over `bundle`, the head's inputs, returning what its exits produced.
    ///
    /// The step ends by flushing this thread's queued work, before anything it read is let go or
    /// anything it produced is returned. Under [StreamPolicy::FlushAtHandOff] the device items in
    /// `bundle` are held until then, so a head dropping its inputs early cannot release them first.
    /// Records offered to the region leave ahead of the head's output, behind anything the head flushed
    /// for a chunk it restarted on.
    pub fn run(
        &mut self,
        context: &mut FrameContext,
        bundle: Vec<Option<AnyItem>>,
    ) -> Result<Vec<Flow>, RunError> {
        let mut outputs: Vec<Vec<Flow>> = Vec::with_capacity(self.nodes.len());
        let held_inputs = hold_until_flushed(context, &bundle);

        let offered = self.offered.take();
        let mut head_watch = NodeWatch {
            chunked: self.edges.chunked.contains(&self.nodes[0].0),
            frame_routed: self.edges.frame_routed,
            authority: self.edges.authorities.get(&self.nodes[0].0).copied(),
            route: self.edges.routes.get(&self.nodes[0].0).copied(),
            last: &mut self.last_chunks[0],
            artefacts: self.artefacts.as_deref(),
            profiler: self.profiler.as_deref(),
        };
        let mut head_output = step_node(&mut self.nodes[0], &mut head_watch, context, bundle, offered)?;
        stamp(&self.edges.paths, self.nodes[0].0, &mut head_output);
        outputs.push(head_output);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let inbox = &mut self.inboxes[index - 1];
            enqueue(feeds, inbox, &outputs, &self.positions);

            let node = self.nodes[index].0;
            let mut watch = NodeWatch {
                chunked: self.edges.chunked.contains(&node),
                frame_routed: self.edges.frame_routed,
                authority: self.edges.authorities.get(&node).copied(),
                route: self.edges.routes.get(&node).copied(),
                last: &mut self.last_chunks[index],
                artefacts: self.artefacts.as_deref(),
                profiler: self.profiler.as_deref(),
            };
            let mut produced = run_fed(feeds, inbox, &mut self.nodes[index], &mut watch, context)?;
            stamp(&self.edges.paths, node, &mut produced);
            outputs.push(produced);
        }

        let exits = collect_exits(&self.edges.exits, &self.positions, &outputs);

        // This thread queued the step's GPU work, so it submits it before the step's items move on.
        context.flush();
        drop(held_inputs);

        Ok(exits)
    }

    /// Flushes every node in topological order, threading each flush through its consumers.
    ///
    /// A node downstream of the flushing one is run on whatever the upstream flush produced,
    /// exactly like a normal step, before its own flush is taken. That way anything the
    /// downstream node ends up holding because of the upstream flush is captured too. It ends by
    /// submitting this thread's queued work, before anything it produced is returned. Every record
    /// still waiting in the region leaves after the end flush of the node it is waiting at. A meta sink
    /// reads the records waiting at it before it ends.
    ///
    /// `ended` is the chunk being ended, when the caller knows it. A node files its artefacts under the
    /// chunk it last ran on, or under `ended` when it has none. In a region routed by frame, a node not
    /// given one chunk at a time files them under no chunk instead. A node given one chunk at a time
    /// that is left with no chunk has its artefacts discarded.
    pub fn end(&mut self, context: &mut FrameContext, ended: Option<u64>) -> Result<Vec<Flow>, RunError> {
        let mut outputs: Vec<Vec<Flow>> = Vec::with_capacity(self.nodes.len());

        let head_chunked = self.edges.chunked.contains(&self.nodes[0].0);
        let head_tagging = Tagging {
            chunked: head_chunked,
            frame_routed: self.edges.frame_routed,
        };
        let head_chunk = ending_chunk(head_tagging, self.last_chunks[0], ended);
        let head_watch = EndWatch {
            chunk: head_chunk,
            discarding: head_chunked && head_chunk.is_none(),
            artefacts: self.artefacts.as_deref(),
            profiler: self.profiler.as_deref(),
        };
        let (head_id, head_stage) = &mut self.nodes[0];
        let offered = self.offered.drain();
        let mut head_flush = end_owing(*head_id, head_stage, context, head_watch, offered)?;
        stamp(&self.edges.paths, *head_id, &mut head_flush);
        outputs.push(head_flush);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let inbox = &mut self.inboxes[index - 1];
            enqueue(feeds, inbox, &outputs, &self.positions);

            let node = self.nodes[index].0;
            let watch = NodeWatch {
                chunked: self.edges.chunked.contains(&node),
                frame_routed: self.edges.frame_routed,
                authority: self.edges.authorities.get(&node).copied(),
                route: self.edges.routes.get(&node).copied(),
                last: &mut self.last_chunks[index],
                artefacts: self.artefacts.as_deref(),
                profiler: self.profiler.as_deref(),
            };
            let mut flushed = end_fed(feeds, inbox, &mut self.nodes[index], watch, ended, context)?;
            stamp(&self.edges.paths, node, &mut flushed);
            outputs.push(flushed);
        }

        let exits = collect_exits(&self.edges.exits, &self.positions, &outputs);

        context.flush();

        Ok(exits)
    }
}

/// Records offered to a region from outside, waiting to leave through its head.
///
/// `seen` names the records offered since the region last ended, so one arriving again by another link
/// is dropped.
struct Offered {
    waiting: VecDeque<MetaRecord>,
    seen: Sightings,
}

impl Offered {
    fn fed_by(links: usize) -> Self {
        Self {
            waiting: VecDeque::new(),
            seen: Sightings::for_routes(links),
        }
    }

    fn offer(&mut self, record: MetaRecord) {
        let first_sighting = self.seen.first(&record);
        if first_sighting {
            self.waiting.push_back(record);
        }
    }

    fn take(&mut self) -> Vec<MetaRecord> {
        self.waiting.drain(..).collect()
    }

    /// Takes every waiting record, and forgets every record seen, as the region ends.
    fn drain(&mut self) -> Vec<MetaRecord> {
        self.seen.forget();

        self.take()
    }
}

/// What one node after the head is waiting on.
///
/// `queues` holds one queue of items per feed. `waiting` holds each record that reached the node beside
/// how many bundles the node must have taken before it leaves, which is every item that arrived ahead
/// of it on its feed. `seen` names every record that reached a node with several feeds, so one arriving by
/// a second feed is dropped.
struct Inbox {
    queues: Vec<VecDeque<AnyItem>>,
    waiting: VecDeque<(u64, MetaRecord)>,
    seen: Sightings,
    consumed: u64,
}

impl Inbox {
    fn new(feeds: usize) -> Self {
        Self {
            queues: vec![VecDeque::new(); feeds],
            waiting: VecDeque::new(),
            seen: Sightings::for_routes(feeds),
            consumed: 0,
        }
    }

    /// Places `flow`, arriving on feed `feed`, behind everything that feed already holds.
    fn receive(&mut self, feed: usize, flow: &Flow) {
        let queue = &mut self.queues[feed];

        match flow {
            Flow::Item(item) => queue.push_back(item.clone()),
            Flow::Record(record) => {
                let first_sighting = self.seen.first(record);
                if !first_sighting {
                    return;
                }

                let threshold = self.consumed + queue.len() as u64;
                self.waiting.push_back((threshold, record.clone()));
            },
        }
    }

    /// Takes every waiting record whose items ahead of it have all been taken, in the order they arrived.
    fn due(&mut self) -> Vec<MetaRecord> {
        let mut due = Vec::new();
        let mut later = VecDeque::with_capacity(self.waiting.len());

        for (threshold, record) in self.waiting.drain(..) {
            if threshold <= self.consumed {
                due.push(record);
            } else {
                later.push_back((threshold, record));
            }
        }

        self.waiting = later;
        due
    }

    /// Takes every waiting record, and forgets every record seen, once nothing more can reach the node.
    fn drain(&mut self) -> Vec<MetaRecord> {
        self.seen.forget();

        self.waiting.drain(..).map(|(_, record)| record).collect()
    }
}

/// Hands each of `records` on from `node`, in order.
///
/// A meta sink reads each one itself. Any other node passes it on at the end of `produced`.
fn release(
    node: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    records: Vec<MetaRecord>,
    produced: &mut Vec<Flow>,
) -> Result<(), RunError> {
    for record in records {
        let delivered = deliver_record(node, stage, context, &record)?;
        refuse_stray_records(node, context)?;

        if !delivered {
            produced.push(Flow::Record(record));
        }
    }

    Ok(())
}

/// The records a node made during its last call, checked and ready to place after that call's output.
fn taken_records(node: NodeId, context: &mut FrameContext) -> Result<Vec<Flow>, RunError> {
    let records = context.take_records();
    let mut flows = Vec::with_capacity(records.len());

    for record in records {
        if let Err(reason) = MetaRecord::check_kind(record.kind()) {
            return Err(malformed_kind(node, reason));
        }

        flows.push(Flow::Record(record));
    }

    Ok(flows)
}

/// Fails naming `node` when it left records on the context during a call whose records are not taken.
fn refuse_stray_records(node: NodeId, context: &mut FrameContext) -> Result<(), RunError> {
    let stray = context.take_records();
    if stray.is_empty() {
        return Ok(());
    }

    Err(stray_records(node))
}

fn stray_records(node: NodeId) -> RunError {
    let source = StageError::msg(STRAY_RECORDS);

    RunError::Stage { node, source }
}

fn malformed_kind(node: NodeId, reason: String) -> RunError {
    let source = StageError::msg(reason);

    RunError::Stage { node, source }
}

/// Copies of the device items in `bundle`, to keep alive until the step's work is flushed.
///
/// Only [StreamPolicy::FlushAtHandOff] needs them. Under [StreamPolicy::SharedStream] nothing is
/// held, so a head that owns its inputs outright can still write into them.
fn hold_until_flushed(context: &FrameContext, bundle: &[Option<AnyItem>]) -> Vec<StreamItem> {
    if context.stream_policy() == StreamPolicy::SharedStream {
        return Vec::new();
    }

    bundle.iter().flatten().filter_map(device_item).collect()
}

fn device_item(item: &AnyItem) -> Option<StreamItem> {
    match item {
        AnyItem::Device(device) => Some(device.clone()),
        AnyItem::Host(_) => None,
    }
}

/// What a node's runs are checked against.
struct NodeWatch<'a> {
    chunked: bool,
    frame_routed: bool,
    authority: Option<usize>,
    route: Option<ChunkRoute>,
    last: &'a mut Option<u64>,
    artefacts: Option<&'a ArtefactCollector>,
    profiler: Option<&'a Profiler>,
}

/// Runs one node, timed when the run is profiled.
fn call_node(
    id: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
    profiler: Option<&Profiler>,
) -> Result<Vec<AnyItem>, RunError> {
    let Some(profiler) = profiler else {
        return run_node(id, stage, context, slots);
    };

    let counted = counted_run(stage);

    profiler.time(id, counted, context, |context| {
        run_node(id, stage, context, slots)
    })
}

/// Whether one run of `stage` counts as a frame. A source run that produced nothing does not.
fn counted_run(stage: &StageBox) -> Counted {
    match stage {
        StageBox::Source(_) | StageBox::HostSource(_) => Counted::FrameIfProduced,
        _ => Counted::Frame,
    }
}

/// What ending a node needs beyond the node itself.
#[derive(Copy, Clone)]
struct EndWatch<'a> {
    /// The chunk being ended, or `None` when the node files its artefacts under no chunk.
    chunk: Option<u64>,
    /// Drops what the node built rather than collecting it, for a chunked node that saw no chunk.
    discarding: bool,
    artefacts: Option<&'a ArtefactCollector>,
    profiler: Option<&'a Profiler>,
}

/// What decides the chunk a node's artefacts are filed under as it ends.
#[derive(Copy, Clone)]
struct Tagging {
    chunked: bool,
    frame_routed: bool,
}

/// The chunk a node's artefacts are filed under as it ends.
///
/// The node uses the chunk it last ran on, or `ended` when it ran on none. A node in a region routed
/// by frame that is not given one chunk at a time keeps `last` instead, since `ended` there names a
/// unit of work, not a chunk.
fn ending_chunk(tagging: Tagging, last: Option<u64>, ended: Option<u64>) -> Option<u64> {
    if tagging.chunked || !tagging.frame_routed {
        return last.or(ended);
    }

    last
}

/// Ends one node, handing on `owed`, the records still waiting at it.
///
/// A meta sink reads them before it ends, since it has no end flush for them to follow. Any other node
/// passes them on after its end flush.
fn end_owing(
    id: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    watch: EndWatch<'_>,
    owed: Vec<MetaRecord>,
) -> Result<Vec<Flow>, RunError> {
    let reads_records = matches!(stage, StageBox::MetaSink(_));

    if reads_records {
        let mut produced = Vec::new();
        release(id, stage, context, owed, &mut produced)?;

        let mut flushed = call_end(id, stage, context, watch)?;
        produced.append(&mut flushed);

        return Ok(produced);
    }

    let mut flushed = call_end(id, stage, context, watch)?;
    release(id, stage, context, owed, &mut flushed)?;

    Ok(flushed)
}

/// Ends one node, placing the records it made after what it flushed.
fn call_end(
    id: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    watch: EndWatch<'_>,
) -> Result<Vec<Flow>, RunError> {
    let flushed = timed_end(id, stage, context, watch)?;
    let records = taken_records(id, context)?;

    Ok(followed_by(flushed, records))
}

/// `items` as flows, followed by `records`.
fn followed_by(items: Vec<AnyItem>, records: Vec<Flow>) -> Vec<Flow> {
    let mut flows: Vec<Flow> = items.into_iter().map(Flow::Item).collect();
    flows.extend(records);

    flows
}

/// Ends one node, timed when the run is profiled.
fn timed_end(
    id: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    watch: EndWatch<'_>,
) -> Result<Vec<AnyItem>, RunError> {
    let Some(profiler) = watch.profiler else {
        return end_and_collect(id, stage, context, watch);
    };

    profiler.time(id, Counted::Never, context, |context| {
        end_and_collect(id, stage, context, watch)
    })
}

/// Ends a node, then takes whatever artefacts it built for the stream or chunk that ended.
fn end_and_collect(
    id: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    watch: EndWatch<'_>,
) -> Result<Vec<AnyItem>, RunError> {
    let flushed = end_node(id, stage, context)?;

    let Some(artefacts) = watch.artefacts else {
        return Ok(flushed);
    };

    match watch.discarding {
        true => artefacts.discard(id, stage, context)?,
        false => artefacts.take(id, watch.chunk, stage, context)?,
    }

    Ok(flushed)
}

/// Runs one node over one bundle of its inputs, handing `arrived` on ahead of what the bundle produces.
///
/// A node given one chunk at a time is ended and started again first when the bundle opens a new chunk,
/// and what that end flushed comes ahead of `arrived`, since those records reached the node after the
/// chunk it ended. The records a call made follow that call's items.
fn step_node(
    node: &mut (NodeId, StageBox),
    watch: &mut NodeWatch<'_>,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
    arrived: Vec<MetaRecord>,
) -> Result<Vec<Flow>, RunError> {
    let mut produced = Vec::new();

    if watch.chunked {
        let chunk = chunk_of(&slots, watch.authority, watch.route);
        let mut flushed = restart_on_new_chunk(node, watch, chunk, context)?;
        produced.append(&mut flushed);
    }

    let (id, stage) = node;
    release(*id, stage, context, arrived, &mut produced)?;

    let result = call_node(*id, stage, context, slots, watch.profiler)?;
    let records = taken_records(*id, context)?;

    let held_back = watch.frame_routed && is_filter(stage) && result.is_empty();
    if held_back {
        return Err(RunError::HeldInFrameRegion { node: *id });
    }

    let mut stepped = followed_by(result, records);
    produced.append(&mut stepped);

    Ok(produced)
}

fn is_filter(stage: &StageBox) -> bool {
    matches!(stage, StageBox::Filter(_) | StageBox::HostFilter(_))
}

/// The chunk a bundle belongs to on `route`, read from the slot `authority` names or else its lowest
/// filled slot.
fn chunk_of(slots: &[Option<AnyItem>], authority: Option<usize>, route: Option<ChunkRoute>) -> Option<u64> {
    let authoritative = authority
        .and_then(|slot| slots.get(slot))
        .and_then(Option::as_ref);
    let first_filled = slots.iter().flatten().next();
    let item = authoritative.or(first_filled)?;

    Some(chunk_in(item.meta(), route))
}

/// Ends and starts `node` again when `chunk` differs from the one it last ran on, returning what it flushed.
fn restart_on_new_chunk(
    node: &mut (NodeId, StageBox),
    watch: &mut NodeWatch<'_>,
    chunk: Option<u64>,
    context: &mut FrameContext,
) -> Result<Vec<Flow>, RunError> {
    let Some(chunk) = chunk else {
        return Ok(Vec::new());
    };

    let Some(previous) = watch.last.replace(chunk) else {
        return Ok(Vec::new());
    };

    if previous == chunk {
        return Ok(Vec::new());
    }

    let ending = EndWatch {
        chunk: Some(previous),
        discarding: false,
        artefacts: watch.artefacts,
        profiler: watch.profiler,
    };
    let restarted = end_and_start(node, context, ending);

    in_chunk(restarted, previous)
}

fn end_and_start(
    node: &mut (NodeId, StageBox),
    context: &mut FrameContext,
    ending: EndWatch<'_>,
) -> Result<Vec<Flow>, RunError> {
    let (id, stage) = node;
    let flushed = call_end(*id, stage, context, ending)?;
    start_node(*id, stage, context)?;
    refuse_stray_records(*id, context)?;

    Ok(flushed)
}

/// Names `chunk` in a failure of `result`.
fn in_chunk<T>(result: Result<T, RunError>, chunk: u64) -> Result<T, RunError> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => Err(error.in_chunk(chunk)),
    }
}

/// Places what each of a node's producers emitted this step on that feed, in order.
fn enqueue(feeds: &[Feed], inbox: &mut Inbox, outputs: &[Vec<Flow>], positions: &HashMap<NodeId, usize>) {
    for (index, feed) in feeds.iter().enumerate() {
        let position = positions[&feed.producer];

        for flow in &outputs[position] {
            inbox.receive(index, flow);
        }
    }
}

/// Runs one non-head node once for every item all of its feeds have ready, in the order they arrived.
///
/// A node with no feeds needs no input at all, so it runs once. A waiting record leaves as soon as the
/// node has taken every item that arrived ahead of it, after the output of the step that took the last.
/// One that arrived after every item already taken leaves ahead of the next step's output, behind
/// anything a restart on a new chunk flushes, or right away when no step is ready.
fn run_fed(
    feeds: &[Feed],
    inbox: &mut Inbox,
    node: &mut (NodeId, StageBox),
    watch: &mut NodeWatch<'_>,
    context: &mut FrameContext,
) -> Result<Vec<Flow>, RunError> {
    let (mut produced, settled) = feed_node(feeds, inbox, node, watch, context)?;
    release(node.0, &mut node.1, context, settled, &mut produced)?;

    Ok(produced)
}

/// Runs a node the same way as [run_fed], handing back unreleased the records due once the last step ran.
///
/// Those are the records due on entry when no step is ready, and the ones that become due after the last
/// step. While the node ends, no item is left behind them on its input.
fn feed_node(
    feeds: &[Feed],
    inbox: &mut Inbox,
    node: &mut (NodeId, StageBox),
    watch: &mut NodeWatch<'_>,
    context: &mut FrameContext,
) -> Result<(Vec<Flow>, Vec<MetaRecord>), RunError> {
    if feeds.is_empty() {
        let produced = step_node(node, watch, context, Vec::new(), Vec::new())?;
        return Ok((produced, Vec::new()));
    }

    let mut produced = Vec::new();
    let mut arrived = inbox.due();
    let ready = ready_count(&inbox.queues);

    for step in 0..ready {
        let slots = take_inputs(node.0, feeds, &mut inbox.queues)?;
        inbox.consumed += 1;

        let arriving = std::mem::take(&mut arrived);
        let mut result = step_node(node, watch, context, slots, arriving)?;
        produced.append(&mut result);

        let due = inbox.due();
        let last_step = step + 1 == ready;
        if last_step {
            arrived = due;
            break;
        }

        release(node.0, &mut node.1, context, due, &mut produced)?;
    }

    Ok((produced, arrived))
}

/// Feeds a node whatever its producers flushed, then flushes the node itself.
///
/// Anything its feeds still hold afterwards was never matched by the others, and is an error. A record
/// with no item left behind it on the node's input leaves after the node's own flush, as does any record
/// still waiting. A meta sink reads those records before it ends instead.
fn end_fed(
    feeds: &[Feed],
    inbox: &mut Inbox,
    node: &mut (NodeId, StageBox),
    mut watch: NodeWatch<'_>,
    ended: Option<u64>,
    context: &mut FrameContext,
) -> Result<Vec<Flow>, RunError> {
    let (mut forwarded, settled) = feed_node(feeds, inbox, node, &mut watch, context)?;

    let leftover: Vec<usize> = inbox.queues.iter().map(VecDeque::len).collect();
    if leftover.iter().any(|&count| count > 0) {
        let node = node.0;
        return Err(RunError::UnpairedFeeds { node, leftover });
    }

    let tagging = Tagging {
        chunked: watch.chunked,
        frame_routed: watch.frame_routed,
    };
    let chunk = ending_chunk(tagging, *watch.last, ended);
    let ending = EndWatch {
        chunk,
        discarding: watch.chunked && chunk.is_none(),
        artefacts: watch.artefacts,
        profiler: watch.profiler,
    };
    let mut owed = settled;
    owed.extend(inbox.drain());

    let (id, stage) = node;
    let mut flushed = end_owing(*id, stage, context, ending, owed)?;
    forwarded.append(&mut flushed);

    Ok(forwarded)
}

/// Files whatever `node` attached to the items and records in `flows` under its path.
///
/// A node with no path, such as a crossing, attaches nothing. A record passed on from another node has
/// nothing left to file, so stamping it again changes nothing.
fn stamp(paths: &HashMap<NodeId, Arc<str>>, node: NodeId, flows: &mut [Flow]) {
    let Some(path) = paths.get(&node) else {
        return;
    };

    for flow in flows {
        let attachments = match flow {
            Flow::Item(item) => &mut item.meta_mut().attachments,
            Flow::Record(record) => record.attachments_mut(),
        };
        attachments.stamp(path);
    }
}

/// How many times a node fed by `queues` is ready to run, which is how many items its shortest queue holds.
fn ready_count(queues: &[VecDeque<AnyItem>]) -> usize {
    queues.iter().map(VecDeque::len).min().unwrap_or(0)
}

/// Takes the next item from every feed and places each in the slot its feed fills.
///
/// Every item taken together must carry the same frame index. Items for different frames lined up as one
/// node's inputs would be processed as if they matched.
fn take_inputs(
    node: NodeId,
    feeds: &[Feed],
    queues: &mut [VecDeque<AnyItem>],
) -> Result<Vec<Option<AnyItem>>, RunError> {
    let width = feeds.iter().map(|feed| feed.slot + 1).max().unwrap_or(0);
    let mut slots: Vec<Option<AnyItem>> = vec![None; width];
    let mut indices = Vec::with_capacity(feeds.len());

    for (feed, queue) in feeds.iter().zip(queues.iter_mut()) {
        let item = queue
            .pop_front()
            .expect("every queue holds at least the ready count");
        indices.push(item.meta().index);
        slots[feed.slot] = Some(item);
    }

    let first = indices[0];
    if indices.iter().any(|&index| index != first) {
        return Err(RunError::OutOfStepFeeds { node, indices });
    }

    Ok(slots)
}

/// Concatenates what every exit node produced, in the order the region names them.
fn collect_exits(exits: &[NodeId], positions: &HashMap<NodeId, usize>, outputs: &[Vec<Flow>]) -> Vec<Flow> {
    let mut collected = Vec::new();

    for exit in exits {
        let position = positions[exit];
        collected.extend(outputs[position].iter().cloned());
    }

    collected
}
