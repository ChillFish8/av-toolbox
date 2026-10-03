use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use av_toolbox_gpu::{AnyItem, FrameContext, StreamItem, StreamPolicy};
use av_toolbox_graph::{NodeId, StageBox};

use super::node::{end_node, run_node, start_node};
use crate::RunError;

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
/// time, which are restarted whenever their inputs move to a new one.
/// `frame_routed` marks a region whose frames are each handed back on their own, so a filter returning no
/// frames in it is an error.
#[derive(Debug)]
pub struct RegionEdges {
    feeds: Vec<Vec<Feed>>,
    exits: Vec<NodeId>,
    paths: HashMap<NodeId, Arc<str>>,
    chunked: HashSet<NodeId>,
    head_authority: Option<usize>,
    frame_routed: bool,
}

impl RegionEdges {
    pub fn new(feeds: Vec<Vec<Feed>>, exits: Vec<NodeId>) -> Self {
        Self {
            feeds,
            exits,
            paths: HashMap::new(),
            chunked: HashSet::new(),
            head_authority: None,
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

    /// The same edges, reading the head's chunk from the slot `authority` names.
    pub fn with_head_authority(mut self, authority: Option<usize>) -> Self {
        self.head_authority = authority;
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
pub struct RegionRunner {
    nodes: Vec<(NodeId, StageBox)>,
    edges: RegionEdges,
    positions: HashMap<NodeId, usize>,
    queues: Vec<Vec<VecDeque<AnyItem>>>,
    last_chunks: Vec<Option<u64>>,
}

impl RegionRunner {
    pub fn new(nodes: Vec<(NodeId, StageBox)>, edges: RegionEdges) -> Self {
        let positions = nodes
            .iter()
            .enumerate()
            .map(|(position, (id, _))| (*id, position))
            .collect();
        let queues = edges
            .feeds
            .iter()
            .map(|feeds| feeds.iter().map(|_| VecDeque::new()).collect())
            .collect();

        let last_chunks = vec![None; nodes.len()];

        Self {
            nodes,
            edges,
            positions,
            queues,
            last_chunks,
        }
    }

    /// Signals the start of a chunk to every node in the region.
    pub fn start(&mut self, context: &mut FrameContext) -> Result<(), RunError> {
        self.last_chunks.fill(None);

        for (id, stage) in &mut self.nodes {
            start_node(*id, stage, context)?;
        }

        Ok(())
    }

    /// Runs the region once over `bundle`, the head's inputs, returning what its exits produced.
    ///
    /// The step ends by flushing this thread's queued work, before anything it read is let go or
    /// anything it produced is returned. Under [StreamPolicy::FlushAtHandOff] the device items in
    /// `bundle` are held until then, so a head dropping its inputs early cannot release them first.
    pub fn run(
        &mut self,
        context: &mut FrameContext,
        bundle: Vec<Option<AnyItem>>,
    ) -> Result<Vec<AnyItem>, RunError> {
        let mut outputs: Vec<Vec<AnyItem>> = Vec::with_capacity(self.nodes.len());
        let held_inputs = hold_until_flushed(context, &bundle);

        let mut head_watch = NodeWatch {
            chunked: self.edges.chunked.contains(&self.nodes[0].0),
            frame_routed: self.edges.frame_routed,
            authority: self.edges.head_authority,
            last: &mut self.last_chunks[0],
        };
        let mut head_output = step_node(&mut self.nodes[0], &mut head_watch, context, bundle)?;
        stamp(&self.edges.paths, self.nodes[0].0, &mut head_output);
        outputs.push(head_output);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let queues = &mut self.queues[index - 1];
            enqueue(feeds, queues, &outputs, &self.positions);

            let node = self.nodes[index].0;
            let watch = NodeWatch {
                chunked: self.edges.chunked.contains(&node),
                frame_routed: self.edges.frame_routed,
                authority: None,
                last: &mut self.last_chunks[index],
            };
            let mut produced = run_fed(feeds, queues, &mut self.nodes[index], watch, context)?;
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
    /// submitting this thread's queued work, before anything it produced is returned.
    pub fn end(&mut self, context: &mut FrameContext) -> Result<Vec<AnyItem>, RunError> {
        let mut outputs: Vec<Vec<AnyItem>> = Vec::with_capacity(self.nodes.len());

        let (head_id, head_stage) = &mut self.nodes[0];
        let mut head_flush = end_node(*head_id, head_stage, context)?;
        stamp(&self.edges.paths, *head_id, &mut head_flush);
        outputs.push(head_flush);

        for index in 1..self.nodes.len() {
            let feeds = &self.edges.feeds[index - 1];
            let queues = &mut self.queues[index - 1];
            enqueue(feeds, queues, &outputs, &self.positions);

            let node = self.nodes[index].0;
            let watch = NodeWatch {
                chunked: self.edges.chunked.contains(&node),
                frame_routed: self.edges.frame_routed,
                authority: None,
                last: &mut self.last_chunks[index],
            };
            let mut flushed = end_fed(feeds, queues, &mut self.nodes[index], watch, context)?;
            stamp(&self.edges.paths, node, &mut flushed);
            outputs.push(flushed);
        }

        let exits = collect_exits(&self.edges.exits, &self.positions, &outputs);

        context.flush();

        Ok(exits)
    }
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
    last: &'a mut Option<u64>,
}

/// Runs one node over one bundle of its inputs.
///
/// A node given one chunk at a time is ended and started again first when the bundle opens a new chunk,
/// and what that end flushed comes ahead of what the bundle produces.
fn step_node(
    node: &mut (NodeId, StageBox),
    watch: &mut NodeWatch<'_>,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
) -> Result<Vec<AnyItem>, RunError> {
    let mut produced = Vec::new();

    if watch.chunked {
        let chunk = chunk_of(&slots, watch.authority);
        let mut flushed = restart_on_new_chunk(node, watch.last, chunk, context)?;
        produced.append(&mut flushed);
    }

    let (id, stage) = node;
    let mut result = run_node(*id, stage, context, slots)?;

    let held_back = watch.frame_routed && is_filter(stage) && result.is_empty();
    if held_back {
        return Err(RunError::HeldInFrameRegion { node: *id });
    }

    produced.append(&mut result);

    Ok(produced)
}

fn is_filter(stage: &StageBox) -> bool {
    matches!(stage, StageBox::Filter(_) | StageBox::HostFilter(_))
}

/// The chunk a bundle belongs to, read from the slot `authority` names or else its lowest filled slot.
fn chunk_of(slots: &[Option<AnyItem>], authority: Option<usize>) -> Option<u64> {
    let authoritative = authority
        .and_then(|slot| slots.get(slot))
        .and_then(Option::as_ref);
    let first_filled = slots.iter().flatten().next();
    let item = authoritative.or(first_filled)?;

    Some(item.meta().chunk)
}

/// Ends and starts `node` again when `chunk` differs from the one it last ran on, returning what it flushed.
fn restart_on_new_chunk(
    node: &mut (NodeId, StageBox),
    last: &mut Option<u64>,
    chunk: Option<u64>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    let Some(chunk) = chunk else {
        return Ok(Vec::new());
    };

    let previous = last.replace(chunk);
    if previous.is_none_or(|previous| previous == chunk) {
        return Ok(Vec::new());
    }

    let (id, stage) = node;
    let flushed = end_node(*id, stage, context)?;
    start_node(*id, stage, context)?;

    Ok(flushed)
}

/// Appends what each of a node's producers emitted this step to that feed's queue.
fn enqueue(
    feeds: &[Feed],
    queues: &mut [VecDeque<AnyItem>],
    outputs: &[Vec<AnyItem>],
    positions: &HashMap<NodeId, usize>,
) {
    for (feed, queue) in feeds.iter().zip(queues.iter_mut()) {
        let position = positions[&feed.producer];
        queue.extend(outputs[position].iter().cloned());
    }
}

/// Runs one non-head node once for every item all of its feeds have ready, in the order they arrived.
///
/// A node with no feeds needs no input at all, so it runs once.
fn run_fed(
    feeds: &[Feed],
    queues: &mut [VecDeque<AnyItem>],
    node: &mut (NodeId, StageBox),
    mut watch: NodeWatch<'_>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    if feeds.is_empty() {
        return step_node(node, &mut watch, context, Vec::new());
    }

    let ready = ready_count(queues);
    let mut produced = Vec::new();

    for _ in 0..ready {
        let slots = take_inputs(node.0, feeds, queues)?;
        let mut result = step_node(node, &mut watch, context, slots)?;
        produced.append(&mut result);
    }

    Ok(produced)
}

/// Feeds a node whatever its producers flushed, then flushes the node itself.
///
/// Anything its feeds still hold afterwards was never matched by the others, and is an error.
fn end_fed(
    feeds: &[Feed],
    queues: &mut [VecDeque<AnyItem>],
    node: &mut (NodeId, StageBox),
    watch: NodeWatch<'_>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    let mut forwarded = run_fed(feeds, queues, node, watch, context)?;

    let leftover: Vec<usize> = queues.iter().map(VecDeque::len).collect();
    if leftover.iter().any(|&count| count > 0) {
        let node = node.0;
        return Err(RunError::UnpairedFeeds { node, leftover });
    }

    let (id, stage) = node;
    let flushed = end_node(*id, stage, context)?;

    forwarded.extend(flushed);
    Ok(forwarded)
}

/// Files whatever `node` attached to `items` under its path.
///
/// A node with no path, such as a crossing, attaches nothing.
fn stamp(paths: &HashMap<NodeId, Arc<str>>, node: NodeId, items: &mut [AnyItem]) {
    let Some(path) = paths.get(&node) else {
        return;
    };

    for item in items {
        item.meta_mut().attachments.stamp(path);
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
fn collect_exits(
    exits: &[NodeId],
    positions: &HashMap<NodeId, usize>,
    outputs: &[Vec<AnyItem>],
) -> Vec<AnyItem> {
    let mut collected = Vec::new();

    for exit in exits {
        let position = positions[exit];
        collected.extend(outputs[position].iter().cloned());
    }

    collected
}
