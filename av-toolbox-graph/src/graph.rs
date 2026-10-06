use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use av_toolbox_core::{Crossing, OpenSlot, Residency, StageHandle, StageKind, StreamType};
use av_toolbox_gpu::LayerId;
use serde_json::Value;

use crate::demands::{NodeDemand, ProviderPlacement};
use crate::diagnostic::stage_path;
use crate::plan::ExecutionPlan;
use crate::{ArgSpans, Span};

/// A node's position in the graph.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct NodeId(usize);

impl NodeId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(&self) -> usize {
        self.0
    }
}

/// Names an edge by the node it enters and its position in that node's inputs.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct EdgeId {
    node: NodeId,
    index: usize,
}

impl EdgeId {
    pub const fn new(node: NodeId, index: usize) -> Self {
        Self { node, index }
    }

    pub const fn node(&self) -> NodeId {
        self.node
    }

    pub const fn index(&self) -> usize {
        self.index
    }
}

/// Whether a consumer can take a producer's output outright or has to share it.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Sharing {
    Move,
    Share,
}

/// One stage reading another stage's output.
#[derive(Debug, Clone)]
pub struct Edge {
    name: Option<String>,
    param: Option<&'static str>,
    slot: usize,
    source: NodeId,
    sharing: Sharing,
    span: Span,
}

impl Edge {
    pub(crate) fn new(
        name: Option<String>,
        param: Option<&'static str>,
        slot: usize,
        source: NodeId,
        span: Span,
    ) -> Self {
        Self {
            name,
            param,
            slot,
            source,
            sharing: Sharing::Share,
            span,
        }
    }

    /// The input name the definition gave, or `None` when it was positional.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// The signature param this input filled, or `None` for the variadic tail.
    pub fn param(&self) -> Option<&'static str> {
        self.param
    }

    /// The `Inputs` slot this edge fills.
    ///
    /// This is the same space [Region::chunk_authority](crate::plan::Region::chunk_authority)
    /// indexes into.
    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn source(&self) -> NodeId {
        self.source
    }

    pub fn sharing(&self) -> Sharing {
        self.sharing
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub(crate) fn set_sharing(&mut self, sharing: Sharing) {
        self.sharing = sharing;
    }

    pub(crate) fn set_source(&mut self, source: NodeId) {
        self.source = source;
    }
}

/// A node is a stage the pipeline declared, a crossing the graph inserted between two stages on
/// different sides, or a provider the graph inserted to serve what stages demand.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeRole {
    Stage,
    Crossing(Crossing),
    /// A provider the graph placed, serving the layers listed, each with the stages that demanded it.
    Provider(ProviderPlacement),
}

/// A provider to append to the graph, with everything its node is built from.
pub(crate) struct ProviderNode {
    /// The full identifier the provider was registered under.
    pub(crate) identifier: String,
    /// The name of the provider's one input.
    pub(crate) input: &'static str,
    /// The name of the source the provider serves.
    pub(crate) source: String,
    pub(crate) args: Value,
    pub(crate) placement: ProviderPlacement,
}

/// The pieces a node is built from.
pub(crate) struct NodeParts {
    pub(crate) id: NodeId,
    pub(crate) name: String,
    pub(crate) kind: StageKind,
    pub(crate) identifier: String,
    pub(crate) args: Value,
    pub(crate) args_span: Span,
    pub(crate) arg_spans: ArgSpans,
    pub(crate) inputs: Vec<Edge>,
    pub(crate) workers: Option<usize>,
    pub(crate) span: Span,
}

/// One stage placed in the graph.
#[derive(Debug, Clone)]
pub struct Node {
    id: NodeId,
    name: String,
    kind: StageKind,
    identifier: String,
    args: Value,
    args_span: Span,
    arg_spans: ArgSpans,
    inputs: Vec<Edge>,
    consumers: Vec<NodeId>,
    workers: Option<usize>,
    span: Span,
    role: NodeRole,
    /// Where this node runs, once transfers have been placed.
    residency: Option<Residency>,
    /// The stream type this node emits.
    stream: Option<StreamType>,
    /// What a source's stage learned by opening its input, shared across every pass that prepares
    /// this node.
    open: OpenSlot,
    /// The scene layer this node routes and restarts its chunks on.
    layer: Option<LayerId>,
    /// What this node demanded, as the demand pass settled it.
    demands: Vec<NodeDemand>,
}

impl Node {
    pub(crate) fn new(parts: NodeParts) -> Self {
        Self {
            id: parts.id,
            name: parts.name,
            kind: parts.kind,
            identifier: parts.identifier,
            args: parts.args,
            args_span: parts.args_span,
            arg_spans: parts.arg_spans,
            inputs: parts.inputs,
            consumers: Vec::new(),
            workers: parts.workers,
            span: parts.span,
            role: NodeRole::Stage,
            residency: None,
            stream: parts.kind.output(),
            open: OpenSlot::new(),
            layer: None,
            demands: Vec::new(),
        }
    }

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn handle(&self) -> StageHandle {
        StageHandle::new(self.id.index() as u64)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn kind(&self) -> StageKind {
        self.kind
    }

    /// The kind and name as one path, such as `filter.scenes.denoise`.
    pub fn path(&self) -> String {
        stage_path(self.kind, &self.name)
    }

    /// The `package/stage` identifier the stage was registered under.
    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn args(&self) -> &Value {
        &self.args
    }

    pub fn args_span(&self) -> Span {
        self.args_span
    }

    pub fn arg_spans(&self) -> &ArgSpans {
        &self.arg_spans
    }

    /// The inputs in the order the definition wrote them.
    pub fn inputs(&self) -> &[Edge] {
        &self.inputs
    }

    /// Every read of this node's output, listed once per edge.
    pub fn consumers(&self) -> &[NodeId] {
        &self.consumers
    }

    /// How many instances of this stage the pipeline asked for, if it asked.
    pub fn workers(&self) -> Option<usize> {
        self.workers
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub(crate) fn inputs_mut(&mut self) -> &mut [Edge] {
        &mut self.inputs
    }

    pub fn role(&self) -> &NodeRole {
        &self.role
    }

    pub fn residency(&self) -> Option<Residency> {
        self.residency
    }

    pub fn output_stream(&self) -> Option<StreamType> {
        self.stream
    }

    /// The slot a source opens its input into, shared by every pass that prepares this node.
    pub fn open_slot(&self) -> &OpenSlot {
        &self.open
    }

    /// The scene layer this node routes and restarts its chunks on, if it has one.
    pub fn layer(&self) -> Option<LayerId> {
        self.layer
    }

    pub(crate) fn set_layer(&mut self, layer: LayerId) {
        self.layer = Some(layer);
    }

    /// What this node demanded, with the layer serving each demand.
    pub fn demands(&self) -> &[NodeDemand] {
        &self.demands
    }

    pub(crate) fn set_demands(&mut self, demands: Vec<NodeDemand>) {
        self.demands = demands;
    }
}

/// A validated pipeline, its nodes indexed by [NodeId](crate::NodeId) and ordered by dependency.
#[derive(Debug, Clone)]
pub struct CompiledGraph {
    nodes: Vec<Node>,
    order: Vec<NodeId>,
    plan: Option<ExecutionPlan>,
}

impl CompiledGraph {
    pub(crate) fn new(nodes: Vec<Node>) -> Self {
        let mut graph = Self {
            nodes,
            order: Vec::new(),
            plan: None,
        };

        graph.refresh();
        graph
    }

    /// How this graph will be run, once it has been planned.
    pub fn execution_plan(&self) -> Option<&ExecutionPlan> {
        self.plan.as_ref()
    }

    pub fn set_execution_plan(&mut self, plan: ExecutionPlan) {
        self.plan = Some(plan);
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn order(&self) -> &[NodeId] {
        &self.order
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn sinks(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes
            .iter()
            .filter(|node| node.kind() == StageKind::Sink)
            .map(|node| node.id())
    }

    pub(crate) fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.index()]
    }

    /// Drops nodes and points every edge that read a dropped node at its replacement.
    ///
    /// Surviving nodes are reindexed, so every [NodeId](crate::NodeId) held across this call is
    /// stale afterwards. A stored plan is keyed by those same stale ids, so it does not survive
    /// the call either.
    pub(crate) fn retain_nodes(&mut self, replacements: &HashMap<NodeId, NodeId>, dropped: &HashSet<NodeId>) {
        let mut moved: HashMap<NodeId, NodeId> = HashMap::new();
        let mut kept: Vec<Node> = Vec::with_capacity(self.nodes.len());

        for node in std::mem::take(&mut self.nodes) {
            if dropped.contains(&node.id) {
                continue;
            }

            moved.insert(node.id, NodeId::new(kept.len()));
            kept.push(node);
        }

        for (position, node) in kept.iter_mut().enumerate() {
            node.id = NodeId::new(position);

            for edge in node.inputs_mut() {
                let survivor = follow_replacements(replacements, edge.source());
                let reindexed = moved
                    .get(&survivor)
                    .copied()
                    .expect("an edge reads a dropped node that has no replacement");

                edge.set_source(reindexed);
            }
        }

        self.nodes = kept;
        self.plan = None;
        self.refresh();
    }

    /// Appends a crossing reading `producer`'s output, and returns it.
    ///
    /// The order is not recomputed until [CompiledGraph::finish_edit](crate::CompiledGraph::finish_edit) is called.
    pub(crate) fn push_crossing(&mut self, producer: NodeId, crossing: Crossing) -> NodeId {
        let source = &self.nodes[producer.index()];
        let id = NodeId::new(self.nodes.len());

        let node = Node {
            id,
            name: format!("{}({})", crossing.verb(), source.name()),
            kind: StageKind::Crossing,
            identifier: crossing.identifier().to_string(),
            args: Value::Object(Default::default()),
            args_span: source.span(),
            arg_spans: ArgSpans::default(),
            inputs: vec![Edge::new(None, None, 0, producer, source.span())],
            consumers: Vec::new(),
            workers: None,
            span: source.span(),
            role: NodeRole::Crossing(crossing),
            residency: Some(crossing.to_side()),
            stream: source.output_stream(),
            open: OpenSlot::new(),
            layer: None,
            demands: Vec::new(),
        };

        self.nodes.push(node);
        id
    }

    /// Appends a provider reading `after`'s output, moves each edge in `moved` onto it, and
    /// returns it.
    ///
    /// The provider is named after its demand kind and the source it serves, such as
    /// `scenes(input)`. The order is not recomputed until
    /// [CompiledGraph::finish_edit](crate::CompiledGraph::finish_edit) is called.
    pub(crate) fn push_provider(
        &mut self,
        after: NodeId,
        provider: ProviderNode,
        moved: &[EdgeId],
    ) -> NodeId {
        let id = NodeId::new(self.nodes.len());
        let producer = &self.nodes[after.index()];
        let name = format!("{}({})", provider.placement.kind.name(), provider.source);

        let mut input = Edge::new(None, Some(provider.input), 0, after, producer.span());
        let reads = self.reads_of(after);
        if reads == moved.len() {
            input.set_sharing(Sharing::Move);
        }

        let node = Node {
            id,
            name,
            kind: StageKind::Provider,
            identifier: provider.identifier,
            args: provider.args,
            args_span: producer.span(),
            arg_spans: ArgSpans::default(),
            inputs: vec![input],
            consumers: Vec::new(),
            workers: None,
            span: producer.span(),
            role: NodeRole::Provider(provider.placement),
            residency: None,
            stream: producer.output_stream(),
            open: OpenSlot::new(),
            layer: None,
            demands: Vec::new(),
        };

        self.nodes.push(node);

        for edge in moved {
            self.redirect(edge.node(), edge.index(), id);
        }

        id
    }

    /// How many edges read `producer`, counted from the edges themselves.
    fn reads_of(&self, producer: NodeId) -> usize {
        self.nodes
            .iter()
            .flat_map(|node| node.inputs())
            .filter(|edge| edge.source() == producer)
            .count()
    }

    pub(crate) fn redirect(&mut self, consumer: NodeId, edge: usize, source: NodeId) {
        self.nodes[consumer.index()].inputs[edge].set_source(source);
    }

    pub(crate) fn set_residency(&mut self, id: NodeId, residency: Residency) {
        self.nodes[id.index()].residency = Some(residency);
    }

    /// Recomputes consumers and order after nodes were added or edges moved.
    ///
    /// A stored plan was keyed to the old node list, so it is dropped.
    pub(crate) fn finish_edit(&mut self) {
        self.plan = None;
        self.refresh();
    }

    /// Recomputes consumers and the topological order after the node list changes.
    fn refresh(&mut self) {
        for node in self.nodes.iter_mut() {
            node.consumers.clear();
        }

        for index in 0..self.nodes.len() {
            let consumer = NodeId::new(index);
            let sources: Vec<NodeId> = self.nodes[index]
                .inputs()
                .iter()
                .map(|edge| edge.source())
                .collect();

            for source in sources {
                self.nodes[source.index()].consumers.push(consumer);
            }
        }

        self.order = topological_order(&self.nodes);
    }
}

/// Follows a chain of replacements through to the node that survived.
fn follow_replacements(replacements: &HashMap<NodeId, NodeId>, node: NodeId) -> NodeId {
    let mut current = node;

    while let Some(next) = replacements.get(&current) {
        if *next == current {
            break;
        }

        current = *next;
    }

    current
}

/// Orders the nodes so every node follows the nodes it reads.
///
/// Ready nodes are taken in index order, so the same nodes always produce the same order.
fn topological_order(nodes: &[Node]) -> Vec<NodeId> {
    let mut remaining: Vec<usize> = nodes.iter().map(|node| node.inputs().len()).collect();
    let mut ready: BinaryHeap<Reverse<usize>> = remaining
        .iter()
        .enumerate()
        .filter(|(_, count)| **count == 0)
        .map(|(index, _)| Reverse(index))
        .collect();

    let mut order = Vec::with_capacity(nodes.len());
    while let Some(Reverse(index)) = ready.pop() {
        order.push(NodeId::new(index));

        for consumer in nodes[index].consumers() {
            let count = &mut remaining[consumer.index()];
            *count -= 1;

            if *count == 0 {
                ready.push(Reverse(consumer.index()));
            }
        }
    }

    debug_assert_eq!(order.len(), nodes.len(), "the graph was built with a cycle in it");
    order
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{HashMap, HashSet};

    use av_toolbox_core::{Crossing, StageKind, StreamType};
    use serde_json::json;

    use super::*;
    use crate::build::build;
    use crate::testing;

    /// Builds a node with no arguments and the given positional inputs.
    fn node(index: usize, name: &str, kind: StageKind, inputs: &[usize]) -> Node {
        let edges = inputs
            .iter()
            .enumerate()
            .map(|(slot, source)| Edge::new(None, None, slot, NodeId::new(*source), Span::unknown()))
            .collect();

        Node::new(NodeParts {
            id: NodeId::new(index),
            name: name.to_string(),
            kind,
            identifier: "test/example".to_string(),
            args: json!({}),
            args_span: Span::unknown(),
            arg_spans: ArgSpans::default(),
            inputs: edges,
            workers: None,
            span: Span::unknown(),
        })
    }

    /// A source, a filter reading it and a sink reading the filter, declared back to front.
    fn chain() -> CompiledGraph {
        let nodes = vec![
            node(0, "sink", StageKind::Sink, &[2]),
            node(1, "source", StageKind::Source, &[]),
            node(2, "filter", StageKind::Filter, &[1]),
        ];

        CompiledGraph::new(nodes)
    }

    #[test]
    fn a_graph_orders_its_nodes_by_dependency_not_declaration() {
        let graph = chain();
        let names: Vec<&str> = graph.order().iter().map(|id| graph.node(*id).name()).collect();

        assert_eq!(names, vec!["source", "filter", "sink"]);
    }

    #[test]
    fn a_graph_records_who_reads_each_node() {
        let graph = chain();

        assert_eq!(graph.node(NodeId::new(1)).consumers(), &[NodeId::new(2)]);
        assert!(graph.node(NodeId::new(0)).consumers().is_empty());
    }

    #[test]
    fn a_node_read_twice_by_one_consumer_is_listed_twice() {
        let nodes = vec![
            node(0, "source", StageKind::Source, &[]),
            node(1, "mask", StageKind::Mask, &[0, 0]),
        ];
        let graph = CompiledGraph::new(nodes);

        assert_eq!(graph.node(NodeId::new(0)).consumers().len(), 2);
    }

    #[test]
    fn a_handle_is_derived_from_the_node_index() {
        let graph = chain();

        assert_eq!(graph.node(NodeId::new(2)).handle().id(), 2);
    }

    #[test]
    fn retaining_nodes_reindexes_and_rewires_the_survivors() {
        let nodes = vec![
            node(0, "source", StageKind::Source, &[]),
            node(1, "duplicate", StageKind::Filter, &[0]),
            node(2, "kept", StageKind::Filter, &[0]),
            node(3, "sink", StageKind::Sink, &[1]),
        ];
        let mut graph = CompiledGraph::new(nodes);

        let replacements = HashMap::from([(NodeId::new(1), NodeId::new(2))]);
        let dropped = HashSet::from([NodeId::new(1)]);
        graph.retain_nodes(&replacements, &dropped);

        let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();
        assert_eq!(names, vec!["source", "kept", "sink"]);

        let sink = graph.node(NodeId::new(2));
        assert_eq!(sink.inputs()[0].source(), NodeId::new(1));
        assert_eq!(sink.id(), NodeId::new(2));
    }

    #[test]
    fn sinks_lists_every_sink() {
        let graph = chain();
        let sinks: Vec<NodeId> = graph.sinks().collect();

        assert_eq!(sinks, vec![NodeId::new(0)]);
    }

    pub(crate) fn file_blur_stdout() -> CompiledGraph {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = crate::Spanned::unknown(serde_json::json!({ "path": "a.mkv" }));

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = crate::Spanned::unknown(serde_json::json!({ "sigma": 1.0 }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        build(&testing::registry(), &[source, blur, sink])
            .expect("should build")
            .into_graph()
    }

    pub(crate) fn node_named(graph: &CompiledGraph, name: &str) -> NodeId {
        graph
            .nodes()
            .iter()
            .find(|node| node.name() == name)
            .map(Node::id)
            .expect("a node with that name")
    }

    #[test]
    fn a_declared_stage_has_the_stage_role_and_no_residency_yet() {
        let graph = file_blur_stdout();
        let blur = graph.node(node_named(&graph, "blur"));

        assert_eq!(*blur.role(), NodeRole::Stage);
        assert_eq!(blur.residency(), None);
        assert_eq!(blur.output_stream(), Some(StreamType::Frames));
    }

    #[test]
    fn a_crossing_reads_its_producer_and_is_named_after_it() {
        let mut graph = file_blur_stdout();
        let blur = node_named(&graph, "blur");

        let crossing = graph.push_crossing(blur, Crossing::Unload);
        graph.finish_edit();

        let node = graph.node(crossing);
        assert_eq!(node.name(), "unload(blur)");
        assert_eq!(node.identifier(), "builtin/unload");
        assert_eq!(*node.role(), NodeRole::Crossing(Crossing::Unload));
        assert_eq!(node.inputs()[0].source(), blur);
        assert!(graph.node(blur).consumers().contains(&crossing));
    }

    #[test]
    fn redirecting_an_edge_orders_the_crossing_before_its_new_consumer() {
        let mut graph = file_blur_stdout();
        let blur = node_named(&graph, "blur");
        let out = node_named(&graph, "out");

        let crossing = graph.push_crossing(blur, Crossing::Unload);
        graph.redirect(out, 0, crossing);
        graph.finish_edit();

        let position = |id: NodeId| {
            graph
                .order()
                .iter()
                .position(|&each| each == id)
                .expect("ordered")
        };
        assert!(position(crossing) < position(out));
        assert_eq!(graph.node(out).inputs()[0].source(), crossing);
        assert!(!graph.node(blur).consumers().contains(&out));
    }

    #[test]
    fn a_crossing_carries_its_producers_stream_type() {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = crate::Spanned::unknown(serde_json::json!({ "path": "a.mkv" }));

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("input"), testing::positional("input")];

        let mut blur = testing::spec(StageKind::Filter, "deband", "test/deband");
        blur.inputs = vec![testing::positional("input"), testing::named("overlay", "merge")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("deband")];

        let mut graph = build(&testing::registry(), &[source, merge, blur, sink])
            .expect("should build")
            .into_graph();
        let merge = node_named(&graph, "merge");

        let crossing = graph.push_crossing(merge, Crossing::Unload);
        graph.finish_edit();

        assert_eq!(graph.node(crossing).output_stream(), Some(StreamType::Mask));
    }

    #[test]
    fn a_node_carries_the_worker_count_its_spec_asked_for() {
        let registry = testing::registry();
        let source = testing::spec(StageKind::Source, "input", "test/file");
        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.workers = Some(4);
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        let specs = vec![source, blur, sink];
        let graph = build(&registry, &specs).expect("should build").into_graph();

        assert_eq!(graph.node(NodeId::new(0)).workers(), None);
        assert_eq!(graph.node(NodeId::new(1)).workers(), Some(4));
    }
}
