use av_toolbox_core::{Granularity, StageKind};
use av_toolbox_graph::{CompiledGraph, NodeId, NodeRole};

/// A node of a stream's compiled pipeline and how the plan runs it, kept after the pipeline is gone.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PlannedNode {
    pub id: NodeId,
    /// The kind and name as one path, such as `filter.denoise`.
    pub path: String,
    pub kind: StageKind,
    /// Whether the node is a written stage, a provider or a crossing the graph inserted.
    pub role: NodeRole,
    pub granularity: Granularity,
    /// How many workers run the node, widths it inherits included.
    pub width: usize,
    /// Every node reading this node's output, once per edge.
    pub consumers: Vec<NodeId>,
}

/// Every node of `graph` in dependency order.
pub(crate) fn planned_nodes(graph: &CompiledGraph) -> Vec<PlannedNode> {
    let plan = graph
        .execution_plan()
        .expect("a compiled pipeline is always planned");

    let mut nodes = Vec::with_capacity(graph.len());
    for id in graph.order() {
        let node = graph.node(*id);
        let node_plan = plan.node(*id);
        let consumers = node.consumers().to_vec();

        nodes.push(PlannedNode {
            id: *id,
            path: node.path(),
            kind: node.kind(),
            role: node.role().clone(),
            granularity: node_plan.granularity(),
            width: node_plan.width(),
            consumers,
        });
    }

    nodes
}
