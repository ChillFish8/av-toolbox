use av_toolbox_core::{PathNode, PathView, PreparedStage};

use super::{Asked, Demander, Instance, Placements, Step};
use crate::graph::{CompiledGraph, EdgeId, Node, NodeId};

/// The real nodes `asked` travels through from its source, source first.
pub(super) fn trace(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    placements: &Placements,
    asked: &Asked,
) -> Option<Vec<Step>> {
    match asked.demander {
        Demander::Stage(node) => stage_steps(graph, prepared, node, asked.demand.input),
        Demander::Instance(index) => instance_steps(graph, prepared, &placements.instances[index]),
    }
}

/// The real nodes a stage's demand on `input` travels through, traced from that input.
fn stage_steps(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    node: NodeId,
    input: &str,
) -> Option<Vec<Step>> {
    let path = trace_to_source(graph, prepared, node, input)?;
    let steps = path.iter().map(|edge| single_step(graph, *edge)).collect();

    Some(steps)
}

/// The real nodes a provider's demand travels through, traced from the real node it sits behind.
///
/// The last step leaves that node along every edge the provider took over.
fn instance_steps(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    instance: &Instance,
) -> Option<Vec<Step>> {
    let upstream = trace_upstream(graph, prepared, instance.node)?;
    let mut steps: Vec<Step> = upstream.iter().map(|edge| single_step(graph, *edge)).collect();
    steps.push(Step {
        node: instance.node,
        edges: instance.edges.clone(),
    });

    Some(steps)
}

fn single_step(graph: &CompiledGraph, edge: EdgeId) -> Step {
    Step {
        node: source_of(graph, edge),
        edges: vec![edge],
    }
}

/// The real nodes a demand travels through along `steps`, as its provider sees them.
pub(super) fn path_view<'g>(graph: &'g CompiledGraph, steps: &[Step]) -> PathView<'g> {
    let nodes = steps
        .iter()
        .map(|step| {
            let node = graph.node(step.node);

            PathNode {
                name: node.name(),
                identifier: node.identifier(),
                args: node.args(),
            }
        })
        .collect();
    let stages = graph.nodes().iter().map(Node::name).collect();

    PathView::new(nodes, stages)
}

/// The edges a demand on `input` of `node` travels along from its source, source end first.
fn trace_to_source(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    node: NodeId,
    input: &str,
) -> Option<Vec<EdgeId>> {
    let start = bound_edge(graph.node(node), input)?;
    let producer = graph.node(node).inputs()[start].source();

    let mut path = trace_upstream(graph, prepared, producer)?;
    path.push(EdgeId::new(node, start));

    Some(path)
}

/// The edges from `node`'s source down to `node`, source end first, or none when `node` is a source.
///
/// A node with one input is followed through it, and a node with several through its chunk
/// authority. The trace ends at a node with no inputs.
fn trace_upstream(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    node: NodeId,
) -> Option<Vec<EdgeId>> {
    let mut path = Vec::new();
    let mut current = node;

    loop {
        let producer = graph.node(current);
        let followed = match producer.inputs().len() {
            0 => break,
            1 => 0,
            _ => authority_edge(producer, prepared)?,
        };

        path.push(EdgeId::new(current, followed));
        current = producer.inputs()[followed].source();
    }

    path.reverse();
    Some(path)
}

/// The position of the edge bound to `input`, by its signature param or its variadic name.
pub(super) fn bound_edge(node: &Node, input: &str) -> Option<usize> {
    node.inputs().iter().position(|edge| match edge.param() {
        Some(param) => param == input,
        None => edge.name() == Some(input),
    })
}

/// The position of the edge filling `node`'s chunk authority.
fn authority_edge(node: &Node, prepared: &[Option<PreparedStage>]) -> Option<usize> {
    let stage = prepared.get(node.id().index())?.as_ref()?;
    let slot = stage.chunk_authority()?;

    node.inputs().iter().position(|edge| edge.slot() == slot)
}

pub(super) fn source_of(graph: &CompiledGraph, edge: EdgeId) -> NodeId {
    graph.node(edge.node()).inputs()[edge.index()].source()
}
