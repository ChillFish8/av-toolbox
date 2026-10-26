use av_toolbox_core::StageKind;

use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{
    SolvedFormats,
    StageSpec,
    build,
    insert_widens,
    place_transfers,
    plan,
    residency,
    solve,
    testing,
};

fn filter(name: &str, stage_type: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, stage_type);
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn sink(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn nothing_in_common() -> Vec<StageSpec> {
    vec![
        testing::spec(StageKind::Source, "input", "test/wide"),
        filter("small", "test/narrow", "input"),
        filter("exact", "test/precise", "small"),
        sink("out", "exact"),
    ]
}

fn two_precise_readers() -> Vec<StageSpec> {
    vec![
        testing::spec(StageKind::Source, "input", "test/wide"),
        filter("small", "test/narrow", "input"),
        filter("first", "test/precise", "small"),
        filter("second", "test/precise", "small"),
        sink("out_first", "first"),
        sink("out_second", "second"),
    ]
}

fn widened(specs: &[StageSpec]) -> (CompiledGraph, SolvedFormats) {
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let mut graph = build(&registry, specs).expect("should build").into_graph();
    let mut formats = solve(&graph, &registry, &capabilities).expect("should solve");

    insert_widens(&mut graph, &mut formats);

    (graph, formats)
}

fn node_named(graph: &CompiledGraph, name: &str) -> NodeId {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .map(Node::id)
        .expect("a node with that name")
}

fn depth_stages(graph: &CompiledGraph) -> Vec<&Node> {
    graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .collect()
}

#[test]
fn a_widen_becomes_a_depth_stage_between_its_ends() {
    let specs = nothing_in_common();
    let (graph, formats) = widened(&specs);

    assert_eq!(graph.len(), specs.len() + 1);
    assert_eq!(formats.len(), graph.len());

    let small = node_named(&graph, "small");
    let exact = node_named(&graph, "exact");
    let depth = graph.node(NodeId::new(specs.len()));
    assert_eq!(depth.identifier(), "resample/depth");
    assert_eq!(depth.name(), "depth(small)");
    assert_eq!(depth.inputs()[0].source(), small);
    assert_eq!(graph.node(exact).inputs()[0].source(), depth.id());
}

#[test]
fn a_widened_graph_still_plans() {
    let registry = testing::registry();
    let (mut graph, mut formats) = widened(&nothing_in_common());

    place_transfers(&mut graph, &registry, &mut formats).expect("should place");
    plan(&graph, &registry).expect("should plan");
}

#[test]
fn two_readers_widened_to_one_sample_share_a_stage() {
    let specs = two_precise_readers();
    let (graph, _) = widened(&specs);

    let stages = depth_stages(&graph);
    assert_eq!(stages.len(), 1);

    let depth = stages[0].id();
    let first = node_named(&graph, "first");
    let second = node_named(&graph, "second");
    assert_eq!(graph.node(first).inputs()[0].source(), depth);
    assert_eq!(graph.node(second).inputs()[0].source(), depth);
}
