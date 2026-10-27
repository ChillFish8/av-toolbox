use av_toolbox_core::StageKind;
use av_toolbox_gpu::{Depth, SampleFormat};
use serde_json::{Value, json};

use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::{
    DiagnosticCode,
    Diagnostics,
    SolvedFormats,
    Spanned,
    StageSpec,
    build,
    inject_output_depth,
    residency,
    satisfy_demands,
    solve,
    testing,
};

const TEN_BIT: SampleFormat = SampleFormat::U16 { depth: Depth::TEN };

const TWELVE_BIT: SampleFormat = SampleFormat::U16 { depth: Depth::TWELVE };

fn source(name: &str, stage_type: &str) -> StageSpec {
    testing::spec(StageKind::Source, name, stage_type)
}

fn depth(name: &str, input: &str, bits: u8) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "resample/depth");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(json!({ "depth": bits }));

    spec
}

fn y4m(name: &str, input: &str, args: Value) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/y4m");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(args);

    spec
}

fn stdout(name: &str, input: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(input)];

    spec
}

fn source_into_y4m(stage_type: &str) -> Vec<StageSpec> {
    vec![source("input", stage_type), y4m("out", "input", json!({}))]
}

fn injected(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let mut graph = build(&registry, specs).expect("should build").into_graph();

    satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");
    inject_output_depth(&mut graph, &registry, &capabilities).expect("should inject");

    graph
}

fn solved(graph: &CompiledGraph) -> Result<SolvedFormats, Diagnostics> {
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();

    solve(graph, &registry, &capabilities)
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

fn sink_sample(graph: &CompiledGraph, formats: &SolvedFormats, sink: &str) -> SampleFormat {
    let sink = node_named(graph, sink);
    let input = formats.get(sink).input(0);

    input
        .map(|format| format.sample())
        .expect("the sink's input settled")
}

#[test]
fn an_eight_bit_source_is_written_at_ten_bits() {
    let graph = injected(&source_into_y4m("test/eight"));
    let formats = solved(&graph).expect("should solve");

    let stages = depth_stages(&graph);
    assert_eq!(stages.len(), 1);
    assert_eq!(stages[0].name(), "depth(out)");
    assert_eq!(*stages[0].args(), json!({ "depth": 10, "at_least": true }));

    let out = node_named(&graph, "out");
    assert_eq!(graph.node(out).inputs()[0].source(), stages[0].id());
    assert_eq!(stages[0].span(), graph.node(out).span());
    assert_eq!(sink_sample(&graph, &formats, "out"), TEN_BIT);
}

#[test]
fn a_twelve_bit_source_is_written_at_twelve_bits() {
    let graph = injected(&source_into_y4m("test/twelve"));
    let formats = solved(&graph).expect("should solve");

    assert_eq!(sink_sample(&graph, &formats, "out"), TWELVE_BIT);
}

#[test]
fn a_sixteen_bit_source_into_y4m_cannot_solve() {
    let graph = injected(&source_into_y4m("test/sixteen"));

    let diagnostics = solved(&graph).expect_err("should fail");
    let reported: Vec<_> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].code(), DiagnosticCode::NoCommonFormat);
    assert_eq!(reported[0].path(), "sink.out.inputs[0]");
}

#[test]
fn grey_output_settles_on_twelve_bits() {
    let graph = injected(&source_into_y4m("test/grey"));
    let formats = solved(&graph).expect("should solve");

    assert_eq!(sink_sample(&graph, &formats, "out"), TWELVE_BIT);
}

#[test]
fn a_deeper_stream_than_the_target_is_not_narrowed() {
    let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
    blur.inputs = vec![testing::positional("twelve")];
    blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));
    let specs = vec![
        source("input", "test/eight"),
        depth("twelve", "input", 12),
        blur,
        y4m("out", "blur", json!({})),
    ];
    let graph = injected(&specs);
    let formats = solved(&graph).expect("should solve");

    assert_eq!(sink_sample(&graph, &formats, "out"), TWELVE_BIT);
}

#[test]
fn a_written_depth_before_the_sink_is_left_alone() {
    let specs = vec![
        source("input", "test/eight"),
        depth("eight", "input", 8),
        y4m("out", "eight", json!({})),
    ];
    let graph = injected(&specs);
    let formats = solved(&graph).expect("should solve");

    assert_eq!(graph.len(), specs.len());
    assert_eq!(sink_sample(&graph, &formats, "out"), SampleFormat::U8);
}

#[test]
fn a_written_depth_behind_a_provider_is_left_alone() {
    let specs = vec![
        source("input", "test/eight"),
        depth("eight", "input", 8),
        y4m("out", "eight", json!({ "breaks_after": "eight" })),
    ];
    let graph = injected(&specs);
    let formats = solved(&graph).expect("should solve");

    let out = node_named(&graph, "out");
    let feeding = graph.node(out).inputs()[0].source();
    let role = graph.node(feeding).role();
    assert!(
        matches!(role, NodeRole::Provider(_)),
        "{}",
        graph.node(feeding).name()
    );
    assert_eq!(depth_stages(&graph).len(), 1);
    assert_eq!(sink_sample(&graph, &formats, "out"), SampleFormat::U8);
}

#[test]
fn two_sinks_each_get_their_own_conversion() {
    let specs = vec![
        source("input", "test/eight"),
        y4m("first", "input", json!({})),
        y4m("second", "input", json!({})),
    ];
    let graph = injected(&specs);
    let formats = solved(&graph).expect("should solve");

    let stages = depth_stages(&graph);
    let mut names: Vec<&str> = stages.iter().map(|stage| stage.name()).collect();
    names.sort();
    assert_eq!(names, ["depth(first)", "depth(second)"]);

    assert_eq!(sink_sample(&graph, &formats, "first"), TEN_BIT);
    assert_eq!(sink_sample(&graph, &formats, "second"), TEN_BIT);
}

#[test]
fn a_sink_that_does_not_want_it_is_left_alone() {
    let specs = vec![source("input", "test/eight"), stdout("out", "input")];
    let graph = injected(&specs);

    assert_eq!(graph.len(), specs.len());
    assert!(depth_stages(&graph).is_empty());
}

#[test]
fn a_mask_into_the_sink_is_left_for_the_sink_to_refuse() {
    let mut edges = testing::spec(StageKind::Mask, "edges", "test/merge");
    edges.inputs = vec![testing::positional("input")];
    let specs = vec![
        source("input", "test/eight"),
        edges,
        y4m("out", "edges", json!({})),
    ];
    let graph = injected(&specs);

    assert!(depth_stages(&graph).is_empty());

    let diagnostics = solved(&graph).expect_err("should fail");
    let reported: Vec<_> = diagnostics.iter().collect();

    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].path(), "sink.out.inputs[0]");
}
