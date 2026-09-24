use av_toolbox_core::StageKind;
use serde_json::json;

use crate::graph::{CompiledGraph, NodeId};
use crate::{Spanned, StageSpec, build, testing};

pub(crate) fn built(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let built = build(&registry, specs).expect("should build");

    built.into_graph()
}

pub(crate) fn planned(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = built(specs);
    let made = crate::plan::plan(&graph, &registry).expect("should plan");

    graph.set_execution_plan(made);
    graph
}

pub(crate) fn node_named(graph: &CompiledGraph, name: &str) -> NodeId {
    let found = graph.nodes().iter().find(|node| node.name() == name);

    found.expect("the node is in the graph").id()
}

fn source(name: &str, path: &str) -> StageSpec {
    let mut source = testing::spec(StageKind::Source, name, "test/file");
    source.args = Spanned::unknown(json!({ "path": path }));
    source
}

fn temporal(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = testing::spec(StageKind::Filter, name, "test/temporal");
    filter.inputs = vec![testing::positional(input)];
    filter.args = Spanned::unknown(json!({ "radius": 4 }));
    filter.workers = workers;
    filter
}

fn blur(name: &str, input: &str) -> StageSpec {
    let mut blur = testing::spec(StageKind::Filter, name, "test/blur");
    blur.inputs = vec![testing::positional(input)];
    blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));
    blur
}

fn stdout(name: &str, input: &str) -> StageSpec {
    let mut sink = testing::spec(StageKind::Sink, name, "test/stdout");
    sink.inputs = vec![testing::positional(input)];
    sink
}

pub(crate) fn source_to_sink() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let out = stdout("out", "input");

    vec![input, out]
}

/// A source and a sink stay one wide, and a `test/temporal` filter between them widens.
pub(crate) fn wide_pipeline(workers: usize) -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let denoise = temporal("denoise", "input", Some(workers));
    let out = stdout("out", "denoise");

    vec![input, denoise, out]
}

/// `test/widening-source` is frame mode, so widening it is the one way to route by frame.
pub(crate) fn frame_pipeline() -> Vec<StageSpec> {
    let mut input = testing::spec(StageKind::Source, "input", "test/widening-source");
    input.workers = Some(4);
    let out = stdout("out", "input");

    vec![input, out]
}

/// A stage explicitly pinned to one worker, so its reason is `Declared`.
pub(crate) fn declared_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let pinned = temporal("pinned", "input", Some(1));
    let out = stdout("out", "pinned");

    vec![input, pinned, out]
}

/// One branch widens and one does not, and both feed a stage that reads both.
pub(crate) fn mixed_inputs_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let wide = temporal("wide", "input", Some(4));
    let narrow = temporal("narrow", "input", None);

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::positional("wide"), testing::positional("narrow")];
    merge.args = Spanned::unknown(json!({ "radius": 0 }));

    let out = stdout("out", "merge");

    vec![input, wide, narrow, merge, out]
}

/// Two sources read by one mask through its named `base` and `overlay` inputs.
pub(crate) fn named_inputs_pipeline() -> Vec<StageSpec> {
    let first = source("first", "a.mkv");
    let second = source("second", "b.mkv");

    let mut matched = testing::spec(StageKind::Mask, "matched", "test/matched");
    matched.inputs = vec![
        testing::named("base", "first"),
        testing::named("overlay", "second"),
    ];

    let out = stdout("out", "matched");

    vec![first, second, matched, out]
}

/// A chain of stages each widened to a different width, so every one gets its own region.
pub(crate) fn many_region_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let first = temporal("first", "input", Some(4));
    let second = temporal("second", "first", Some(2));
    let third = temporal("third", "second", Some(8));
    let out = stdout("out", "third");

    vec![input, first, second, third, out]
}

/// A widened stage feeding two branches that each end at their own sink.
pub(crate) fn two_exit_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let wide = temporal("wide", "input", Some(4));
    let left = temporal("left", "wide", None);
    let right = temporal("right", "wide", None);
    let out_left = stdout("out_left", "left");
    let out_right = stdout("out_right", "right");

    vec![input, wide, left, right, out_left, out_right]
}

/// A source read directly by a mask, and also through a filter, so one edge skips a column.
pub(crate) fn skipping_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let blurred = blur("blur", "input");

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
    merge.inputs = vec![testing::positional("blur"), testing::positional("input")];

    let out = stdout("out", "merge");

    vec![input, blurred, merge, out]
}

/// Two sources whose consumers are declared in the opposite order, which crosses the edges.
pub(crate) fn crossed_pipeline() -> Vec<StageSpec> {
    let first_source = source("first_source", "a.mkv");
    let second_source = source("second_source", "b.mkv");
    let first_blur = blur("first_blur", "second_source");
    let second_blur = blur("second_blur", "first_source");
    let first_out = stdout("first_out", "first_blur");
    let second_out = stdout("second_out", "second_blur");

    vec![
        first_source,
        second_source,
        first_blur,
        second_blur,
        first_out,
        second_out,
    ]
}

/// `first` is read by a later region as well as by the stages after it in its own region, so
/// that region's exit sits in its first column.
pub(crate) fn middle_exit_pipeline() -> Vec<StageSpec> {
    let first = source("first", "a.mkv");
    let second = source("second", "b.mkv");
    let blurred = blur("blur", "first");
    let out = stdout("out", "blur");

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
    merge.inputs = vec![testing::positional("first"), testing::positional("second")];

    let merged_out = stdout("merged_out", "merge");

    vec![first, second, blurred, out, merge, merged_out]
}

/// A host source feeding a device blur feeding a host sink, with both crossings placed.
pub(crate) fn placed_round_trip() -> CompiledGraph {
    let mut input = testing::spec(StageKind::Source, "input", "test/host-file");
    input.args = Spanned::unknown(json!({}));
    let blurred = blur("blur", "input");

    let mut out = testing::spec(StageKind::Sink, "out", "test/host-stdout");
    out.inputs = vec![testing::positional("blur")];

    let registry = testing::registry();
    let mut graph = built(&[input, blurred, out]);
    let capabilities = crate::residency::tests::capabilities();
    let mut formats = crate::solve(&graph, &registry, &capabilities).expect("should solve");
    crate::place_transfers(&mut graph, &registry, &mut formats).expect("should place");

    graph
}

pub(crate) fn planned_round_trip() -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = placed_round_trip();
    let made = crate::plan::plan(&graph, &registry).expect("should plan");

    graph.set_execution_plan(made);
    graph
}
