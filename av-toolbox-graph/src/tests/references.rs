use av_toolbox_core::StageKind;
use serde_json::json;

use crate::graph::CompiledGraph;
use crate::graph::tests::node_named;
use crate::{
    CommonStageElimination,
    DeadStageElimination,
    DiagnosticCode,
    Diagnostics,
    OptimisationPass,
    Span,
    Spanned,
    StageSpec,
    build,
    testing,
};

fn source(name: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/file");
    spec.args = Spanned::unknown(json!({ "path": "a.mkv" }));

    spec
}

fn blur(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/blur");
    spec.inputs = vec![testing::positional(from)];
    spec.args = Spanned::unknown(json!({ "sigma": 1.0 }));

    spec
}

/// A blur reading `from` whose sigma is a template reading `referenced`.
fn deferred_blur(name: &str, from: &str, referenced: &str) -> StageSpec {
    let text = format!("{{{{ {referenced}.width }}}}");
    let mut spec = blur(name, from);
    spec.args = Spanned::unknown(json!({ "sigma": text }));
    spec.deferred = vec![testing::deferred("/sigma", &text)];

    spec
}

fn sink(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(from)];

    spec
}

/// `clip_sized` reads `score` and references `distorted`, which is declared after it.
fn referenced_after_its_reader() -> Vec<StageSpec> {
    vec![
        source("score"),
        deferred_blur("clip_sized", "score", "distorted"),
        source("distorted"),
        sink("out", "clip_sized"),
    ]
}

fn built(specs: &[StageSpec]) -> CompiledGraph {
    build(&testing::registry(), specs)
        .expect("should build")
        .into_graph()
}

fn position(graph: &CompiledGraph, name: &str) -> usize {
    let id = node_named(graph, name);

    graph
        .order()
        .iter()
        .position(|each| *each == id)
        .expect("ordered")
}

#[test]
fn a_reference_orders_its_stage_first() {
    let graph = built(&referenced_after_its_reader());

    assert!(position(&graph, "distorted") < position(&graph, "clip_sized"));
}

#[test]
fn a_reference_is_bound_to_its_stage() {
    let graph = built(&referenced_after_its_reader());
    let clip_sized = graph.node(node_named(&graph, "clip_sized"));

    assert!(clip_sized.is_deferred());
    assert_eq!(clip_sized.deferred().len(), 1);
    assert_eq!(clip_sized.references().len(), 1);
    assert_eq!(clip_sized.references()[0].source, node_named(&graph, "distorted"));
}

#[test]
fn a_stage_referenced_twice_is_one_reference() {
    let mut specs = referenced_after_its_reader();
    specs[1].deferred = vec![testing::deferred(
        "/sigma",
        "{{ distorted.width + distorted.height }}",
    )];

    let graph = built(&specs);
    let clip_sized = graph.node(node_named(&graph, "clip_sized"));

    assert_eq!(clip_sized.references().len(), 1);
}

#[test]
fn a_reference_loop_is_a_cycle() {
    let specs = vec![
        source("input"),
        deferred_blur("a", "input", "b"),
        blur("b", "a"),
        sink("out", "b"),
    ];

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let cycle = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::Cycle)
        .expect("should report a cycle");

    let notes: Vec<&str> = cycle.notes().iter().map(|note| note.message.as_str()).collect();
    assert!(notes.iter().any(|note| note.contains("reads the format of")));
}

#[test]
fn an_unknown_reference_suggests_a_name() {
    let specs = vec![
        source("score"),
        deferred_blur("clip_sized", "score", "distorted_"),
        source("distorted"),
        sink("out", "clip_sized"),
    ];

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let unknown = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::UnknownStageReference)
        .expect("should report an unknown reference");

    let notes: Vec<&str> = unknown.notes().iter().map(|note| note.message.as_str()).collect();
    assert_eq!(notes, vec!["did you mean `distorted`?"]);
}

#[test]
fn a_misspelled_stage_is_reported_once() {
    let mut specs = referenced_after_its_reader();
    specs[1].deferred = vec![testing::deferred(
        "/sigma",
        "{{ distorted_.width + distorted_.height }}",
    )];

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

    assert_eq!(codes, vec![DiagnosticCode::UnknownStageReference]);
}

#[test]
fn a_reference_to_a_sink_is_rejected() {
    let specs = vec![
        source("score"),
        deferred_blur("clip_sized", "score", "out"),
        sink("out", "clip_sized"),
    ];

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

    assert!(codes.contains(&DiagnosticCode::SinkHasNoOutput));
}

#[test]
fn a_stage_only_referenced_is_not_unused() {
    let built = build(&testing::registry(), &referenced_after_its_reader()).expect("should build");

    let unused = built
        .warnings()
        .iter()
        .any(|warning| warning.code() == DiagnosticCode::UnusedStage);
    assert!(!unused);
}

#[test]
fn dead_stage_elimination_keeps_a_referenced_stage() {
    let mut specs = referenced_after_its_reader();
    specs.insert(1, blur("spare", "score"));
    let mut graph = built(&specs);

    DeadStageElimination.run(&mut graph);

    let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();
    assert_eq!(names, vec!["score", "clip_sized", "distorted", "out"]);

    let clip_sized = graph.node(node_named(&graph, "clip_sized"));
    assert_eq!(clip_sized.references()[0].source, node_named(&graph, "distorted"));
}

#[test]
fn deferred_stages_are_never_merged() {
    let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
    merge.inputs = vec![testing::positional("first"), testing::positional("second")];

    let specs = vec![
        source("input"),
        deferred_blur("first", "input", "input"),
        deferred_blur("second", "input", "input"),
        merge,
        sink("out", "merge"),
    ];
    let mut graph = built(&specs);

    let report = CommonStageElimination.run(&mut graph);

    assert!(report.is_empty());
    assert_eq!(graph.len(), 5);
}

#[test]
fn a_deferred_stage_is_not_prepared() {
    let registry = testing::registry();
    let graph = built(&referenced_after_its_reader());
    let mut diagnostics = Diagnostics::new();

    let prepared = graph.prepare_stages(&registry, &mut diagnostics);

    let clip_sized = node_named(&graph, "clip_sized");
    assert!(prepared[clip_sized.index()].is_none());
    assert!(prepared[node_named(&graph, "score").index()].is_some());
    assert!(diagnostics.is_empty());
}

#[test]
fn filling_args_clears_the_deferred_args_and_references() {
    let mut graph = built(&referenced_after_its_reader());
    let clip_sized = node_named(&graph, "clip_sized");

    graph.fill_args(clip_sized, json!({ "sigma": 2.0 }));

    let node = graph.node(clip_sized);
    assert_eq!(node.args(), &json!({ "sigma": 2.0 }));
    assert!(!node.is_deferred());
    assert!(node.references().is_empty());
}

/// `first` and `second` are identical sources, and `clip_sized` references the given ones.
fn references_to_identical_sources(text: &str) -> Vec<StageSpec> {
    let mut clip_sized = blur("clip_sized", "input");
    clip_sized.args = Spanned::unknown(json!({ "sigma": text }));
    clip_sized.deferred = vec![testing::deferred("/sigma", text)];

    vec![
        source("input"),
        source("first"),
        source("second"),
        clip_sized,
        sink("out", "clip_sized"),
    ]
}

#[test]
fn a_reference_to_a_merged_stage_follows_the_kept_stage() {
    let specs = references_to_identical_sources("{{ second.width }}");
    let mut graph = built(&specs);

    CommonStageElimination.run(&mut graph);

    let clip_sized = graph.node(node_named(&graph, "clip_sized"));
    assert_eq!(clip_sized.references()[0].source, node_named(&graph, "input"));
}

#[test]
fn references_to_merged_stages_fold_into_one() {
    let specs = references_to_identical_sources("{{ first.width + second.width }}");
    let mut graph = built(&specs);

    CommonStageElimination.run(&mut graph);

    let clip_sized = graph.node(node_named(&graph, "clip_sized"));
    let sources: Vec<_> = clip_sized
        .references()
        .iter()
        .map(|reference| reference.source)
        .collect();
    assert_eq!(sources, vec![node_named(&graph, "input")]);
}

/// Two blurs reading `referenced` through one call variable written at `variable`.
fn read_through_one_variable(referenced: &str, variable: Span) -> Vec<StageSpec> {
    let mut first = deferred_blur("first", "score", referenced);
    first.deferred[0].span = Span::new(40, 50);
    testing::through_variable(&mut first.deferred[0], variable);

    let mut second = deferred_blur("second", "score", referenced);
    second.deferred[0].span = Span::new(60, 70);
    testing::through_variable(&mut second.deferred[0], variable);

    vec![
        source("score"),
        first,
        second,
        sink("first_out", "first"),
        sink("second_out", "second"),
    ]
}

#[test]
fn an_unknown_stage_in_a_call_variable_is_reported_once_at_the_variable() {
    let variable = Span::new(5, 9);
    let specs = read_through_one_variable("scroe", variable);

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let unknown: Vec<_> = diagnostics
        .iter()
        .filter(|entry| entry.code() == DiagnosticCode::UnknownStageReference)
        .collect();

    let [only] = unknown.as_slice() else {
        panic!("one diagnostic per bad variable, found {unknown:?}");
    };
    let used: Vec<Span> = only
        .notes()
        .iter()
        .filter(|note| note.message == "used here")
        .map(|note| note.span)
        .collect();

    assert_eq!(only.span(), variable);
    assert_eq!(used, vec![Span::new(40, 50), Span::new(60, 70)]);
}

#[test]
fn an_unknown_reference_path_names_the_arg() {
    let specs = vec![
        source("score"),
        deferred_blur("clip_sized", "score", "missing"),
        sink("out", "clip_sized"),
    ];

    let diagnostics = build(&testing::registry(), &specs).expect_err("should fail");
    let unknown = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::UnknownStageReference)
        .expect("should report an unknown reference");

    assert_eq!(unknown.path(), "filter.clip_sized.args.sigma");
}
