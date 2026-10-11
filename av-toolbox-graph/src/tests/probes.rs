use std::sync::Arc;
use std::sync::atomic::Ordering;

use av_toolbox_core::{AnyProbe, ProbeKind, Probed, ProbedValue, StageKind};
use serde_json::{Value, json};

use crate::graph::CompiledGraph;
use crate::graph::tests::node_named;
use crate::spec::{Spanned, StageSpec};
use crate::testing::{self, FixtureProbe, MarkRequest, PROBE_CALLS, SEEKABLE_OPENS, SEEKER_READS};
use crate::{
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    build,
    record_probes,
    residency,
    run_probes,
    satisfy_demands,
    solve,
};

pub(crate) fn seekable(name: &str) -> StageSpec {
    testing::spec(StageKind::Source, name, "test/seekable")
}

pub(crate) fn unseekable(name: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/file");
    spec.args = Spanned::unknown(json!({ "path": "a.mkv" }));

    spec
}

pub(crate) fn probing(name: &str, input: &str, args: Value) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/probing");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(args);

    spec
}

pub(crate) fn pair(name: &str, clip: &str, other: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/pair");
    spec.inputs = vec![testing::named("clip", clip), testing::named("other", other)];

    spec
}

pub(crate) fn sink(name: &str, input: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(input)];

    spec
}

/// Builds, satisfies demands and records probes, as `Pipeline::build` does.
pub(crate) fn recorded(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = build(&registry, specs).expect("should build").into_graph();
    satisfy_demands(&mut graph, &registry).expect("should satisfy");
    record_probes(&mut graph, &registry).expect("should record");

    graph
}

fn probed(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = recorded(specs);
    run_probes(&mut graph, &registry).expect("should probe");

    graph
}

fn probe_refused(specs: &[StageSpec]) -> Vec<Diagnostic> {
    let registry = testing::registry();
    let mut graph = recorded(specs);
    let Err(diagnostics) = run_probes(&mut graph, &registry) else {
        panic!("the probes should be refused");
    };

    diagnostics.into_inner()
}

/// What the [FixtureProbe] on `input` of the stage `name` found.
fn found_on(graph: &CompiledGraph, name: &str) -> u32 {
    let id = node_named(graph, name);
    let probed = graph.node(id).probed().expect("probed");

    *probed.get::<FixtureProbe>("input").expect("an answer")
}

/// What a [FixtureProbe] on `input` found, answering `value`.
fn found(value: u32) -> Probed {
    let mut probed = Probed::new();
    let output: ProbedValue = Arc::new(value);
    let kind = ProbeKind::of::<FixtureProbe>();
    probed.insert("input", kind, output);

    probed
}

#[test]
fn a_probing_stage_records_its_probe() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = recorded(&specs);

    let id = node_named(&graph, "crop");
    let node = graph.node(id);
    assert_eq!(node.probes().len(), 1);
    assert_eq!(node.probes()[0].input, "input");
    assert_eq!(node.probes()[0].description, "mark 3");
}

#[test]
fn recording_never_opens_a_seeker() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    recorded(&specs);

    assert_eq!(SEEKABLE_OPENS.load(Ordering::SeqCst), 0);
    assert_eq!(SEEKER_READS.load(Ordering::SeqCst), 0);
    assert_eq!(PROBE_CALLS.load(Ordering::SeqCst), 0);
}

#[test]
fn a_probe_on_an_unbound_optional_input_is_skipped() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3, "on_extra": true })),
        sink("out", "crop"),
    ];

    let graph = recorded(&specs);

    let id = node_named(&graph, "crop");
    let node = graph.node(id);
    assert!(node.probes().is_empty());
}

#[test]
fn a_probe_of_an_unregistered_kind_is_refused() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];
    let registry = testing::registry_without_probe_kinds();
    let mut graph = build(&registry, &specs).expect("should build").into_graph();

    let Err(diagnostics) = record_probes(&mut graph, &registry) else {
        panic!("an unregistered kind should be refused");
    };

    let diagnostics = diagnostics.into_inner();
    assert_eq!(diagnostics[0].code(), DiagnosticCode::UnknownProbe);
    assert_eq!(diagnostics[0].path(), "filter.crop");
}

#[test]
fn a_recorded_probe_carries_its_request() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = recorded(&specs);

    let id = node_named(&graph, "crop");
    let request = MarkRequest { mark: 3 };
    let expected = AnyProbe::new::<FixtureProbe>(request);
    assert_eq!(graph.node(id).probes()[0].probe, expected);
}

#[test]
fn a_probe_of_an_unseekable_source_is_still_recorded() {
    let specs = [
        unseekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = recorded(&specs);

    let id = node_named(&graph, "crop");
    assert_eq!(graph.node(id).probes().len(), 1);
}

#[test]
fn a_probe_behind_another_stage_is_recorded_on_the_prober_alone() {
    let specs = [
        seekable("left"),
        seekable("right"),
        pair("joined", "left", "right"),
        probing("crop", "joined", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = recorded(&specs);

    let crop = node_named(&graph, "crop");
    let joined = node_named(&graph, "joined");
    assert_eq!(graph.node(crop).probes().len(), 1);
    assert!(graph.node(joined).probes().is_empty());
}

#[test]
fn preparing_settles_a_stage_with_what_its_probes_found() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let mut graph = recorded(&specs);
    let unsettled = solve(&graph, &registry, &capabilities);
    assert!(unsettled.is_err());

    let id = node_named(&graph, "crop");
    let probed = found(11);
    graph.node_mut(id).set_probed(probed);

    solve(&graph, &registry, &capabilities).expect("a settled stage solves");
}

#[test]
fn a_stage_refusing_what_its_probes_found_is_reported() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3, "refuse": true })),
        sink("out", "crop"),
    ];
    let registry = testing::registry();
    let mut graph = recorded(&specs);
    let id = node_named(&graph, "crop");
    let probed = found(11);
    graph.node_mut(id).set_probed(probed);

    let mut diagnostics = Diagnostics::new();
    graph.prepare_stages(&registry, &mut diagnostics);

    let diagnostics = diagnostics.into_inner();
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), DiagnosticCode::ProbeFailed);
    assert_eq!(diagnostics[0].path(), "filter.crop");
    assert!(diagnostics[0].message().contains("the fixture refuses to settle"));
}

#[test]
fn one_probing_stage_is_answered_and_settled() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = probed(&specs);

    assert_eq!(found_on(&graph, "crop"), 11);
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    solve(&graph, &registry, &capabilities).expect("the settled stage produces");
}

#[test]
fn equal_requests_on_one_source_share_one_call() {
    let specs = [
        seekable("input"),
        probing("first", "input", json!({ "mark": 3 })),
        probing("second", "input", json!({ "mark": 3 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = probed(&specs);

    assert_eq!(PROBE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(found_on(&graph, "first"), 11);
    assert_eq!(found_on(&graph, "second"), 11);
}

#[test]
fn distinct_requests_on_one_source_share_one_call() {
    let specs = [
        seekable("input"),
        probing("first", "input", json!({ "mark": 3 })),
        probing("second", "input", json!({ "mark": 5 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = probed(&specs);

    assert_eq!(PROBE_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(found_on(&graph, "first"), 11);
    assert_eq!(found_on(&graph, "second"), 13);
}

#[test]
fn two_sources_on_separate_branches_probe_separately() {
    let specs = [
        seekable("a"),
        seekable("b"),
        probing("first", "a", json!({ "mark": 3 })),
        probing("second", "b", json!({ "mark": 3 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    probed(&specs);

    assert_eq!(PROBE_CALLS.load(Ordering::SeqCst), 2);
}

#[test]
fn a_probe_reaching_two_sources_is_combined() {
    let specs = [
        seekable("a"),
        seekable("b"),
        pair("joined", "a", "b"),
        probing("crop", "joined", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = probed(&specs);

    assert_eq!(found_on(&graph, "crop"), 11);
}

#[test]
fn a_probe_that_cannot_combine_names_where_the_paths_meet() {
    let specs = [
        seekable("a"),
        seekable("b"),
        pair("joined", "a", "b"),
        probing("crop", "joined", json!({ "mark": 3, "plain": true })),
        sink("out", "crop"),
    ];

    let diagnostics = probe_refused(&specs);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::ProbeFailed);
    assert_eq!(diagnostics[0].path(), "filter.crop");
    assert!(diagnostics[0].message().contains("`joined`"), "{diagnostics:?}");
}

#[test]
fn a_probe_reaching_a_source_that_cannot_seek_is_refused() {
    let specs = [
        unseekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let diagnostics = probe_refused(&specs);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::UnseekableSource);
    assert_eq!(diagnostics[0].path(), "filter.crop");
    assert!(diagnostics[0].message().contains("`input`"), "{diagnostics:?}");
}

#[test]
fn probing_opens_its_source_once() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];

    let graph = probed(&specs);
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    solve(&graph, &registry, &capabilities).expect("solves");

    assert_eq!(SEEKABLE_OPENS.load(Ordering::SeqCst), 1);
}

#[test]
fn a_failed_settle_is_reported_on_the_next_prepare() {
    let specs = [
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3, "refuse": true })),
        sink("out", "crop"),
    ];
    let graph = probed(&specs);
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();

    let Err(diagnostics) = solve(&graph, &registry, &capabilities) else {
        panic!("the settle should fail");
    };

    let diagnostics = diagnostics.into_inner();
    assert_eq!(diagnostics[0].code(), DiagnosticCode::ProbeFailed);
}

#[test]
fn a_graph_without_probes_prepares_nothing() {
    let specs = [unseekable("input"), sink("out", "input")];

    probed(&specs);

    assert_eq!(SEEKER_READS.load(Ordering::SeqCst), 0);
}
