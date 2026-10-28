use std::sync::atomic::Ordering;

use av_toolbox_core::StageKind;
use serde_json::{Value, json};

use crate::graph::tests::node_named;
use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::testing::{self, PROBE_CALLS};
use crate::tests::probes::{probing, seekable, sink};
use crate::{
    Spanned,
    StageSpec,
    build,
    fill_references,
    record_probes,
    record_probes_for,
    residency,
    run_probes,
    run_probes_for,
    satisfy_demands,
    satisfy_demands_for,
};

fn source(name: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/file");
    spec.args = Spanned::unknown(json!({ "path": "a.mkv" }));

    spec
}

fn sized(name: &str, width: u32) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/sized");
    spec.args = Spanned::unknown(json!({ "width": width, "height": 48 }));

    spec
}

fn demanding(name: &str, input: &str, every: Value) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/demanding");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(json!({ "every": every }));

    spec
}

/// A [demanding] stage whose `every` is the template `text`.
fn late_demanding(name: &str, input: &str, text: &str) -> StageSpec {
    let mut spec = demanding(name, input, json!(text));
    spec.deferred = vec![testing::deferred("/every", text)];

    spec
}

/// A [probing](crate::tests::probes::probing) stage whose `mark` is the template `text`.
fn late_probing(name: &str, input: &str, text: &str) -> StageSpec {
    let mut spec = probing(name, input, json!({ "mark": text }));
    spec.deferred = vec![testing::deferred("/mark", text)];

    spec
}

/// Runs the passes `Pipeline::build` and `Pipeline::compile` run up to filling references, then
/// places the demands and runs the probes of the filled nodes and the providers placed for them.
fn filled(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let mut graph = build(&registry, specs).expect("should build").into_graph();

    satisfy_demands(&mut graph, &registry).expect("should satisfy");
    record_probes(&mut graph, &registry).expect("should record");
    run_probes(&mut graph, &registry).expect("should probe");

    let filled = fill_references(&mut graph, &registry, &capabilities).expect("should fill");
    let first_inserted = graph.len();
    satisfy_demands_for(&mut graph, &registry, &filled).expect("should satisfy the filled nodes");

    let probing: Vec<NodeId> = graph
        .order()
        .iter()
        .copied()
        .filter(|id| filled.contains(id) || id.index() >= first_inserted)
        .collect();
    record_probes_for(&mut graph, &registry, &probing).expect("should record the filled nodes");
    run_probes_for(&mut graph, &registry, &probing).expect("should probe the filled nodes");

    graph
}

fn named<'a>(graph: &'a CompiledGraph, name: &str) -> &'a Node {
    let id = node_named(graph, name);

    graph.node(id)
}

fn providers(graph: &CompiledGraph) -> Vec<&Node> {
    graph
        .order()
        .iter()
        .map(|id| graph.node(*id))
        .filter(|node| matches!(node.role(), NodeRole::Provider(_)))
        .collect()
}

#[test]
fn a_filled_demand_takes_a_layer_no_earlier_provider_serves() {
    let specs = [
        source("input"),
        sized("size", 64),
        demanding("first", "input", json!(4)),
        late_demanding("late", "input", "{{ size.width }}"),
        sink("first_out", "first"),
        sink("late_out", "late"),
    ];

    let graph = filled(&specs);

    let first_layer = named(&graph, "first").demands()[0].layer;
    let late_layer = named(&graph, "late").demands()[0].layer;
    assert_ne!(first_layer, late_layer);
}

#[test]
fn a_filled_stage_that_demands_gets_one_provider() {
    let specs = [
        source("input"),
        sized("size", 64),
        late_demanding("late", "input", "{{ size.width }}"),
        sink("out", "late"),
    ];

    let graph = filled(&specs);

    let late = named(&graph, "late");
    assert_eq!(providers(&graph).len(), 1);
    assert_eq!(late.demands().len(), 1);
}

#[test]
fn a_filled_stage_that_probes_is_probed_once() {
    let specs = [
        seekable("input"),
        sized("size", 3),
        late_probing("late", "input", "{{ size.width }}"),
        sink("out", "late"),
    ];

    let graph = filled(&specs);

    let late = named(&graph, "late");
    assert_eq!(late.probes().len(), 1);
    assert!(late.probes()[0].found.is_some());
    assert_eq!(PROBE_CALLS.load(Ordering::SeqCst), 1);
}
