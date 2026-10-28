use av_toolbox_core::StageKind;
use av_toolbox_gpu::SampleFormat;
use serde_json::json;

use crate::graph::CompiledGraph;
use crate::graph::tests::node_named;
use crate::natural::natural_formats;
use crate::tests::probes::{probing, recorded, seekable, sink};
use crate::{DiagnosticCode, Spanned, StageSpec, build, residency, testing};

fn built(specs: &[StageSpec]) -> CompiledGraph {
    build(&testing::registry(), specs)
        .expect("should build")
        .into_graph()
}

fn source(name: &str, stage_type: &str) -> StageSpec {
    testing::spec(StageKind::Source, name, stage_type)
}

fn blur(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/blur");
    spec.inputs = vec![testing::positional(from)];
    spec.args = Spanned::unknown(json!({ "sigma": 1.0 }));

    spec
}

#[test]
fn a_natural_format_takes_the_most_preferred_sample() {
    let specs = vec![
        source("input", "test/floatfirst"),
        blur("soft", "input"),
        sink("out", "soft"),
    ];
    let graph = built(&specs);
    let soft = node_named(&graph, "soft");
    let capabilities = residency::tests::capabilities();

    let formats =
        natural_formats(&graph, &testing::registry(), &capabilities, &[soft]).expect("should work out");

    assert_eq!(formats.len(), 1);
    assert_eq!(formats[&soft].sample(), SampleFormat::F32);
}

#[test]
fn a_stage_with_unsettled_probes_has_no_natural_format() {
    let specs = vec![
        seekable("input"),
        probing("crop", "input", json!({ "mark": 3 })),
        sink("out", "crop"),
    ];
    let graph = recorded(&specs);
    let crop = node_named(&graph, "crop");
    let capabilities = residency::tests::capabilities();

    let diagnostics =
        natural_formats(&graph, &testing::registry(), &capabilities, &[crop]).expect_err("should fail");
    let failed = diagnostics.iter().next().expect("a diagnostic");

    assert_eq!(failed.code(), DiagnosticCode::ProbeFailed);
    assert!(failed.message().contains("crop"), "{}", failed.message());
}

#[test]
fn a_deferred_stage_has_no_natural_format() {
    let mut deferred = blur("soft", "input");
    deferred.args = Spanned::unknown(json!({ "sigma": "{{ input.width }}" }));
    deferred.deferred = vec![testing::deferred("/sigma", "{{ input.width }}")];

    let specs = vec![source("input", "test/eight"), deferred, sink("out", "soft")];
    let graph = built(&specs);
    let soft = node_named(&graph, "soft");
    let capabilities = residency::tests::capabilities();

    let diagnostics =
        natural_formats(&graph, &testing::registry(), &capabilities, &[soft]).expect_err("should fail");
    let failed = diagnostics.iter().next().expect("a diagnostic");

    assert!(failed.message().contains("soft"), "{}", failed.message());
}
