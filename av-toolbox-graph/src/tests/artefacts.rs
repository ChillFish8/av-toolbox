use av_toolbox_core::StageKind;
use serde_json::json;

use crate::graph::{CompiledGraph, NodeId};
use crate::{DiagnosticCode, Spanned, StageSpec, build, testing};

/// A source feeding a grain filter and a frame sink, plus an artefact sink reading `inputs` with `args`.
fn specs_with_files(inputs: &[&str], args: serde_json::Value) -> Vec<StageSpec> {
    let source = testing::spec(StageKind::Source, "input", "test/file");

    let mut grain = testing::spec(StageKind::Filter, "grain", "test/grain");
    grain.inputs = vec![testing::positional("input")];

    let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
    sink.inputs = vec![testing::positional("grain")];

    let mut files = testing::spec(StageKind::Sink, "files", "test/files");
    files.inputs = inputs.iter().map(|input| testing::positional(input)).collect();
    files.args = Spanned::unknown(args);

    vec![source, grain, sink, files]
}

/// Adds a filter of `stage_type` named `name` reading the source.
fn add_filter(specs: &mut Vec<StageSpec>, name: &str, stage_type: &str) {
    let mut filter = testing::spec(StageKind::Filter, name, stage_type);
    filter.inputs = vec![testing::positional("input")];

    specs.insert(2, filter);
}

fn codes_of(specs: &[StageSpec]) -> Vec<DiagnosticCode> {
    let registry = testing::registry();
    let diagnostics = build(&registry, specs).expect_err("should fail");

    diagnostics.iter().map(|entry| entry.code()).collect()
}

fn node_id(graph: &CompiledGraph, name: &str) -> NodeId {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .expect("the node exists")
        .id()
}

#[test]
fn an_artefact_sink_selects_every_artefact_its_inputs_declare_by_default() {
    let registry = testing::registry();
    let specs = specs_with_files(&["grain"], json!({}));

    let built = build(&registry, &specs).expect("should build");
    let graph = built.graph();
    let sinks = graph.artefact_sinks();
    let grain = node_id(graph, "grain");

    assert_eq!(sinks.len(), 1);
    assert_eq!(sinks[0].path(), "sink.files");
    assert_eq!(sinks[0].selected(), &[(grain, "table"), (grain, "cuts")]);
}

#[test]
fn an_artefact_sink_never_becomes_a_node() {
    let registry = testing::registry();
    let specs = specs_with_files(&["grain"], json!({}));

    let built = build(&registry, &specs).expect("should build");
    let names: Vec<&str> = built.graph().nodes().iter().map(|node| node.name()).collect();

    assert_eq!(names, vec!["input", "grain", "out"]);
}

#[test]
fn an_artefact_sink_selects_only_its_inputs_artefacts() {
    let registry = testing::registry();
    let mut specs = specs_with_files(&["grain"], json!({}));
    add_filter(&mut specs, "twin", "test/grain");

    let built = build(&registry, &specs).expect("should build");
    let graph = built.graph();
    let grain = node_id(graph, "grain");

    assert_eq!(
        graph.artefact_sinks()[0].selected(),
        &[(grain, "table"), (grain, "cuts")]
    );
}

#[test]
fn inputs_are_selected_in_the_order_written() {
    let registry = testing::registry();
    let mut specs = specs_with_files(&["scenes", "grain"], json!({}));
    add_filter(&mut specs, "scenes", "test/scenes");

    let built = build(&registry, &specs).expect("should build");
    let graph = built.graph();
    let grain = node_id(graph, "grain");
    let scenes = node_id(graph, "scenes");

    assert_eq!(
        graph.artefact_sinks()[0].selected(),
        &[(scenes, "shots"), (grain, "table"), (grain, "cuts")]
    );
}

#[test]
fn include_and_exclude_narrow_the_selection_by_name() {
    let registry = testing::registry();
    let args = json!({ "include": ["*"], "exclude": ["table"] });
    let specs = specs_with_files(&["grain"], args);

    let built = build(&registry, &specs).expect("should build");
    let selected = built.graph().artefact_sinks()[0].selected().to_vec();
    let names: Vec<&str> = selected.iter().map(|(_, name)| *name).collect();

    assert_eq!(names, vec!["cuts"]);
}

#[test]
fn a_pattern_matching_one_input_is_accepted() {
    let registry = testing::registry();
    let mut specs = specs_with_files(&["grain", "scenes"], json!({ "include": ["shots"] }));
    add_filter(&mut specs, "scenes", "test/scenes");

    let built = build(&registry, &specs).expect("should build");
    let graph = built.graph();
    let scenes = node_id(graph, "scenes");

    assert_eq!(graph.artefact_sinks()[0].selected(), &[(scenes, "shots")]);
}

#[test]
fn a_pattern_matching_only_a_stage_not_read_fails() {
    let mut specs = specs_with_files(&["grain"], json!({ "include": ["shots"] }));
    add_filter(&mut specs, "scenes", "test/scenes");

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::UnmatchedArtefactPattern]);
}

#[test]
fn a_pattern_matching_no_artefact_fails_with_the_closest_name() {
    let registry = testing::registry();
    let specs = specs_with_files(&["grain"], json!({ "include": ["tabel"] }));

    let diagnostics = build(&registry, &specs).expect_err("should fail");
    let unmatched = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::UnmatchedArtefactPattern)
        .expect("an unmatched pattern");

    assert!(unmatched.message().contains("did you mean table"));
}

#[test]
fn an_input_named_twice_selects_its_artefacts_once() {
    let registry = testing::registry();
    let specs = specs_with_files(&["grain", "grain"], json!({}));

    let built = build(&registry, &specs).expect("should build");
    let graph = built.graph();
    let grain = node_id(graph, "grain");

    assert_eq!(
        graph.artefact_sinks()[0].selected(),
        &[(grain, "table"), (grain, "cuts")]
    );
}

#[test]
fn an_artefact_sink_with_no_inputs_fails() {
    let specs = specs_with_files(&[], json!({}));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::ArtefactSinkNoInputs]);
}

#[test]
fn a_named_input_on_an_artefact_sink_fails() {
    let mut specs = specs_with_files(&[], json!({}));
    specs[3].inputs = vec![testing::named("table", "grain")];

    let registry = testing::registry();
    let diagnostics = build(&registry, &specs).expect_err("should fail");
    let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

    assert_eq!(codes, vec![DiagnosticCode::InvalidInputs]);
    assert_eq!(
        diagnostics.iter().next().expect("one").path(),
        "sink.files.inputs[0]"
    );
}

#[test]
fn an_input_naming_an_unknown_stage_fails() {
    let specs = specs_with_files(&["grian"], json!({}));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::UnknownStageReference]);
}

#[test]
fn an_input_naming_a_sink_fails() {
    let specs = specs_with_files(&["out"], json!({}));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::SinkHasNoOutput]);
}

#[test]
fn an_input_declaring_no_artefacts_fails() {
    let specs = specs_with_files(&["grain", "input"], json!({}));

    let registry = testing::registry();
    let diagnostics = build(&registry, &specs).expect_err("should fail");
    let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

    assert_eq!(codes, vec![DiagnosticCode::InputDeclaresNoArtefacts]);
    assert_eq!(
        diagnostics.iter().next().expect("one").path(),
        "sink.files.inputs[1]"
    );
}

#[test]
fn a_rejected_producer_reports_no_follow_on_pattern_error() {
    let mut specs = specs_with_files(&["grain"], json!({ "include": ["table"] }));
    specs[1].stage_type = Spanned::unknown(String::from("test/nope"));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::UnknownStage]);
}

#[test]
fn a_rejected_producer_reports_no_follow_on_empty_selection() {
    let mut specs = specs_with_files(&["grain"], json!({}));
    specs[1].stage_type = Spanned::unknown(String::from("test/nope"));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::UnknownStage]);
}

#[test]
fn a_rejected_producer_still_reports_an_artefact_sink_with_no_inputs() {
    let mut specs = specs_with_files(&[], json!({}));
    specs[1].stage_type = Spanned::unknown(String::from("test/nope"));

    let codes = codes_of(&specs);

    assert!(codes.contains(&DiagnosticCode::UnknownStage));
    assert!(codes.contains(&DiagnosticCode::ArtefactSinkNoInputs));
}

#[test]
fn a_rejected_producer_still_reports_bad_artefact_sink_arguments() {
    let mut specs = specs_with_files(&["grain"], json!({ "include": [""] }));
    specs[1].stage_type = Spanned::unknown(String::from("test/nope"));

    let codes = codes_of(&specs);

    assert!(codes.contains(&DiagnosticCode::UnknownStage));
    assert!(codes.contains(&DiagnosticCode::InvalidArgs));
}

#[test]
fn an_empty_include_fails_as_selecting_nothing() {
    let specs = specs_with_files(&["grain"], json!({ "include": [] }));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::ArtefactSinkSelectsNothing]);
}

#[test]
fn an_artefact_sink_with_bad_arguments_fails() {
    let specs = specs_with_files(&["grain"], json!({ "include": [""] }));

    assert_eq!(codes_of(&specs), vec![DiagnosticCode::InvalidArgs]);
}

#[test]
fn an_artefact_sink_alone_is_not_a_sink_for_frames() {
    let mut specs = specs_with_files(&["grain"], json!({}));
    specs.remove(2);

    let codes = codes_of(&specs);

    assert!(codes.contains(&DiagnosticCode::NoSinks));
}

#[test]
fn a_stage_inside_a_called_file_is_read_by_its_prefixed_name() {
    let registry = testing::registry();
    let mut specs = specs_with_files(&["scenes.grain"], json!({ "include": ["cuts"] }));
    specs[1].name = Spanned::unknown(String::from("scenes.grain"));
    specs[2].inputs = vec![testing::positional("scenes.grain")];

    let built = build(&registry, &specs).expect("should build");
    let selected = built.graph().artefact_sinks()[0].selected().to_vec();

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].1, "cuts");
}

#[test]
fn the_selected_sinks_build_through_the_registry() {
    let registry = testing::registry();
    let specs = specs_with_files(&["grain"], json!({}));
    let graph = build(&registry, &specs).expect("should build").into_graph();

    let mut built = graph.build_artefact_sinks(&registry).expect("should build");
    let written = built[0].write(&[]).expect("writes nothing");

    assert_eq!(built.len(), 1);
    assert_eq!(written, 0);
}

/// A source feeding a frame sink directly, and a grain filter only an artefact sink reads.
fn specs_with_a_producer_only() -> Vec<StageSpec> {
    let source = testing::spec(StageKind::Source, "input", "test/file");

    let mut grain = testing::spec(StageKind::Filter, "grain", "test/grain");
    grain.inputs = vec![testing::positional("input")];

    let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
    sink.inputs = vec![testing::positional("input")];

    let mut files = testing::spec(StageKind::Sink, "files", "test/files");
    files.inputs = vec![testing::positional("grain")];

    vec![source, grain, sink, files]
}

#[test]
fn a_producer_only_an_artefact_sink_reads_is_not_unused() {
    let registry = testing::registry();
    let specs = specs_with_a_producer_only();

    let built = build(&registry, &specs).expect("should build");

    assert!(built.warnings().is_empty());
}

#[test]
fn a_producer_only_an_artefact_sink_reads_survives_optimisation() {
    let registry = testing::registry();
    let specs = specs_with_a_producer_only();
    let mut graph = build(&registry, &specs).expect("should build").into_graph();

    crate::optimise(&mut graph, &crate::default_passes());
    let grain = node_id(&graph, "grain");

    assert_eq!(
        graph.artefact_sinks()[0].selected(),
        &[(grain, "table"), (grain, "cuts")]
    );
}

#[test]
fn selections_follow_their_nodes_when_others_are_dropped() {
    let registry = testing::registry();
    let mut specs = specs_with_a_producer_only();
    let mut spare = testing::spec(StageKind::Filter, "spare", "test/blur");
    spare.inputs = vec![testing::positional("input")];
    specs.insert(1, spare);
    let mut graph = build(&registry, &specs).expect("should build").into_graph();

    crate::optimise(&mut graph, &crate::default_passes());
    let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();
    let grain = node_id(&graph, "grain");

    assert!(!names.contains(&"spare"));
    assert_eq!(graph.artefact_sinks()[0].selected()[0].0, grain);
}

#[test]
fn two_identical_producers_are_never_merged() {
    let registry = testing::registry();
    let mut specs = specs_with_a_producer_only();
    let mut twin = testing::spec(StageKind::Filter, "twin", "test/grain");
    twin.inputs = vec![testing::positional("input")];
    specs.insert(2, twin);
    specs[4].inputs.push(testing::positional("twin"));
    let mut graph = build(&registry, &specs).expect("should build").into_graph();

    crate::optimise(&mut graph, &crate::default_passes());
    let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();

    assert!(names.contains(&"grain"));
    assert!(names.contains(&"twin"));
    assert_eq!(graph.artefact_sinks()[0].selected().len(), 4);
}
