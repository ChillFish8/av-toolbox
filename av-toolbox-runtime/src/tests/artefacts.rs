use av_toolbox_core::ArtefactData;
use av_toolbox_gpu::testing_context;
use av_toolbox_graph::{CompiledGraph, NodeId, StageBox};
use serde_json::json;

use crate::RunError;
use crate::artefacts::ArtefactCollector;
use crate::engine::tests::{
    ArtefactingArgs,
    ArtefactingInstance,
    WANTED,
    artefacting_spec,
    collecting_spec,
    planned_graph,
    sink_spec,
    source_spec,
};

fn graph_with(mode: &str, include: &[&str]) -> (CompiledGraph, NodeId) {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        artefacting_spec("count", "input", 1, false, mode),
        sink_spec("out", "count"),
        collecting_spec("files", "count", include),
    ];
    let (graph, _) = planned_graph(&specs);
    let count = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "count")
        .expect("a count node")
        .id();

    (graph, count)
}

fn instance(mode: &str) -> StageBox {
    let args: ArtefactingArgs = serde_json::from_value(json!({ "mode": mode })).expect("valid args");
    let built = ArtefactingInstance::from_args(args);

    StageBox::Filter(Box::new(built))
}

#[test]
fn a_node_is_told_what_is_wanted_of_it() {
    let (graph, count) = graph_with("counting", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("counting");

    collector.announce(count, &mut stage).expect("accepts");

    assert_eq!(*WANTED.lock().expect("unpoisoned"), vec![true]);
}

#[test]
fn a_node_wanted_for_nothing_is_still_told() {
    let (graph, count) = graph_with("counting", &["broken"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("counting");

    collector.announce(count, &mut stage).expect("accepts");

    assert_eq!(*WANTED.lock().expect("unpoisoned"), vec![false]);
}

#[test]
fn a_wanted_piece_is_kept_and_merged_under_its_stage_path() {
    let (graph, count) = graph_with("counting", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("counting");
    let mut context = testing_context();

    collector
        .take(count, Some(3), &mut stage, &mut context)
        .expect("takes");
    let merged = collector.merge().expect("merges");
    let artefact = &merged[&(count, "frames")];

    assert_eq!(artefact.key(), "filter.count.frames");
    assert_eq!(artefact.data, ArtefactData::Structured(json!([[3, 0]])));
}

#[test]
fn an_unwanted_piece_is_dropped() {
    let (graph, count) = graph_with("counting", &["broken"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("counting");
    let mut context = testing_context();

    collector
        .take(count, None, &mut stage, &mut context)
        .expect("takes");
    let merged = collector.merge().expect("merges");

    assert!(merged.is_empty());
}

#[test]
fn an_undeclared_artefact_fails_naming_its_node() {
    let (graph, count) = graph_with("undeclared", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("undeclared");
    let mut context = testing_context();

    let error = collector
        .take(count, None, &mut stage, &mut context)
        .expect_err("refused");

    assert!(matches!(error, RunError::Stage { node, .. } if node == count));
    assert!(error.to_string().contains("surprise"));
}

#[test]
fn a_piece_of_the_wrong_kind_fails_at_the_merge_naming_its_node() {
    let (graph, count) = graph_with("wrong_kind", &["bytes"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("wrong_kind");
    let mut context = testing_context();

    collector
        .take(count, None, &mut stage, &mut context)
        .expect("takes");
    let error = collector.merge().expect_err("refused");

    assert!(matches!(error, RunError::Stage { node, .. } if node == count));
    assert!(error.to_string().contains("bytes"));
}

#[test]
fn a_failing_merge_fails_naming_its_node() {
    let (graph, count) = graph_with("broken", &["broken"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("broken");
    let mut context = testing_context();

    collector
        .take(count, None, &mut stage, &mut context)
        .expect("takes");
    let error = collector.merge().expect_err("refused");

    assert!(matches!(error, RunError::Stage { node, .. } if node == count));
}

#[test]
fn a_discarded_piece_is_never_merged() {
    let (graph, count) = graph_with("counting", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("counting");
    let mut context = testing_context();

    collector
        .discard(count, &mut stage, &mut context)
        .expect("discards");
    let merged = collector.merge().expect("merges");

    assert!(merged.is_empty());
}

#[test]
fn a_discarded_undeclared_artefact_still_fails_naming_its_node() {
    let (graph, count) = graph_with("undeclared", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("undeclared");
    let mut context = testing_context();

    let error = collector
        .discard(count, &mut stage, &mut context)
        .expect_err("refused");

    assert!(matches!(error, RunError::Stage { node, .. } if node == count));
}

#[test]
fn a_structured_piece_merges_into_an_opaque_artefact() {
    let (graph, count) = graph_with("tabled", &["table"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("tabled");
    let mut context = testing_context();

    collector
        .take(count, Some(0), &mut stage, &mut context)
        .expect("takes");
    let merged = collector.merge().expect("merges");
    let artefact = &merged[&(count, "table")];

    assert_eq!(artefact.data, ArtefactData::Opaque(b"0\n".to_vec()));
}

#[test]
fn a_merge_of_the_wrong_kind_fails_naming_its_node() {
    let (graph, count) = graph_with("mismerged", &["mismerged"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("mismerged");
    let mut context = testing_context();

    collector
        .take(count, None, &mut stage, &mut context)
        .expect("takes");
    let error = collector.merge().expect_err("refused");

    assert!(matches!(error, RunError::Stage { node, .. } if node == count));
    assert!(error.to_string().contains("mismerged"));
}

#[test]
fn a_refusal_is_handed_back() {
    let (graph, count) = graph_with("refusing", &["frames"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("refusing");

    let error = collector.announce(count, &mut stage).expect_err("refused");

    assert!(error.to_string().contains("cannot produce frames"));
}

#[test]
fn a_stage_wanted_for_nothing_it_refuses_accepts() {
    let (graph, count) = graph_with("refusing", &["broken"]);
    let collector = ArtefactCollector::new(&graph);
    let mut stage = instance("refusing");

    collector.announce(count, &mut stage).expect("accepts");
}
