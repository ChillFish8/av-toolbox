use serde_json::json;

use crate::engine::tests::{
    COLLECTED,
    WANTED,
    artefacting_spec,
    assert_every_thread_exited,
    build_engine,
    collecting_spec,
    engine_from,
    planned_graph,
    planned_width,
    run_result_with_watchdog,
    run_with_watchdog,
    sink_spec,
    source_spec,
};
use crate::{RunError, Settings};

fn collected() -> Vec<(String, serde_json::Value)> {
    COLLECTED.lock().expect("unpoisoned").clone()
}

#[test]
fn a_whole_stream_stage_hands_over_one_piece() {
    let property = "a_whole_stream_stage_hands_over_one_piece";
    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        artefacting_spec("count", "input", 1, false, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];

    let engine = build_engine(&specs, Settings::default());
    let report = run_with_watchdog(engine, property);

    assert_eq!(
        collected(),
        vec![(String::from("filter.count.frames"), json!([[null, 3]]))]
    );
    assert_eq!(report.artefacts, 1);
    assert_every_thread_exited(property);
}

#[test]
fn pieces_from_a_chunked_stage_on_two_workers_merge_in_chunk_order() {
    let property = "pieces_from_a_chunked_stage_on_two_workers_merge_in_chunk_order";
    let specs = vec![
        source_spec("input", &[0, 0, 1, 1, 1, 2, 3, 3], 0),
        artefacting_spec("count", "input", 2, true, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(planned_width(&graph, "count"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = json!([[0, 2], [1, 3], [2, 1], [3, 2]]);

    assert_eq!(collected(), vec![(String::from("filter.count.frames"), expected)]);
    assert_eq!(report.artefacts, 1);
    assert_every_thread_exited(property);
}

#[test]
fn a_chunked_worker_given_no_chunk_hands_over_nothing() {
    let property = "a_chunked_worker_given_no_chunk_hands_over_nothing";
    let specs = vec![
        source_spec("input", &[0, 0, 1], 0),
        artefacting_spec("count", "input", 4, true, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(planned_width(&graph, "count"), 4);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = json!([[0, 2], [1, 1]]);

    assert_eq!(collected(), vec![(String::from("filter.count.frames"), expected)]);
    assert_eq!(report.artefacts, 1);
    assert_every_thread_exited(property);
}

#[test]
fn every_worker_is_told_what_is_wanted() {
    let property = "every_worker_is_told_what_is_wanted";
    let specs = vec![
        source_spec("input", &[0, 1, 2, 3], 0),
        artefacting_spec("count", "input", 2, true, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];

    let engine = build_engine(&specs, Settings::default());
    run_with_watchdog(engine, property);

    assert_eq!(*WANTED.lock().expect("unpoisoned"), vec![true, true]);
}

#[test]
fn a_failing_merge_fails_the_run() {
    let property = "a_failing_merge_fails_the_run";
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        artefacting_spec("count", "input", 1, false, "broken"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.broken"]),
    ];

    let engine = build_engine(&specs, Settings::default());
    let result = run_result_with_watchdog(engine, property);

    assert!(matches!(result, Err(RunError::Stage { .. })));
    assert!(collected().is_empty());
}

#[test]
fn a_cancelled_run_hands_nothing_to_an_artefact_sink() {
    let property = "a_cancelled_run_hands_nothing_to_an_artefact_sink";
    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        artefacting_spec("count", "input", 1, false, "counting"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.frames"]),
    ];

    let engine = build_engine(&specs, Settings::default());
    engine.cancel_handle().stop();
    let report = run_with_watchdog(engine, property);

    assert!(report.cancelled);
    assert_eq!(report.artefacts, 0);
    assert!(collected().is_empty());
}

#[test]
fn a_failing_artefact_sink_fails_the_run_naming_it() {
    let property = "a_failing_artefact_sink_fails_the_run_naming_it";
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        artefacting_spec("count", "input", 1, false, "opaque"),
        sink_spec("out", "count"),
        collecting_spec("files", &["filter.count.bytes"]),
    ];

    let engine = build_engine(&specs, Settings::default());
    let result = run_result_with_watchdog(engine, property);

    assert!(matches!(result, Err(RunError::ArtefactSink { ref sink, .. }) if sink == "sink.files"));
}
