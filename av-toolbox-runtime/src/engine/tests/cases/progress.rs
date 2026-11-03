use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use av_toolbox_graph::NodeId;

use crate::engine::tests::harness::{build_engine, engine_from, node_id, planned_graph, run_with_watchdog};
use crate::engine::tests::specs::{filter_spec, meta_sink_spec, pairing_spec, sink_spec, source_spec};
use crate::{Settings, SinkProgress};

fn counts(sinks: &[SinkProgress]) -> Vec<(NodeId, u64)> {
    sinks.iter().map(|sink| (sink.node, sink.steps)).collect()
}

#[test]
fn every_sink_has_a_row_at_zero_before_the_run() {
    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        sink_spec("out", "input"),
        meta_sink_spec("log", "input"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let engine = engine_from(&graph, &solved, Settings::default());

    let progress = engine.progress();
    let mut expected = vec![(node_id(&graph, "out"), 0), (node_id(&graph, "log"), 0)];
    expected.sort();

    assert_eq!(counts(&progress.sinks()), expected);
    assert_eq!(progress.read(), 0);

    run_with_watchdog(engine, "every_sink_has_a_row_at_zero_before_the_run");
}

#[test]
fn each_sink_counts_the_frames_it_received() {
    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
        meta_sink_spec("log", "input"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let engine = engine_from(&graph, &solved, Settings::default());
    let progress = engine.progress();

    run_with_watchdog(engine, "each_sink_counts_the_frames_it_received");

    let mut expected = vec![(node_id(&graph, "out"), 3), (node_id(&graph, "log"), 3)];
    expected.sort();

    assert_eq!(counts(&progress.sinks()), expected);
}

#[test]
fn a_two_input_sink_counts_one_per_step() {
    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &chunks, 0),
        source_spec("overlay_input", &chunks, 1),
        pairing_spec("out", "base_input", "overlay_input"),
    ];
    let engine = build_engine(&specs, Settings::default());
    let progress = engine.progress();

    let report = run_with_watchdog(engine, "a_two_input_sink_counts_one_per_step");

    let sinks = progress.sinks();
    assert_eq!(sinks.len(), 1);
    assert_eq!(sinks[0].steps, 3);
    assert_eq!(report.frames, 6);
}

#[test]
fn a_sink_after_a_widened_filter_counts_every_frame() {
    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());
    let progress = engine.progress();

    run_with_watchdog(engine, "a_sink_after_a_widened_filter_counts_every_frame");

    let sinks = progress.sinks();
    assert_eq!(sinks.len(), 1);
    assert_eq!(sinks[0].steps, 4);
}

#[test]
fn read_counts_every_source_together() {
    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &chunks, 0),
        source_spec("overlay_input", &chunks, 1),
        pairing_spec("out", "base_input", "overlay_input"),
    ];
    let engine = build_engine(&specs, Settings::default());
    let progress = engine.progress();

    run_with_watchdog(engine, "read_counts_every_source_together");

    assert_eq!(progress.read(), 6);
}

#[test]
fn read_stops_at_the_frame_limit() {
    let specs = vec![
        source_spec("input", &[0, 0, 0, 0, 0], 0),
        sink_spec("out", "input"),
    ];
    let settings = Settings {
        frame_limit: Some(2),
        ..Settings::default()
    };
    let engine = build_engine(&specs, settings);
    let progress = engine.progress();

    run_with_watchdog(engine, "read_stops_at_the_frame_limit");

    assert_eq!(progress.read(), 2);
}

#[test]
fn elapsed_is_zero_until_the_run_starts() {
    let specs = vec![source_spec("input", &[0, 0, 0], 0), sink_spec("out", "input")];
    let engine = build_engine(&specs, Settings::default());
    let progress = engine.progress();

    assert_eq!(progress.elapsed(), Duration::ZERO);

    run_with_watchdog(engine, "elapsed_is_zero_until_the_run_starts");

    assert!(progress.elapsed() > Duration::ZERO);
}

#[test]
fn the_first_output_action_runs_once_across_every_sink() {
    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        sink_spec("out", "input"),
        meta_sink_spec("log", "input"),
    ];
    let engine = build_engine(&specs, Settings::default());
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = runs.clone();

    engine.progress().on_first_output(move || {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    run_with_watchdog(engine, "the_first_output_action_runs_once_across_every_sink");

    assert_eq!(runs.load(Ordering::SeqCst), 1);
}

#[test]
fn a_run_with_no_output_drops_the_first_output_action() {
    let specs = vec![source_spec("input", &[], 0), sink_spec("out", "input")];
    let engine = build_engine(&specs, Settings::default());
    let progress = engine.progress();
    let held = Arc::new(());
    let captured = held.clone();

    progress.on_first_output(move || drop(captured));
    run_with_watchdog(engine, "a_run_with_no_output_drops_the_first_output_action");

    assert_eq!(Arc::strong_count(&held), 1, "a live Progress kept the action");
    drop(progress);
}
