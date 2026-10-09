use av_toolbox_graph::{CompiledGraph, Routing, Spanned, StageSpec};
use serde_json::json;

use crate::engine::tests::harness::{node_id, share_a_region};
use crate::engine::tests::specs::{delay_spec, per_frame_flow_recording_spec};
use crate::engine::tests::{
    FLOWS,
    assert_every_thread_exited,
    build_engine,
    clear_flows,
    clear_matched,
    engine_from,
    filter_spec,
    flow_recording_spec,
    hold_workers,
    marking_spec,
    matched,
    matching_spec,
    planned_graph,
    planned_width,
    routing_of,
    run_driven_with_watchdog,
    run_with_watchdog,
    sink_spec,
    source_spec,
};
use crate::{Progress, Settings};

fn flows() -> Vec<String> {
    FLOWS.lock().expect("unpoisoned").clone()
}

/// The steps each named sink took, in the order `names` lists them.
fn steps_by_name(graph: &CompiledGraph, progress: &Progress, names: &[&str]) -> Vec<u64> {
    let sinks = progress.sinks();
    let mut steps = Vec::with_capacity(names.len());

    for name in names {
        let node = node_id(graph, name);
        let row = sinks
            .iter()
            .find(|sink| sink.node == node)
            .unwrap_or_else(|| panic!("the progress has no row for {name}"));
        steps.push(row.steps);
    }

    steps
}

/// The frames counted by a `chunk` entry, read from its value.
fn chunk_frames(entry: &str) -> u64 {
    let value = entry
        .strip_prefix("chunk filter.mark ")
        .unwrap_or_else(|| panic!("{entry} is not a chunk record from mark"));

    value.parse().expect("a chunk record counts its frames")
}

#[test]
fn records_reach_a_meta_sink_in_place_through_a_whole_stream_stage() {
    let property = "records_reach_a_meta_sink_in_place_through_a_whole_stream_stage";
    clear_flows();
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        marking_spec("mark", "input", 1, false),
        flow_recording_spec("log", "mark"),
    ];

    let engine = build_engine(&specs, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "chunk filter.mark 2",
    ];

    assert_eq!(flows(), expected);
    assert_eq!(report.frames, 2);
    assert_eq!(report.records, 3);
    assert_every_thread_exited(property);
}

/// Chunk 0 holds frames 0 and 1, chunk 1 frames 2 to 4 and chunk 2 frame 5. Each chunk runs on one
/// worker, which marks every frame and ends with that chunk's count, and the aggregate reads the
/// chunks back in order.
#[test]
fn records_keep_chunk_order_across_two_chunk_routed_workers() {
    let property = "records_keep_chunk_order_across_two_chunk_routed_workers";
    clear_flows();
    let specs = vec![
        source_spec("input", &[0, 0, 1, 1, 1, 2], 0),
        marking_spec("mark", "input", 2, true),
        flow_recording_spec("log", "mark"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "mark"), Routing::ByChunk);
    assert_eq!(planned_width(&graph, "mark"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "chunk filter.mark 2",
        "frame 2",
        "mark filter.mark 2",
        "frame 3",
        "mark filter.mark 3",
        "frame 4",
        "mark filter.mark 4",
        "chunk filter.mark 3",
        "frame 5",
        "mark filter.mark 5",
        "chunk filter.mark 1",
    ];

    assert_eq!(flows(), expected);
    assert_eq!(report.frames, 6);
    assert_eq!(report.records, 9);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// Each frame is its own unit, so its `mark` travels with it. Each worker ends once, as the stream
/// does, so the two `chunk` records come last. Which worker took how many frames depends on which
/// was free, so only their total is fixed.
#[test]
fn records_stay_with_their_unit_in_a_frame_routed_region() {
    let property = "records_stay_with_their_unit_in_a_frame_routed_region";
    clear_flows();
    let mut marking = marking_spec("mark", "input", 2, false);
    marking.args = Spanned::unknown(json!({ "per_frame": true }));

    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        marking,
        flow_recording_spec("log", "mark"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "mark"), Routing::ByFrame);
    assert_eq!(planned_width(&graph, "mark"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let seen = flows();
    let expected_units = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "frame 2",
        "mark filter.mark 2",
        "frame 3",
        "mark filter.mark 3",
    ];

    assert_eq!(seen.len(), 10);
    assert_eq!(seen[..8], expected_units);

    let counted: u64 = seen[8..].iter().map(|entry| chunk_frames(entry)).sum();
    assert_eq!(counted, 4);

    assert_eq!(report.frames, 4);
    assert_eq!(report.records, 6);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// The width two `pass` splits `mark` from `log`, so every record crosses a disperse, an aggregate and
/// the channel into `log`'s own region. `mark 2` and the closing `chunk` arrive after the last frame,
/// so they ride the first closing unit and still come out last.
#[test]
fn records_cross_a_region_boundary_in_place() {
    let property = "records_cross_a_region_boundary_in_place";
    clear_flows();
    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        marking_spec("mark", "input", 1, false),
        filter_spec("pass", "mark", Some(2)),
        flow_recording_spec("log", "pass"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "mark", "pass"));
    assert!(!share_a_region(&graph, "pass", "log"));

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "frame 2",
        "mark filter.mark 2",
        "chunk filter.mark 3",
    ];

    assert_eq!(flows(), expected);
    assert_eq!(report.frames, 3);
    assert_eq!(report.records, 4);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

fn counting_specs(marked: bool) -> Vec<StageSpec> {
    let chunks = [0, 0, 1, 1, 1, 2];

    if !marked {
        return vec![
            source_spec("input", &chunks, 0),
            sink_spec("out", "input"),
            flow_recording_spec("log", "input"),
        ];
    }

    vec![
        source_spec("input", &chunks, 0),
        marking_spec("mark", "input", 2, true),
        sink_spec("out", "mark"),
        flow_recording_spec("log", "mark"),
    ]
}

#[test]
fn records_change_no_frame_count() {
    let property = "records_change_no_frame_count";
    clear_flows();
    let sinks = ["out", "log"];

    let (plain_graph, plain_solved) = planned_graph(&counting_specs(false));
    let plain_engine = engine_from(&plain_graph, &plain_solved, Settings::default());
    let plain_progress = plain_engine.progress();
    let plain_report = run_with_watchdog(plain_engine, property);

    let (marked_graph, marked_solved) = planned_graph(&counting_specs(true));
    let marked_engine = engine_from(&marked_graph, &marked_solved, Settings::default());
    let marked_progress = marked_engine.progress();
    let marked_report = run_with_watchdog(marked_engine, property);

    let plain_steps = steps_by_name(&plain_graph, &plain_progress, &sinks);
    let marked_steps = steps_by_name(&marked_graph, &marked_progress, &sinks);

    assert_eq!(plain_report.frames, 12);
    assert_eq!(marked_report.frames, plain_report.frames);
    assert_eq!(plain_steps, vec![6, 6]);
    assert_eq!(marked_steps, plain_steps);
    assert_eq!(marked_progress.read(), plain_progress.read());
    assert_eq!(plain_report.records, 0);
    assert_eq!(marked_report.records, 9);
    assert_every_thread_exited(property);
}

/// `mark` and `pass` both read `input`, so `match` is fed records on its first slot only. Each record
/// rides the next bundle, and `mark 3` and the closing `chunk` ride the first closing unit.
#[test]
fn a_record_reaching_an_exchanged_region_on_one_slot_is_not_paired() {
    let property = "a_record_reaching_an_exchanged_region_on_one_slot_is_not_paired";
    clear_matched();
    clear_flows();

    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        marking_spec("mark", "input", 1, false),
        filter_spec("pass", "input", None),
        matching_spec("match", "mark", "pass", 2),
        flow_recording_spec("log", "match"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "mark", "match"));
    assert_eq!(planned_width(&graph, "match"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "frame 2",
        "mark filter.mark 2",
        "frame 3",
        "mark filter.mark 3",
        "chunk filter.mark 4",
    ];
    let mut pairs = matched();
    pairs.sort();

    assert_eq!(flows(), expected);
    assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
    assert_eq!(report.frames, 4);
    assert_eq!(report.records, 5);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// Both of `match`'s slots read `mark`, so every record reaches its disperse twice and leaves once.
#[test]
fn a_record_reaching_an_exchanged_region_on_two_slots_leaves_once() {
    let property = "a_record_reaching_an_exchanged_region_on_two_slots_leaves_once";
    clear_matched();
    clear_flows();

    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        marking_spec("mark", "input", 1, false),
        matching_spec("match", "mark", "mark", 2),
        flow_recording_spec("log", "match"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "mark", "match"));

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "frame 2",
        "mark filter.mark 2",
        "chunk filter.mark 3",
    ];
    let mut pairs = matched();
    pairs.sort();

    assert_eq!(flows(), expected);
    assert_eq!(pairs, vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(report.records, 4);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// `delay` heads its own region and holds one frame back, so it passes `mark k` on ahead of `frame k`.
/// The copy on `match`'s second slot is seen before bundle `k` and the copy on its first slot before
/// bundle `k + 1`. `mark 7` and the closing `chunk` leave `delay` after its end flush, behind `frame 7`,
/// so they ride the first closing unit. This checks where each record lands. `log` sits in a region of
/// its own, whose head would drop a second copy anyway.
#[test]
fn a_record_reaching_an_exchanged_region_by_a_lagging_path_leaves_once() {
    let property = "a_record_reaching_an_exchanged_region_by_a_lagging_path_leaves_once";
    clear_matched();
    clear_flows();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        marking_spec("mark", "input", 1, false),
        delay_spec("delay", "mark", &[1], 1, false),
        matching_spec("match", "mark", "delay", 2),
        flow_recording_spec("log", "match"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "mark", "delay"));
    assert!(!share_a_region(&graph, "mark", "match"));
    assert_eq!(routing_of(&graph, "match"), Routing::ByFrame);
    assert_eq!(planned_width(&graph, "match"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    let mut expected: Vec<String> = Vec::new();
    for index in 0..7 {
        expected.push(format!("mark filter.mark {index}"));
        expected.push(format!("frame {index}"));
    }
    expected.push(String::from("frame 7"));
    expected.push(String::from("mark filter.mark 7"));
    expected.push(String::from("chunk filter.mark 8"));

    let mut pairs = matched();
    pairs.sort();
    let expected_pairs: Vec<(u64, u64)> = (0..8).map(|second| (second, second)).collect();

    assert_eq!(flows(), expected);
    assert_eq!(pairs, expected_pairs);
    assert_eq!(report.frames, 8);
    assert_eq!(report.records, 9);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// `match` is one worker wide with two inputs, so its region has a disperse and no aggregate, and
/// whatever it produces leaves through the relay into the width two `pass`.
#[test]
fn records_cross_a_relay_in_place() {
    let property = "records_cross_a_relay_in_place";
    clear_matched();
    clear_flows();

    let specs = vec![
        source_spec("base", &[0, 0, 0], 0),
        source_spec("overlay", &[0, 0, 0], 1),
        marking_spec("mark", "base", 1, false),
        matching_spec("match", "mark", "overlay", 1),
        filter_spec("pass", "match", Some(2)),
        flow_recording_spec("log", "pass"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "mark", "match"));
    assert!(!share_a_region(&graph, "match", "pass"));
    assert_eq!(planned_width(&graph, "match"), 1);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let expected = vec![
        "frame 0",
        "mark filter.mark 0",
        "frame 1",
        "mark filter.mark 1",
        "frame 2",
        "mark filter.mark 2",
        "chunk filter.mark 3",
    ];

    assert_eq!(flows(), expected);
    assert_eq!(matched(), vec![(0, 0), (1, 1), (2, 2)]);
    assert_eq!(report.frames, 3);
    assert_eq!(report.records, 4);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// The same lagging layout, with `log` declared per frame so each of `match`'s two workers runs its own.
///
/// Both workers are held inside their first `match` call until each has been given a bundle, so bundle 0
/// goes to one worker and bundle 1 to the other. The copy of `mark 0` on the lagging slot goes ahead of
/// bundle 0 and the copy on the direct slot ahead of bundle 1, and nothing after the workers drops a
/// second copy. Only the disperse's own check keeps `mark 0` out of the second `log`. The two `log`s
/// interleave, so the test reads what they saw without its order.
#[test]
fn a_record_reaching_a_widened_meta_sink_by_a_lagging_path_is_logged_once() {
    let property = "a_record_reaching_a_widened_meta_sink_by_a_lagging_path_is_logged_once";
    clear_matched();
    clear_flows();

    let held = hold_workers(2);
    let specs = vec![
        source_spec("input", &[0; 8], 0),
        marking_spec("mark", "input", 1, false),
        delay_spec("delay", "mark", &[1], 1, false),
        matching_spec("match", "mark", "delay", 2),
        per_frame_flow_recording_spec("log", "match"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(share_a_region(&graph, "match", "log"));
    assert_eq!(routing_of(&graph, "match"), Routing::ByFrame);
    assert_eq!(planned_width(&graph, "log"), 2);

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });
    let report = result.expect("the run should finish once both workers are released");

    let mut expected: Vec<String> = Vec::new();
    for index in 0..8 {
        expected.push(format!("frame {index}"));
        expected.push(format!("mark filter.mark {index}"));
    }
    expected.push(String::from("chunk filter.mark 8"));
    expected.sort();

    let mut seen = flows();
    seen.sort();

    assert_eq!(
        arrived, 2,
        "{property}: only {arrived} of the two workers were given a bundle"
    );
    assert_eq!(seen, expected);
    assert_eq!(report.frames, 8);
    assert_eq!(report.records, 9);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}
