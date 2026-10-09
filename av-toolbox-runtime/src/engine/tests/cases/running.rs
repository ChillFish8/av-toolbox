use std::sync::atomic::Ordering;
use std::time::Duration;

use av_toolbox_graph::StageSpec;

use crate::engine::tests::fixtures::filters::{PASS_BUILD_CALLS, PASS_HAD_BUDGET};
use crate::engine::tests::fixtures::sinks::PAIRING_STEPS;
use crate::engine::tests::fixtures::sources::SOURCE_BUDGETS;
use crate::engine::tests::fixtures::tags::{TAGGED, Tag};
use crate::engine::tests::harness::{
    build_engine,
    clear_recorded,
    engine_from,
    planned_graph,
    recorded,
    run_with_watchdog,
    share_a_region,
};
use crate::engine::tests::specs::{
    filter_spec,
    host_cut_tag_recording_spec,
    host_pass_spec,
    host_sink_spec,
    host_source_spec,
    meta_sink_spec,
    pairing_spec,
    sink_spec,
    source_spec,
    tag_recording_spec,
};
use crate::{RunReport, Settings};

#[test]
fn a_single_region_graph_runs_end_to_end() {
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "a_single_region_graph_runs_end_to_end");

    assert_eq!(recorded(), vec![0, 1, 2]);
    assert_eq!(report.frames, 3);
}

#[test]
fn an_all_host_chain_runs_end_to_end() {
    clear_recorded();

    let specs = vec![
        host_source_spec("input", &[0, 0, 0], 0),
        host_pass_spec("pass", "input", None),
        host_sink_spec("out", "pass"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "an_all_host_chain_runs_end_to_end");

    assert_eq!(recorded(), vec![0, 1, 2]);
    assert_eq!(report.frames, 3);
}

#[test]
fn a_meta_sinks_frames_are_counted_in_the_report() {
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        meta_sink_spec("log", "input"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "a_meta_sinks_frames_are_counted_in_the_report");

    assert_eq!(recorded(), vec![0, 1, 2]);
    assert_eq!(report.frames, 3);
}

/// Runs five frames from `source` through `pass` into a sink demanding tags, with the tags provider
/// sharing the source's region.
fn tags_every_frame_in_order(source: StageSpec, pass: StageSpec, property: &str) {
    TAGGED.lock().expect("not poisoned").clear();

    let specs = vec![source, pass, tag_recording_spec("out", "pass")];
    let (graph, solved) = planned_graph(&specs);
    assert!(share_a_region(&graph, "input", "tags(input)"));
    assert!(share_a_region(&graph, "tags(input)", "pass"));

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    let tagged = TAGGED.lock().expect("not poisoned").clone();
    let expected: Vec<(u64, Option<Tag>)> = (0..5).map(|timestamp| (timestamp, Some(Tag(true)))).collect();
    assert_eq!(tagged, expected);
    assert_eq!(report.frames, 5);
}

#[test]
fn a_meta_provider_in_its_producers_region_tags_every_device_frame_in_order() {
    tags_every_frame_in_order(
        source_spec("input", &[0, 0, 0, 0, 0], 0),
        filter_spec("pass", "input", None),
        "a_meta_provider_in_its_producers_region_tags_every_device_frame_in_order",
    );
}

#[test]
fn a_meta_provider_in_its_producers_region_tags_every_host_frame_in_order() {
    tags_every_frame_in_order(
        host_source_spec("input", &[0, 0, 0, 0, 0], 0),
        host_pass_spec("pass", "input", None),
        "a_meta_provider_in_its_producers_region_tags_every_host_frame_in_order",
    );
}

#[test]
fn a_meta_provider_in_a_host_providers_region_tags_every_frame_in_order() {
    TAGGED.lock().expect("not poisoned").clear();

    let specs = vec![
        host_source_spec("input", &[0, 0, 1, 1, 1], 0),
        host_pass_spec("pass", "input", None),
        host_cut_tag_recording_spec("out", "pass"),
    ];
    let (graph, solved) = planned_graph(&specs);
    assert!(share_a_region(&graph, "host-fixture-cuts(input)", "tags(input)"));
    assert!(!share_a_region(&graph, "input", "tags(input)"));
    assert!(!share_a_region(&graph, "tags(input)", "pass"));

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(
        engine,
        "a_meta_provider_in_a_host_providers_region_tags_every_frame_in_order",
    );

    let tagged = TAGGED.lock().expect("not poisoned").clone();
    let expected: Vec<(u64, Option<Tag>)> = (0..5).map(|timestamp| (timestamp, Some(Tag(true)))).collect();
    assert_eq!(tagged, expected);
    assert_eq!(report.frames, 5);
}

#[test]
fn a_widened_region_runs_every_frame_exactly_once() {
    clear_recorded();

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_widened_region_runs_every_frame_exactly_once");

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..8).collect::<Vec<u64>>());
}

#[test]
fn output_comes_back_in_input_order_at_any_width() {
    let chunks = vec![0, 0, 1, 1, 2, 2, 3];

    clear_recorded();
    let narrow_specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let narrow = build_engine(&narrow_specs, Settings::default());
    run_with_watchdog(
        narrow,
        "output_comes_back_in_input_order_at_any_width (width one)",
    );
    let narrow_order = recorded();

    clear_recorded();
    let wide_specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let wide = build_engine(&wide_specs, Settings::default());
    run_with_watchdog(wide, "output_comes_back_in_input_order_at_any_width (width four)");
    let wide_order = recorded();

    assert_eq!(narrow_order, vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(narrow_order, wide_order);
}

#[test]
fn every_worker_of_a_region_gets_its_own_instances() {
    PASS_BUILD_CALLS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "every_worker_of_a_region_gets_its_own_instances");

    assert_eq!(PASS_BUILD_CALLS.load(Ordering::SeqCst), 4);
}

#[test]
fn a_source_gets_its_own_share_of_the_budget() {
    SOURCE_BUDGETS.lock().expect("not poisoned").clear();

    let specs = vec![
        source_spec("left", &[], 0),
        sink_spec("left_out", "left"),
        source_spec("right", &[], 1),
        sink_spec("right_out", "right"),
    ];
    let engine = build_engine(
        &specs,
        Settings {
            frame_budget: Some(1000),
            ..Settings::default()
        },
    );

    run_with_watchdog(engine, "a_source_gets_its_own_share_of_the_budget");

    let recorded = SOURCE_BUDGETS.lock().expect("not poisoned").clone();
    let left = recorded
        .iter()
        .find(|(id, _)| *id == 0)
        .map(|(_, limit)| *limit)
        .expect("the left source should have started");
    let right = recorded
        .iter()
        .find(|(id, _)| *id == 1)
        .map(|(_, limit)| *limit)
        .expect("the right source should have started");

    assert_eq!(left, Some(500));
    assert_eq!(right, Some(500));
}

#[test]
fn a_worker_that_is_not_a_source_has_no_budget() {
    *PASS_HAD_BUDGET.lock().expect("not poisoned") = None;

    // Widened so the filter lands in a region of its own, separate from the source's, since a
    // filter left at the source's own width would share its worker's context and therefore its
    // budget, proving nothing about a non-source worker.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_worker_that_is_not_a_source_has_no_budget");

    let had_budget = PASS_HAD_BUDGET
        .lock()
        .expect("not poisoned")
        .expect("the filter should have started");

    assert!(!had_budget);
}

#[test]
fn a_graph_with_no_frames_still_finishes() {
    clear_recorded();

    // Widened, so the filter lands in a region with a disperse, and this run reaches
    // `Disperse::finish` with no chunk ever having been in flight. Left at width one, all
    // three nodes collapse into a single region with no disperse at all, and never touches
    // that path.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "a_graph_with_no_frames_still_finishes");

    assert!(recorded().is_empty());
    assert_eq!(report.frames, 0);
}

/// A sink of two inputs consumes two items every step, and the report counts both of them.
#[test]
fn a_report_counts_every_input_slot_a_sink_consumed() {
    let property = "a_report_counts_every_input_slot_a_sink_consumed";
    PAIRING_STEPS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &chunks, 0),
        source_spec("overlay_input", &chunks, 1),
        pairing_spec("out", "base_input", "overlay_input"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    assert_eq!(PAIRING_STEPS.load(Ordering::SeqCst), 3);
    assert_eq!(
        report.frames, 6,
        "{property}: both of the sink's slots should count"
    );
}

#[test]
fn settings_and_reports_carry_the_derives_a_caller_needs() {
    let settings = Settings::default();
    let report = RunReport {
        frames: 2,
        artefacts: 0,
        cancelled: false,
        dropped: 0,
        records: 0,
        wall: Duration::ZERO,
        counted_wall: Duration::ZERO,
        timings: Vec::new(),
    };

    let copied_settings = settings.clone();
    let copied_report = report.clone();

    assert!(copied_settings.frame_budget.is_none());
    assert!(copied_settings.stall_deadline.is_none());
    assert_eq!(copied_report.frames, 2);
    assert!(format!("{report:?}").contains("frames"));
}
