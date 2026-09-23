use std::sync::atomic::Ordering;

use av_toolbox_core::Crossing;
use av_toolbox_graph::NodeRole;

use crate::Settings;
use crate::engine::tests as engine;

fn crossings_of(graph: &av_toolbox_graph::CompiledGraph, crossing: Crossing) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|node| node.role() == NodeRole::Crossing(crossing))
        .count()
}

#[test]
fn a_host_source_through_a_device_filter_to_a_host_sink_delivers_every_frame_in_order() {
    let property = "a_host_source_through_a_device_filter_to_a_host_sink_delivers_every_frame_in_order";
    engine::clear_recorded();

    let specs = vec![
        engine::host_source_spec("input", &[0, 0, 0, 0, 0, 0], 0),
        engine::filter_spec("filter", "input", None),
        engine::host_sink_spec("out", "filter"),
    ];
    let (graph, solved) = engine::planned_graph(&specs);
    assert_eq!(crossings_of(&graph, Crossing::Upload), 1);
    assert_eq!(crossings_of(&graph, Crossing::Unload), 1);

    let engine = engine::engine_from(&graph, &solved, Settings::default());
    let report = engine::run_with_watchdog(engine, property);

    assert_eq!(report.frames, 6);
    assert_eq!(engine::recorded_indices(), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn a_widened_region_ending_in_an_unload_still_emits_in_order() {
    let property = "a_widened_region_ending_in_an_unload_still_emits_in_order";
    engine::clear_recorded();

    let specs = vec![
        engine::source_spec("input", &[0; 32], 0),
        engine::filter_spec("filter", "input", Some(4)),
        engine::host_sink_spec("out", "filter"),
    ];
    let (graph, solved) = engine::planned_graph(&specs);
    assert_eq!(engine::planned_width(&graph, "filter"), 4);

    let engine = engine::engine_from(&graph, &solved, Settings::default());
    let report = engine::run_with_watchdog(engine, property);

    assert_eq!(report.frames, 32);
    assert_eq!(engine::recorded_indices(), (0..32).collect::<Vec<_>>());
}

#[test]
fn a_round_trip_through_a_host_filter_still_runs() {
    let property = "a_round_trip_through_a_host_filter_still_runs";
    engine::clear_recorded();

    let specs = vec![
        engine::source_spec("input", &[0, 0, 0, 0], 0),
        engine::host_pass_spec("detect", "input", None),
        engine::filter_spec("filter", "detect", None),
        engine::sink_spec("out", "filter"),
    ];

    let report = engine::run_with_watchdog(engine::build_engine(&specs, Settings::default()), property);

    assert_eq!(report.frames, 4);
}

#[test]
fn a_meta_transform_runs_on_whichever_side_its_input_is_on() {
    let property = "a_meta_transform_runs_on_whichever_side_its_input_is_on";
    engine::clear_recorded();

    let specs = vec![
        engine::host_source_spec("input", &[0, 0, 1, 1], 0),
        engine::restamp_spec("restamp", "input"),
        engine::host_sink_spec("out", "restamp"),
    ];
    let (graph, solved) = engine::planned_graph(&specs);
    assert_eq!(crossings_of(&graph, Crossing::Upload), 0);

    let engine = engine::engine_from(&graph, &solved, Settings::default());
    let report = engine::run_with_watchdog(engine, property);

    assert_eq!(report.frames, 4);
}

#[test]
fn a_host_source_under_a_tight_budget_still_finishes() {
    let property = "a_host_source_under_a_tight_budget_still_finishes";
    engine::clear_recorded();
    engine::SOURCE_BUDGET_PEAK.store(0, Ordering::SeqCst);
    let frame = engine::format().total_bytes();

    let specs = vec![
        engine::host_source_spec("input", &[0; 24], 0),
        engine::filter_spec("filter", "input", None),
        engine::host_sink_spec("out", "filter"),
    ];
    let settings = Settings {
        frame_budget: Some(3 * frame),
        ..Settings::default()
    };

    let report = engine::run_with_watchdog(engine::build_engine(&specs, settings), property);

    assert_eq!(report.frames, 24);
    assert!(engine::SOURCE_BUDGET_PEAK.load(Ordering::SeqCst) <= 3 * frame);
}

#[test]
fn a_widened_stage_feeding_a_device_sink_and_a_host_sink_delivers_every_frame_to_both() {
    let property = "a_widened_stage_feeding_a_device_sink_and_a_host_sink_delivers_every_frame_to_both";
    engine::clear_recorded();

    let specs = vec![
        engine::source_spec("input", &[0; 16], 0),
        engine::filter_spec("filter", "input", Some(4)),
        engine::sink_spec("gpu", "filter"),
        engine::host_sink_spec("host", "filter"),
    ];
    let (graph, solved) = engine::planned_graph(&specs);
    assert_eq!(crossings_of(&graph, Crossing::Unload), 1);

    let engine = engine::engine_from(&graph, &solved, Settings::default());
    let report = engine::run_with_watchdog(engine, property);

    let mut delivered = engine::recorded_indices();
    delivered.sort_unstable();
    let expected: Vec<u64> = (0..16).flat_map(|index| [index, index]).collect();
    assert_eq!(report.frames, 32, "every frame counts once at each sink");
    assert_eq!(delivered, expected);
}
