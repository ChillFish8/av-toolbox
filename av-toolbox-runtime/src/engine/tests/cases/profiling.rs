use std::sync::Arc;
use std::time::Duration;

use crate::engine::tests::harness::{
    build_engine,
    engine_from,
    node_id,
    planned_graph,
    planned_width,
    run_with_watchdog,
    share_a_region,
};
use crate::engine::tests::specs::{filter_spec, sink_spec, source_spec};
use crate::engine::tests::support::format;
use crate::{Profiling, Settings};

fn profiled(warmup: u64) -> Settings {
    let profiling = Profiling {
        sync: Arc::new(|| Ok(())),
        warmup,
    };

    Settings {
        profiling: Some(profiling),
        ..Settings::default()
    }
}

#[test]
fn a_profiled_run_times_every_node() {
    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let engine = engine_from(&graph, &solved, profiled(1));

    let report = run_with_watchdog(engine, "a_profiled_run_times_every_node");

    let input = node_id(&graph, "input");
    let filter = node_id(&graph, "filter");
    let source_timing = report
        .timings
        .iter()
        .find(|timing| timing.node == input)
        .expect("the source is timed");
    let filter_timing = report
        .timings
        .iter()
        .find(|timing| timing.node == filter)
        .expect("the filter is timed");
    assert_eq!(report.timings.len(), 3);
    assert_eq!(source_timing.frames, 3);
    assert_eq!(filter_timing.frames, 3);
}

/// The source's budget only has room for one frame, so it waits on the other regions to drop it.
#[test]
fn a_profiled_source_waiting_for_room_lets_the_other_regions_run() {
    let property = "a_profiled_source_waiting_for_room_lets_the_other_regions_run";
    let specs = vec![
        source_spec("input", &[0; 8], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let settings = Settings {
        frame_budget: Some(format().total_bytes()),
        stall_deadline: Some(Duration::from_secs(2)),
        ..profiled(0)
    };
    let engine = engine_from(&graph, &solved, settings);

    assert_eq!(planned_width(&graph, "filter"), 2);
    assert!(!share_a_region(&graph, "input", "filter"));

    let report = run_with_watchdog(engine, property);

    let input = node_id(&graph, "input");
    let source_timing = report
        .timings
        .iter()
        .find(|timing| timing.node == input)
        .expect("the source is timed");
    assert_eq!(report.frames, 8);
    assert_eq!(source_timing.frames, 8);
}

#[test]
fn an_unprofiled_run_records_no_timings() {
    let specs = vec![source_spec("input", &[0, 0], 0), sink_spec("out", "input")];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "an_unprofiled_run_records_no_timings");

    assert!(report.timings.is_empty());
}
