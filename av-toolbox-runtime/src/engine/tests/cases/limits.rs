use std::time::Duration;

use crate::Settings;
use crate::engine::tests::harness::{build_engine, clear_recorded, recorded, run_with_watchdog};
use crate::engine::tests::specs::{filter_spec, sink_spec, source_spec};

#[test]
fn a_frame_limit_ends_every_source() {
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0, 0, 0], 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let settings = Settings {
        frame_limit: Some(2),
        ..Settings::default()
    };
    let engine = build_engine(&specs, settings);

    let report = run_with_watchdog(engine, "a_frame_limit_ends_every_source");

    assert_eq!(recorded(), vec![0, 1]);
    assert_eq!(report.frames, 2);
}

#[test]
fn without_a_warmup_the_counted_wall_is_the_whole_run() {
    let specs = vec![source_spec("input", &[0, 0, 0], 0), sink_spec("out", "input")];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "without_a_warmup_the_counted_wall_is_the_whole_run");

    assert_eq!(report.counted_wall, report.wall);
}

#[test]
fn a_passed_warmup_counts_part_of_the_run() {
    let specs = vec![source_spec("input", &[0, 0, 0, 0], 0), sink_spec("out", "input")];
    let settings = Settings {
        warmup: 2,
        ..Settings::default()
    };
    let engine = build_engine(&specs, settings);

    let report = run_with_watchdog(engine, "a_passed_warmup_counts_part_of_the_run");

    assert!(report.counted_wall <= report.wall);
    assert!(report.counted_wall > Duration::ZERO);
}

#[test]
fn a_warmup_never_reached_counts_no_wall_time() {
    let specs = vec![source_spec("input", &[0, 0, 0], 0), sink_spec("out", "input")];
    let settings = Settings {
        warmup: 10,
        ..Settings::default()
    };
    let engine = build_engine(&specs, settings);

    let report = run_with_watchdog(engine, "a_warmup_never_reached_counts_no_wall_time");

    assert_eq!(report.counted_wall, Duration::ZERO);
}
