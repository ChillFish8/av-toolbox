use std::sync::mpsc;

use av_toolbox_graph::{NodeId, StageSpec};

use super::engine;
use crate::{RunWarning, Settings};

/// `join_spec`, widened to `workers` instances.
fn widened_join_spec(name: &str, base: &str, overlay: &str, workers: usize) -> StageSpec {
    let mut join = engine::join_spec(name, base, overlay);
    join.workers = Some(workers);
    join
}

fn warned_nodes(warnings: &[RunWarning]) -> Vec<NodeId> {
    warnings
        .iter()
        .map(|warning| match warning {
            RunWarning::NeverChunked { node } => *node,
        })
        .collect()
}

#[test]
fn a_widened_join_whose_authority_never_chunked_is_warned_about() {
    let specs = vec![
        engine::source_spec("base_input", &[0, 0, 0, 0], 0),
        engine::source_spec("overlay_input", &[0, 1, 2, 3], 1),
        widened_join_spec("join", "base_input", "overlay_input", 4),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report = engine::run_with_watchdog(
        run_engine,
        "a_widened_join_whose_authority_never_chunked_is_warned_about",
    );

    assert_eq!(warned_nodes(&report.warnings), vec![NodeId::new(2)]);
}

#[test]
fn a_widened_join_whose_authority_did_chunk_is_not_warned_about() {
    let specs = vec![
        engine::source_spec("base_input", &[0, 1, 2, 3], 0),
        engine::source_spec("overlay_input", &[0, 1, 2, 3], 1),
        widened_join_spec("join", "base_input", "overlay_input", 4),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report = engine::run_with_watchdog(
        run_engine,
        "a_widened_join_whose_authority_did_chunk_is_not_warned_about",
    );

    assert!(report.warnings.is_empty());
}

/// The ordinary shape of a mask stream nobody split sitting beside a frame stream that was: the
/// authority carries every chunk id, the other input never carries more than zero. Nothing here
/// is a mistake, so this must stay quiet even though one input never chunked.
#[test]
fn an_authority_that_chunked_while_another_input_never_did_is_not_warned_about() {
    let specs = vec![
        engine::source_spec("base_input", &[0, 1, 2, 3], 0),
        engine::source_spec("overlay_input", &[0, 0, 0, 0], 1),
        widened_join_spec("join", "base_input", "overlay_input", 4),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report = engine::run_with_watchdog(
        run_engine,
        "an_authority_that_chunked_while_another_input_never_did_is_not_warned_about",
    );

    assert!(report.warnings.is_empty());
}

#[test]
fn a_pipeline_with_no_chunk_ids_at_all_is_not_warned_about() {
    let specs = vec![
        engine::source_spec("base_input", &[0, 0, 0, 0], 0),
        engine::source_spec("overlay_input", &[0, 0, 0, 0], 1),
        widened_join_spec("join", "base_input", "overlay_input", 4),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report = engine::run_with_watchdog(
        run_engine,
        "a_pipeline_with_no_chunk_ids_at_all_is_not_warned_about",
    );

    assert!(report.warnings.is_empty());
}

#[test]
fn a_region_running_one_worker_wide_is_not_warned_about() {
    let specs = vec![
        engine::source_spec("base_input", &[0, 0, 0, 0], 0),
        engine::source_spec("overlay_input", &[0, 1, 2, 3], 1),
        engine::join_spec("join", "base_input", "overlay_input"),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report =
        engine::run_with_watchdog(run_engine, "a_region_running_one_worker_wide_is_not_warned_about");

    assert!(report.warnings.is_empty());
}

/// The join's authority frame is what `test/join` passes through unchanged, so a `Hoarding` sink
/// downstream holds that same frame forever and starves its source of room for a third one. The
/// overlay frame is dropped by the join on every step, so only the authority side ever stalls.
///
/// By the time two bundles are held, bundle one has already carried the overlay's first nonzero
/// chunk through the router, so `other_chunked` is already true before the cancel lands. Nothing
/// here waits on a timer or a busy poll, only a channel receive, so the timing is exact.
#[test]
fn a_cancelled_run_is_not_warned_about() {
    let (hoarded_sender, hoarded_receiver) = mpsc::channel();
    *engine::HOARDED_EVERY.lock().expect("not poisoned") = Some(hoarded_sender);

    let specs = vec![
        engine::source_spec("base_input", &[0, 0, 0, 0], 0),
        engine::source_spec("overlay_input", &[0, 1, 2, 3], 1),
        widened_join_spec("join", "base_input", "overlay_input", 4),
        engine::hoarding_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(
        &specs,
        Settings {
            frame_budget: Some(4 * engine::format().total_bytes()),
            ..Settings::default()
        },
    );

    let result =
        engine::run_driven_with_watchdog(run_engine, "a_cancelled_run_is_not_warned_about", |cancel| {
            for _ in 0..2 {
                hoarded_receiver
                    .recv_timeout(engine::WATCHDOG)
                    .expect("the sink should hold the first two bundles before the budget stalls the rest");
            }

            cancel.stop();
        });

    let report = result.expect("a cancelled run is not a failed one");

    assert!(report.cancelled);
    assert!(report.warnings.is_empty());
}

#[test]
fn a_run_with_nothing_wrong_reports_no_warnings() {
    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        engine::source_spec("input", &chunks, 0),
        engine::filter_spec("filter", "input", Some(4)),
        engine::sink_spec("out", "filter"),
    ];
    let run_engine = engine::build_engine(&specs, Settings::default());

    let report = engine::run_with_watchdog(run_engine, "a_run_with_nothing_wrong_reports_no_warnings");

    assert!(report.warnings.is_empty());
}
