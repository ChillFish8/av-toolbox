use av_toolbox_graph::Routing;

use crate::Settings;
use crate::engine::tests::fixtures::filters::FLUSHED_TIMESTAMP;
use crate::engine::tests::harness::{
    assert_every_thread_exited,
    build_engine,
    clear_recorded,
    engine_from,
    planned_graph,
    planned_width,
    recorded,
    routing_of,
    run_with_watchdog,
};
use crate::engine::tests::specs::{flushing_spec, sink_spec, source_spec};

/// Eight frames through a frame mode stage four workers wide, where every worker's instance emits
/// one frame of its own as its stream ends.
///
/// A frame routed worker starts once and ends once, so its flush arrives after the last unit it
/// was assigned. The output carries one flushed frame per worker on top of the eight that went in.
#[test]
fn a_frame_routed_region_delivers_every_workers_flush() {
    let property = "a_frame_routed_region_delivers_every_workers_flush";
    clear_recorded();

    let width = 4;
    let specs = vec![
        source_spec("input", &[0; 8], 0),
        flushing_spec("flush", "input", width, false),
        sink_spec("out", "flush"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "flush"), Routing::ByFrame);
    assert_eq!(planned_width(&graph, "flush"), width);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    let mut expected: Vec<u64> = (0..8).collect();
    expected.extend(vec![FLUSHED_TIMESTAMP; width]);

    assert_eq!(recorded(), expected);
    assert_eq!(report.frames, expected.len() as u64);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// Two frames and eight workers, so six of them are never given anything to run.
///
/// A worker that got no bundle was still told its stream started, so it still has a flush to
/// deliver and its frame still belongs in the output.
#[test]
fn a_frame_routed_worker_given_no_work_still_delivers_its_flush() {
    let property = "a_frame_routed_worker_given_no_work_still_delivers_its_flush";
    clear_recorded();

    let width = 8;
    let specs = vec![
        source_spec("input", &[0; 2], 0),
        flushing_spec("flush", "input", width, false),
        sink_spec("out", "flush"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "flush"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    let mut expected: Vec<u64> = (0..2).collect();
    expected.extend(vec![FLUSHED_TIMESTAMP; width]);

    assert_eq!(recorded(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// The same fixture at width one, which runs as a single region with no exchange at all.
///
/// This is the sequential reference the widened runs are worth comparing against, so it has to be
/// delivering its own flush for that comparison to mean anything.
#[test]
fn a_width_one_region_delivers_its_flush() {
    let property = "a_width_one_region_delivers_its_flush";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        flushing_spec("flush", "input", 1, false),
        sink_spec("out", "flush"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    let mut expected: Vec<u64> = (0..8).collect();
    expected.push(FLUSHED_TIMESTAMP);

    assert_eq!(recorded(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// A chunk routed worker restarts its region at every chunk boundary, so it flushes once per
/// chunk and each flushed frame lands inside the chunk that produced it.
#[test]
fn a_chunk_routed_region_delivers_every_chunks_flush() {
    let property = "a_chunk_routed_region_delivers_every_chunks_flush";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 1, 1, 2, 2, 3, 3], 0),
        flushing_spec("flush", "input", 4, true),
        sink_spec("out", "flush"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "flush"), Routing::ByChunk);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    let flushed = FLUSHED_TIMESTAMP;
    let expected = vec![0, 1, flushed, 2, 3, flushed, 4, 5, flushed, 6, 7, flushed];

    assert_eq!(recorded(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

#[test]
fn a_frame_mode_stage_on_one_worker_is_not_restarted_per_chunk() {
    let property = "a_frame_mode_stage_on_one_worker_is_not_restarted_per_chunk";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 1, 1, 2, 2], 0),
        flushing_spec("flush", "input", 1, false),
        sink_spec("out", "flush"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    let mut expected: Vec<u64> = (0..6).collect();
    expected.push(FLUSHED_TIMESTAMP);

    assert_eq!(recorded(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}
