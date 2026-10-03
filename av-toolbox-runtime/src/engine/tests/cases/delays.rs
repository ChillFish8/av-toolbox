use std::time::Duration;

use av_toolbox_gpu::testing_context;
use av_toolbox_graph::{DiagnosticCode, Routing, StageSpec};

use crate::engine::tests::harness::{
    assert_every_thread_exited,
    build_engine,
    clear_matched,
    clear_recorded,
    declared_held,
    engine_from,
    instantiate,
    matched,
    node_id,
    planned_graph,
    planned_width,
    recorded,
    recorded_indices,
    routing_of,
    run_result_with_watchdog,
    run_with_watchdog,
    share_a_region,
    solve_result,
};
use crate::engine::tests::specs::{
    delay_spec,
    double_spec,
    filter_spec,
    join_spec,
    matching_spec,
    rated_source_spec,
    sink_spec,
    source_spec,
    undeclared_delay_spec,
};
use crate::{Engine, RunError, Settings};

const NINE_FRAMES: [u64; 9] = [0, 0, 0, 1, 1, 1, 1, 2, 2];

/// Runs `specs` and asserts every frame came out once, in order, with its own index.
fn assert_whole_stream(property: &str, specs: &[StageSpec], frames: u64) {
    let engine = build_engine(specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    let expected: Vec<u64> = (0..frames).collect();
    assert_eq!(recorded(), expected);
    assert_eq!(recorded_indices(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}

/// The error a failed run came from, whichever chunk it was reported against.
fn unwrapped(error: RunError) -> RunError {
    match error {
        RunError::InChunk { source, .. } => *source,
        other => other,
    }
}

/// The source's own region holds the source, the delay and the sink, so the delay is restarted inside the
/// region rather than by a worker.
#[test]
fn a_delay_in_a_source_region_flushes_every_chunk() {
    let property = "a_delay_in_a_source_region_flushes_every_chunk";
    clear_recorded();

    let specs = vec![
        source_spec("input", &NINE_FRAMES, 0),
        delay_spec("delay", "input", &[2], 1, false),
        sink_spec("out", "delay"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert!(share_a_region(&graph, "input", "delay"));

    assert_whole_stream(property, &specs, 9);
}

/// A width two stage in front leaves the delay heading a width one region of its own, fed through a channel.
#[test]
fn a_delay_in_a_passthrough_region_flushes_every_chunk() {
    let property = "a_delay_in_a_passthrough_region_flushes_every_chunk";
    clear_recorded();

    let specs = vec![
        source_spec("input", &NINE_FRAMES, 0),
        filter_spec("pass", "input", Some(2)),
        delay_spec("delay", "pass", &[2], 1, false),
        sink_spec("out", "delay"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "input", "delay"));
    assert!(!share_a_region(&graph, "pass", "delay"));
    assert_eq!(planned_width(&graph, "delay"), 1);

    assert_whole_stream(property, &specs, 9);
}

#[test]
fn a_delay_in_a_chunk_routed_region_flushes_every_chunk() {
    let property = "a_delay_in_a_chunk_routed_region_flushes_every_chunk";
    clear_recorded();

    let chunks = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3];
    let specs = vec![
        source_spec("input", &chunks, 0),
        delay_spec("delay", "input", &[2], 4, false),
        sink_spec("out", "delay"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "delay"), Routing::ByChunk);

    assert_whole_stream(property, &specs, 12);
}

/// Every stage is width one, so the join is fused with the delay and pairs its lagging feed with the direct
/// one by index.
#[test]
fn a_lagging_branch_and_a_direct_one_join_on_one_worker() {
    let property = "a_lagging_branch_and_a_direct_one_join_on_one_worker";
    clear_recorded();

    let specs = vec![
        source_spec("input", &NINE_FRAMES, 0),
        delay_spec("delay", "input", &[2], 1, false),
        filter_spec("pass", "input", None),
        join_spec("join", "delay", "pass"),
        sink_spec("out", "join"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert!(share_a_region(&graph, "delay", "join"));
    assert!(share_a_region(&graph, "pass", "join"));

    assert_whole_stream(property, &specs, 9);
}

/// The delay is two wide, so the matching stage reads it through a disperse beside the direct branch.
/// It records each pair it was given, so a wrongly paired frame shows.
#[test]
fn a_lagging_branch_joins_a_direct_one_across_regions() {
    let property = "a_lagging_branch_joins_a_direct_one_across_regions";
    clear_recorded();
    clear_matched();

    let specs = vec![
        source_spec("input", &NINE_FRAMES, 0),
        delay_spec("delay", "input", &[2], 2, false),
        filter_spec("pass", "input", None),
        matching_spec("match", "delay", "pass", 1),
        sink_spec("out", "match"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "delay", "match"));

    assert_whole_stream(property, &specs, 9);

    let expected: Vec<(u64, u64)> = (0..9).map(|second| (second, second)).collect();
    assert_eq!(matched(), expected);
}

/// Lag four over chunks of one, two and exactly four frames, so each comes out whole at its flush.
#[test]
fn a_chunk_shorter_than_its_lag_comes_out_at_its_flush() {
    let property = "a_chunk_shorter_than_its_lag_comes_out_at_its_flush";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 1, 1, 2, 2, 2, 2], 0),
        delay_spec("delay", "input", &[4], 1, false),
        sink_spec("out", "delay"),
    ];

    assert_whole_stream(property, &specs, 7);
}

#[test]
fn a_single_frame_chunk_comes_out_whole_at_its_flush() {
    let property = "a_single_frame_chunk_comes_out_whole_at_its_flush";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 1, 2, 2, 3], 0),
        delay_spec("delay", "input", &[2], 4, false),
        sink_spec("out", "delay"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "delay"), Routing::ByChunk);

    assert_whole_stream(property, &specs, 5);
}

#[test]
fn a_chunk_of_exactly_its_lag_comes_out_whole_at_its_flush() {
    let property = "a_chunk_of_exactly_its_lag_comes_out_whole_at_its_flush";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0, 1, 1, 1, 2, 2, 2], 0),
        delay_spec("delay", "input", &[3], 4, false),
        sink_spec("out", "delay"),
    ];

    assert_whole_stream(property, &specs, 9);
}

#[test]
fn a_lag_that_changes_between_chunks_keeps_every_frame() {
    let property = "a_lag_that_changes_between_chunks_keeps_every_frame";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 1, 1, 1, 1, 2, 2, 2], 0),
        delay_spec("delay", "input", &[0, 3, 1], 1, false),
        sink_spec("out", "delay"),
    ];

    assert_whole_stream(property, &specs, 9);
}

#[test]
fn a_delay_over_an_unchunked_stream_flushes_once_at_the_end() {
    let property = "a_delay_over_an_unchunked_stream_flushes_once_at_the_end";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        delay_spec("delay", "input", &[3], 1, false),
        sink_spec("out", "delay"),
    ];

    assert_whole_stream(property, &specs, 8);
}

#[test]
fn a_delay_of_nothing_matches_a_passthrough() {
    let property = "a_delay_of_nothing_matches_a_passthrough";
    clear_recorded();

    let specs = vec![
        source_spec("input", &NINE_FRAMES, 0),
        delay_spec("delay", "input", &[0], 1, false),
        sink_spec("out", "delay"),
    ];

    assert_whole_stream(property, &specs, 9);
}

#[test]
fn a_declared_hold_in_a_frame_routed_region_fails_to_build() {
    let specs = vec![
        source_spec("input", &[0; 8], 0),
        delay_spec("delay", "input", &[1], 4, true),
        sink_spec("out", "delay"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "delay"), Routing::ByFrame);

    let held = declared_held(&graph, &solved);

    let result = Engine::build_with(
        &graph,
        &solved,
        &held,
        Settings::default(),
        |region, worker| instantiate(&graph, region, worker),
        testing_context,
    );

    let Err(diagnostics) = result else {
        panic!("a frame routed stage declaring held frames should have been refused");
    };
    let refused = diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == DiagnosticCode::HeldInFrameRegion && diagnostic.message().contains("delay")
    });
    assert!(
        refused,
        "no HeldInFrameRegion diagnostic for delay in: {diagnostics}"
    );
}

#[test]
fn an_undeclared_hold_in_a_frame_routed_region_fails_the_run() {
    let property = "an_undeclared_hold_in_a_frame_routed_region_fails_the_run";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        undeclared_delay_spec("delay", "input", &[1], 4),
        sink_spec("out", "delay"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "delay"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("{property}: a frame routed stage returning no frames should have failed the run");
    };
    let delay = node_id(&graph, "delay");
    assert!(matches!(unwrapped(error), RunError::HeldInFrameRegion { node } if node == delay));
}

#[test]
fn a_doubling_filter_runs_frame_routed_and_keeps_every_frame() {
    let property = "a_doubling_filter_runs_frame_routed_and_keeps_every_frame";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        double_spec("double", "input", Some(4)),
        sink_spec("out", "double"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "double"), Routing::ByFrame);
    assert_eq!(planned_width(&graph, "double"), 4);

    assert_whole_stream(property, &specs, 16);
}

/// The doubled stream runs at twice the original's rate, which the solver refuses to join to it.
#[test]
fn a_doubled_stream_joined_with_the_original_is_refused_by_the_solver() {
    let specs = vec![
        rated_source_spec("input", &[0; 8], 0, [25, 1]),
        double_spec("double", "input", None),
        join_spec("join", "double", "input"),
        sink_spec("out", "join"),
    ];

    let Err(diagnostics) = solve_result(&specs) else {
        panic!("a join of two frame rates should have been refused");
    };

    let refused = diagnostics.iter().any(|diagnostic| {
        diagnostic.code() == DiagnosticCode::FrameRateMismatch && diagnostic.path().contains("join")
    });
    assert!(
        refused,
        "no FrameRateMismatch diagnostic for join in: {diagnostics}"
    );
}

/// From index zero the first eight pairs agree, so the doubled stream's other eight frames are left over.
#[test]
fn a_doubled_stream_joined_with_the_original_leaves_feeds_unpaired() {
    let property = "a_doubled_stream_joined_with_the_original_leaves_feeds_unpaired";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 8], 0),
        double_spec("double", "input", None),
        join_spec("join", "double", "input"),
        sink_spec("out", "join"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert!(share_a_region(&graph, "double", "join"));

    let engine = engine_from(&graph, &solved, Settings::default());
    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("{property}: a join left with unmatched frames should have failed the run");
    };
    let join = node_id(&graph, "join");
    let leftover = vec![8, 0];
    assert!(matches!(
        unwrapped(error),
        RunError::UnpairedFeeds { node, leftover: found } if node == join && found == leftover
    ));
}

/// With no budget given, the engine sizes the source's share from the delay's declared hold.
#[test]
fn a_lag_beyond_the_default_headroom_completes() {
    let property = "a_lag_beyond_the_default_headroom_completes";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0; 30], 0),
        delay_spec("delay", "input", &[20], 1, false),
        sink_spec("out", "delay"),
    ];
    let settings = Settings {
        frame_budget: None,
        stall_deadline: Some(Duration::from_secs(2)),
        frame_limit: None,
        warmup: 0,
        profiling: None,
    };
    let engine = build_engine(&specs, settings);

    let result = run_result_with_watchdog(engine, property);

    let report = match result {
        Ok(report) => report,
        Err(error) => panic!("{property}: the run failed: {error}"),
    };
    let expected: Vec<u64> = (0..30).collect();
    assert_eq!(recorded(), expected);
    assert_eq!(recorded_indices(), expected);
    assert_eq!(report.dropped, 0);
    assert_every_thread_exited(property);
}
