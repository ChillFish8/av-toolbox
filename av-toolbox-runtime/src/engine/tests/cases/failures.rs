use std::sync::atomic::Ordering;
use std::sync::mpsc;

use crate::engine::tests::fixtures::failing::{
    FAILING_CALLS,
    FAILING_ENDS,
    FAILING_GATES,
    FAILING_PARKED,
    FAILING_STARTS,
};
use crate::engine::tests::fixtures::filters::{PASS_ENDS, PASS_STARTS};
use crate::engine::tests::fixtures::gated::{SLOW_GATES, SLOW_PARKED};
use crate::engine::tests::fixtures::sinks::{HOARDED_ONE, RECORDING_ENDED};
use crate::engine::tests::fixtures::sources::{SOURCE_DONE, SOURCE_WAITING};
use crate::engine::tests::harness::{
    WATCHDOG,
    assert_every_thread_exited,
    build_engine,
    clear_recorded,
    engine_from,
    node_id,
    planned_graph,
    run_driven_with_watchdog,
    run_result_with_watchdog,
    run_with_watchdog,
};
use crate::engine::tests::specs::{
    failing_spec,
    failing_worker_spec,
    filter_spec,
    hoarding_spec,
    join_spec,
    panicking_spec,
    sink_spec,
    slow_spec,
    source_spec,
};
use crate::{RunError, Settings};

/// The failing filter is widened, so the stage that fails is one worker of four and the other
/// three are left waiting on work that is never coming.
#[test]
fn a_stage_failure_stops_the_run_and_names_its_node() {
    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 3, Some(4)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_stops_the_run_and_names_its_node");

    let Err(error) = result else {
        panic!("a stage that failed should have failed the run");
    };
    assert_eq!(error.node(), Some(failing), "the run reported {error}");
}

#[test]
fn a_stage_failure_joins_every_thread() {
    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 3, Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_joins_every_thread");

    assert!(result.is_err(), "a stage that failed should have failed the run");
    assert_every_thread_exited("a_stage_failure_joins_every_thread");
}

/// A budget of one byte and a sink that never lets go of a frame, so the source waits for room
/// that will never appear.
///
/// The test waits for the sink to take the first frame before it cancels, because a frame still
/// on its way somewhere is a frame that could yet be dropped, and dropping one credits the budget
/// and frees the source without any of this being involved. Once the sink is holding it, only
/// releasing the budget can get the source out of its wait.
///
/// The source signals before the allocation that finds the budget full. Nothing here depends on
/// the source having reached the wait by then, because releasing a budget is sticky rather than a
/// one-off nudge.
#[test]
fn a_run_cancelled_while_a_source_waits_on_the_budget_joins_every_thread() {
    let property = "a_run_cancelled_while_a_source_waits_on_the_budget_joins_every_thread";
    let (waiting_sender, waiting_receiver) = mpsc::channel();
    let (hoarded_sender, hoarded_receiver) = mpsc::channel();
    *SOURCE_WAITING.lock().expect("not poisoned") = Some(waiting_sender);
    *HOARDED_ONE.lock().expect("not poisoned") = Some(hoarded_sender);

    // Far more frames than this run will ever get through, so the source is still going when the
    // budget stops it.
    let chunks: Vec<u64> = (0..64).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(2)),
        hoarding_spec("out", "filter"),
    ];
    let engine = build_engine(
        &specs,
        Settings {
            frame_budget: Some(1),
            ..Settings::default()
        },
    );

    let result = run_driven_with_watchdog(engine, property, |cancel| {
        hoarded_receiver
            .recv_timeout(WATCHDOG)
            .expect("the sink should take the first frame and keep it");
        waiting_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should reach the frame that has to wait for room");

        cancel.stop();
    });

    let report = result.expect("a cancelled run is not a failed one");

    assert!(report.cancelled);
    assert_every_thread_exited(property);
}

#[test]
fn a_panicking_worker_becomes_an_error_rather_than_a_hang() {
    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        panicking_spec("filter", "input", 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, "a_panicking_worker_becomes_an_error_rather_than_a_hang");

    let Err(error) = result else {
        panic!("a stage that panicked should have failed the run");
    };
    assert!(
        matches!(error, RunError::Panicked { .. }),
        "the run reported {error}",
    );
}

#[test]
fn a_panicking_worker_still_lets_every_other_thread_be_joined() {
    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        panicking_spec("filter", "input", 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(
        engine,
        "a_panicking_worker_still_lets_every_other_thread_be_joined",
    );

    assert!(
        result.is_err(),
        "a stage that panicked should have failed the run"
    );
    assert_every_thread_exited("a_panicking_worker_still_lets_every_other_thread_be_joined");
}

/// Two stages that both fail, ordered against each other rather than raced.
///
/// `second` announces that it is parked and waits. Only then is `first` let go, so it fails while
/// `second` is already sitting inside its own call, and `second` is only released once the run has
/// stopped, which is something that only happens after the first failure has been recorded.
#[test]
fn the_first_error_is_the_one_reported() {
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();

    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_PARKED[1][0].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[1][0].lock().expect("not poisoned") = Some(second_gate_receiver);

    let specs = vec![
        source_spec("first_input", &[0, 0], 0),
        failing_spec("first", "first_input", 0, 1, None),
        sink_spec("first_out", "first"),
        source_spec("second_input", &[0, 0], 1),
        failing_spec("second", "second_input", 1, 1, None),
        sink_spec("second_out", "second"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let first = node_id(&graph, "first");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_driven_with_watchdog(engine, "the_first_error_is_the_one_reported", |cancel| {
        let (stopped_sender, stopped_receiver) = mpsc::channel();
        cancel.on_stop(move || {
            stopped_sender.send(()).ok();
        });

        parked_receiver
            .recv_timeout(WATCHDOG)
            .expect("the second stage should park before the first is let go");
        first_gate_sender.send(()).ok();

        stopped_receiver
            .recv_timeout(WATCHDOG)
            .expect("the first failure should have stopped the run");
        second_gate_sender.send(()).ok();
    });

    let Err(error) = result else {
        panic!("both stages failed, so the run should have failed");
    };

    assert_eq!(error.node(), Some(first), "the run reported {error}");
    assert_eq!(FAILING_CALLS[0].load(Ordering::SeqCst), 1);
    assert_eq!(
        FAILING_CALLS[1].load(Ordering::SeqCst),
        1,
        "the second stage never reached the failure it was released for",
    );
}

/// A worker whose aggregate has already gone still has to end what it started.
///
/// Worker zero fails while worker one is held inside its own first frame. That failure is what
/// makes the aggregate give up, and the sink being told its stream ended is how this test knows
/// it has, since an aggregate closes the stream downstream on its way out. Only then is worker
/// one let go, so the results it comes back with have nowhere to go, which is the path that used
/// to leave the loop early with a start still open.
#[test]
fn a_worker_whose_aggregate_has_gone_still_ends_what_it_started() {
    let property = "a_worker_whose_aggregate_has_gone_still_ends_what_it_started";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (sink_ended_sender, sink_ended_receiver) = mpsc::channel();

    *FAILING_PARKED[0][0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *FAILING_PARKED[0][1].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_GATES[0][1].lock().expect("not poisoned") = Some(second_gate_receiver);
    *RECORDING_ENDED.lock().expect("not poisoned") = Some(sink_ended_sender);

    let specs = vec![
        source_spec("input", &[0, 1], 0),
        failing_worker_spec("filter", "input", 0, 0, 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        first_gate_sender.send(()).ok();
        sink_ended_receiver
            .recv_timeout(WATCHDOG)
            .expect("the aggregate should give up on the failed worker and close the stream");

        second_gate_sender.send(()).ok();
    });

    assert!(
        result.is_err(),
        "the first worker failed, so the run should have failed"
    );

    let starts = FAILING_STARTS[1].load(Ordering::SeqCst);
    let ends = FAILING_ENDS[1].load(Ordering::SeqCst);

    assert_eq!(ends, 1, "{property}: the surviving worker never ended its chunk");
    assert_eq!(starts, ends, "{property}: {starts} starts against {ends} ends");
}

/// Chunk ids starting at five, so the one reported cannot be a default or a frame index.
#[test]
fn a_stage_failure_names_the_chunk_it_came_from() {
    let specs = vec![
        source_spec("input", &[5, 6, 7], 0),
        failing_worker_spec("filter", "input", 0, 0, 2),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_names_the_chunk_it_came_from");

    let Err(error) = result else {
        panic!("a stage that failed should have failed the run");
    };

    assert_eq!(error.node(), Some(failing), "the run reported {error}");
    assert_eq!(error.chunk(), Some(5), "the run reported {error}");
}

/// The inherited hang, as the catalogue describes it: a worker fails, its aggregate gives up on
/// it, and nothing will ever mark a worker free again while the disperse is waiting for one.
///
/// Both workers are held inside their own first frame and then fail the moment they are let go,
/// so neither ever closes the chunk it was given and nothing ever frees a worker. The only thing
/// that can get the disperse out of its wait is the stop the first failure raises. Six more
/// frames are produced behind the two the workers took, so by the time they are let go the
/// disperse has long since taken the next bundle and found nobody free for it.
#[test]
fn a_failure_while_the_disperse_waits_for_a_worker_still_ends_the_run() {
    let property = "a_failure_while_the_disperse_waits_for_a_worker_still_ends_the_run";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (source_done_sender, source_done_receiver) = mpsc::channel();

    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    *FAILING_PARKED[0][0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *FAILING_PARKED[0][1].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_GATES[0][1].lock().expect("not poisoned") = Some(second_gate_receiver);

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 1, Some(2)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        source_done_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should finish producing every frame");

        first_gate_sender.send(()).ok();
        second_gate_sender.send(()).ok();
    });

    let Err(error) = result else {
        panic!("both workers failed, so the run should have failed");
    };

    assert_eq!(error.node(), Some(failing), "the run reported {error}");
    assert_every_thread_exited(property);
}

/// Three chunks and two workers, both held inside their own first frame, so the third chunk's
/// bundle completes with nobody free to take it and the disperse is left waiting.
///
/// Cancelling is what gets the disperse out of that wait, and the bundle it was still holding is
/// what the report has to account for. The workers are released afterwards, because a stage
/// blocked inside a call of its own is the one place cancellation cannot reach.
#[test]
fn a_bundle_dropped_at_shutdown_is_reported_rather_than_discarded() {
    let property = "a_bundle_dropped_at_shutdown_is_reported_rather_than_discarded";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (source_done_sender, source_done_receiver) = mpsc::channel();

    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    *SLOW_PARKED[0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *SLOW_PARKED[1].lock().expect("not poisoned") = Some(parked_sender);
    *SLOW_GATES[0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *SLOW_GATES[1].lock().expect("not poisoned") = Some(second_gate_receiver);

    let specs = vec![
        source_spec("input", &[0, 1, 2], 0),
        slow_spec("branch", "input", 2),
        sink_spec("out", "branch"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        source_done_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should finish producing every frame");

        cancel.stop();
        first_gate_sender.send(()).ok();
        second_gate_sender.send(()).ok();
    });

    let report = result.expect("a cancelled run is not a failed one");

    assert!(report.cancelled);
    assert_eq!(
        report.dropped, 1,
        "the bundle held when the run stopped was not accounted for",
    );
    assert_eq!(report.frames, 2, "the two chunks already with a worker still ran");
    assert_every_thread_exited(property);
}

#[test]
fn a_stage_gets_no_start_without_a_matching_end() {
    PASS_STARTS.store(0, Ordering::SeqCst);
    PASS_ENDS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_stage_gets_no_start_without_a_matching_end");

    let starts = PASS_STARTS.load(Ordering::SeqCst);
    let ends = PASS_ENDS.load(Ordering::SeqCst);

    assert_eq!(starts, ends, "{starts} starts against {ends} ends");
}

/// Two sources of different lengths feeding one join, which can only pair as many as the shorter
/// one has.
///
/// The five frames left over never reach a worker at all. A run that answered `Ok` for that would
/// leave a caller checking only the result with no sign that anything went missing.
#[test]
fn a_run_that_lost_items_without_being_cancelled_fails() {
    let property = "a_run_that_lost_items_without_being_cancelled_fails";
    clear_recorded();

    let long: Vec<u64> = (0..8).collect();
    let short: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &long, 0),
        source_spec("overlay_input", &short, 1),
        join_spec("join", "base_input", "overlay_input"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("a run that lost five frames should not have reported success");
    };
    let RunError::DroppedItems { dropped } = error else {
        panic!("the run reported {error}");
    };

    assert_eq!(dropped, 5);
}
