use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use av_toolbox_gpu::testing_context;
use av_toolbox_graph::Routing;

use crate::engine::tests::fixtures::gated::SLOW_GATES;
use crate::engine::tests::fixtures::sources::SOURCE_DONE;
use crate::engine::tests::harness::{
    assert_every_thread_exited,
    authority_of,
    build_engine,
    clear_recorded,
    engine_from,
    instantiate,
    planned_graph,
    recorded,
    recorded_indices,
    routing_of,
    run_result_with_watchdog,
    run_with_watchdog,
};
use crate::engine::tests::specs::{
    filter_spec,
    join_spec,
    offset_source_spec,
    referring_spec,
    sink_spec,
    slow_spec,
    source_spec,
    widened_source_spec,
};
use crate::{Engine, ExchangeError, RunError, Settings};

#[test]
fn a_widened_source_is_rejected() {
    let specs = vec![widened_source_spec("input", &[], 0, 4), sink_spec("out", "input")];
    let (graph, solved) = planned_graph(&specs);

    let result = Engine::build_with(
        &graph,
        &solved,
        Settings::default(),
        |region, worker| instantiate(&graph, region, worker),
        testing_context,
    );

    let Err(diagnostics) = result else {
        panic!("a source asked to run wider than one worker should have been refused");
    };
    // "workers" rather than the stage's own name "input", which would also match a message
    // about literally anything else that happened to mention the stage.
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("workers"))
    );
}

#[test]
fn a_region_with_more_than_one_exit_is_rejected() {
    // `filter1` and `filter2` chain off the source at width one, so they join its region. Each
    // is then read by its own widened filter, forcing both to be exits of that one region, which
    // this engine does not know how to give one aggregate each.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter1", "input", None),
        filter_spec("filter2", "filter1", None),
        filter_spec("filter3", "filter1", Some(4)),
        filter_spec("filter4", "filter2", Some(4)),
        sink_spec("sink3", "filter3"),
        sink_spec("sink4", "filter4"),
    ];
    let (graph, solved) = planned_graph(&specs);

    let result = Engine::build_with(
        &graph,
        &solved,
        Settings::default(),
        |region, worker| instantiate(&graph, region, worker),
        testing_context,
    );

    let Err(diagnostics) = result else {
        panic!("a region with two exits should have been refused");
    };
    // "exits" rather than just `has_errors()`, which would also pass if the build failed for
    // any unrelated reason.
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("exits"))
    );
}

/// A source, a widened branch off it, and a two input join reading both directly.
///
/// `join`'s two inputs sit in different regions (the source's own, and the branch's widened
/// one), so its disperse has two bound slots fed by two independent upstream channels. Before
/// the fix, offering each message the instant it arrived meant a fast slot receiving a second
/// item before its sibling received a first tripped `Disperse`'s own one-pending-item-per-slot
/// limit: `SlotRanAhead` on a single thread, or the same panic raised nondeterministically by
/// two threads racing into one `Disperse` behind a mutex before that. Either way it cascaded into
/// `Engine::run`'s own join panicking on the crashed thread.
///
/// The fix keeps its own queue per bound slot and only offers once every slot has something
/// waiting, so a fast slot queues rather than panics. This runs with the plan's own default
/// budget, `Settings::default()`, deliberately: it is what proves the fix does not
/// depend on a hand-picked size pinning the two paths together, the way the first attempt at
/// this test did.
#[test]
fn a_two_input_join_reads_both_producers_without_racing_the_disperse() {
    clear_recorded();

    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("branch", "input", Some(4)),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(
        engine,
        "a_two_input_join_reads_both_producers_without_racing_the_disperse",
    );

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..4).collect::<Vec<u64>>());
    assert_eq!(report.frames, 4);
}

/// The same shape, but `branch` is deliberately held back rather than merely slower by one
/// thread hop, so the fast slot genuinely accumulates several frames rather than at most one.
///
/// `branch`'s first worker blocks on [SLOW_GATE] until the source has finished producing and
/// forwarding every frame down both paths (signalled through [SOURCE_DONE], set the moment the
/// source discovers it has nothing left, which is always after the last real frame was already
/// forwarded). By the time the gate is released, every frame the source ever produced is
/// already queued against `join`'s direct slot, and none has reached its other slot at all.
/// Releasing the gate then lets `branch` drain its own backlog and catch up. Nothing here is
/// timed: both the hold and the release are channel sends, never a delay.
#[test]
fn a_slow_producer_still_joins_correctly_once_a_fast_ones_backlog_has_built_up() {
    clear_recorded();

    let (source_done_sender, source_done_receiver) = mpsc::channel();
    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    let (gate_sender, gate_receiver) = mpsc::channel();
    *SLOW_GATES[0].lock().expect("not poisoned") = Some(gate_receiver);

    let chunks: Vec<u64> = (0..6).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        slow_spec("branch", "input", 2),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let (run_result_sender, run_result_receiver) = mpsc::channel();
    thread::spawn(move || {
        run_result_sender.send(engine.run()).ok();
    });

    source_done_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("the source should finish producing before the gate is ever released");

    gate_sender.send(()).ok();

    let report = run_result_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("a_slow_producer_still_joins_correctly_once_a_fast_ones_backlog_has_built_up: the engine run did not finish within the watchdog timeout")
        .expect("the engine run failed");

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..6).collect::<Vec<u64>>());
    assert_eq!(report.frames, 6);
}

/// Three sources whose frames start at index one hundred, and two joins chained off them.
///
/// Each join heads a region of one worker with two inputs, which the plan gives a disperse to
/// pair them and no aggregate. Nothing in either region renumbers, so the indices the sources
/// chose reach the sink untouched. Building an aggregate for the first join would renumber its
/// output from zero, and the second join would then be pairing frame nought with frame one
/// hundred, which is not a pairing at all.
#[test]
fn a_width_one_region_with_a_two_input_head_does_not_renumber() {
    let property = "a_width_one_region_with_a_two_input_head_does_not_renumber";
    clear_recorded();

    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        offset_source_spec("base_input", &chunks, 0, 100),
        offset_source_spec("overlay_input", &chunks, 1, 100),
        join_spec("first_join", "base_input", "overlay_input"),
        offset_source_spec("third_input", &chunks, 2, 100),
        join_spec("second_join", "first_join", "third_input"),
        sink_spec("out", "second_join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    let mut indices = recorded_indices();
    indices.sort_unstable();

    assert_eq!(indices, vec![100, 101, 102]);
    assert_eq!(report.frames, 3);
}

/// A join whose two branches disagree about which frame they are carrying.
///
/// `branch` is a widened region, so its aggregate renumbers what it emits from zero, while the
/// source's own output reaches the join still numbered from one hundred. Pairing them is what
/// `OutOfStep` names, and it used to surface as a panic about a slot running ahead instead.
#[test]
fn a_join_whose_inputs_carry_different_frames_fails_the_run() {
    let property = "a_join_whose_inputs_carry_different_frames_fails_the_run";
    let specs = vec![
        offset_source_spec("input", &[0, 1, 2], 0, 100),
        filter_spec("branch", "input", Some(2)),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("a join given two different frames should have failed the run");
    };
    assert!(
        matches!(
            error,
            RunError::Exchange {
                source: ExchangeError::OutOfStep { .. }
            }
        ),
        "the run reported {error}",
    );
}

/// A frame routed head declaring its chunk authority on an optional input nothing bound.
///
/// Frame routing reads no chunk id, so an authority the disperse never looks at must not stand
/// in the way of the region being built.
#[test]
fn a_frame_routed_head_may_declare_an_authority_it_never_reads() {
    let property = "a_frame_routed_head_may_declare_an_authority_it_never_reads";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        referring_spec("referring", "input", 4, false),
        sink_spec("out", "referring"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "referring"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    assert_eq!(recorded(), (0..4).collect::<Vec<u64>>());
    assert_eq!(report.frames, 4);
    assert_every_thread_exited(property);
}

/// The same head declared per chunk, so the region routes by chunk and the build check a frame
/// routed region skips is the one that runs.
///
/// Binding resolves an unbound authority onto the first bound slot rather than leaving it
/// dangling, so this shape is accepted on the chunk path too.
#[test]
fn a_chunk_routed_head_resolves_an_unbound_authority_onto_a_bound_slot() {
    let property = "a_chunk_routed_head_resolves_an_unbound_authority_onto_a_bound_slot";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 1, 2, 3], 0),
        referring_spec("referring", "input", 4, true),
        sink_spec("out", "referring"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "referring"), Routing::ByChunk);
    assert_eq!(authority_of(&graph, "referring"), Some(0));

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    assert_eq!(recorded(), (0..4).collect::<Vec<u64>>());
    assert_eq!(report.frames, 4);
    assert_every_thread_exited(property);
}
