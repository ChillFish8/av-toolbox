use std::sync::atomic::Ordering;
use std::sync::mpsc;

use crate::Settings;
use crate::engine::tests as engine;

/// One frame's worth of bytes, the unit every budget in this file is sized against.
fn one_frame_bytes() -> usize {
    engine::format().total_bytes()
}

/// One long chunk pins a worker while the others run ahead, and memory stays near the bound.
///
/// `branch` is widened to four workers. Its worker zero is held on its very first chunk, standing
/// in for one chunk that takes far longer than the rest. Because the region has to hand its
/// output back in input order, nothing behind that held chunk can be reported downstream and
/// credited back until it finishes, so the source saturates the budget and then genuinely parks,
/// which [engine::SOURCE_WAITING] confirms. That parking is the proof: outstanding bytes plateau
/// at the budget rather than growing with the thirty-two chunks queued up behind the hold.
/// Releasing the hold then drains everything and the run finishes normally.
#[test]
fn chunk_skew_costs_parallelism_rather_than_memory() {
    let property = "chunk_skew_costs_parallelism_rather_than_memory";
    engine::SOURCE_BUDGET_PEAK.store(0, Ordering::SeqCst);
    engine::clear_recorded();

    let (parked_sender, parked_receiver) = mpsc::channel();
    let (gate_sender, gate_receiver) = mpsc::channel();
    let (waiting_sender, waiting_receiver) = mpsc::channel();

    *engine::SLOW_PARKED[0].lock().expect("not poisoned") = Some(parked_sender);
    *engine::SLOW_GATES[0].lock().expect("not poisoned") = Some(gate_receiver);
    *engine::SOURCE_WAITING.lock().expect("not poisoned") = Some(waiting_sender);

    let frame_bytes = one_frame_bytes();
    let budget = 4 * frame_bytes;
    let chunks: Vec<u64> = (0..32).collect();
    let specs = vec![
        engine::source_spec("input", &chunks, 0),
        engine::slow_spec("branch", "input", 4),
        engine::sink_spec("out", "branch"),
    ];
    let run_engine = engine::build_engine(
        &specs,
        Settings {
            frame_budget: Some(budget),
            ..Settings::default()
        },
    );

    let report = engine::run_driven_with_watchdog(run_engine, property, |_cancel| {
        parked_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the held worker should take its chunk and park inside it");

        waiting_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the source should saturate the budget and park behind the held chunk");

        gate_sender.send(()).ok();
    })
    .expect("a run with a held worker should still finish once it is released");

    let peak = engine::SOURCE_BUDGET_PEAK.load(Ordering::SeqCst);

    assert!(
        peak <= budget,
        "{property}: outstanding bytes reached {peak}, past the {budget} byte budget",
    );
    assert!(
        peak > frame_bytes,
        "{property}: outstanding bytes only reached {peak}, so the budget was never exercised",
    );
    assert_eq!(report.frames, 32, "{property}: not every chunk reached the sink");
    engine::assert_every_thread_exited(property);
}

/// The slow consumer is on the host too, so the source parks in its own allocation rather than
/// behind an upload.
#[test]
fn a_host_source_parks_on_its_budget_while_a_slow_consumer_holds_its_frames() {
    let property = "a_host_source_parks_on_its_budget_while_a_slow_consumer_holds_its_frames";
    engine::SOURCE_BUDGET_PEAK.store(0, Ordering::SeqCst);
    engine::clear_recorded();

    let (parked_sender, parked_receiver) = mpsc::channel();
    let (gate_sender, gate_receiver) = mpsc::channel();
    let (waiting_sender, waiting_receiver) = mpsc::channel();

    *engine::SLOW_PARKED[0].lock().expect("not poisoned") = Some(parked_sender);
    *engine::SLOW_GATES[0].lock().expect("not poisoned") = Some(gate_receiver);
    *engine::SOURCE_WAITING.lock().expect("not poisoned") = Some(waiting_sender);

    let frame_bytes = one_frame_bytes();
    let budget = 4 * frame_bytes;
    let chunks: Vec<u64> = (0..32).collect();
    let specs = vec![
        engine::host_source_spec("input", &chunks, 0),
        engine::host_slow_spec("branch", "input", 4),
        engine::host_sink_spec("out", "branch"),
    ];
    let run_engine = engine::build_engine(
        &specs,
        Settings {
            frame_budget: Some(budget),
            ..Settings::default()
        },
    );

    let report = engine::run_driven_with_watchdog(run_engine, property, |_cancel| {
        parked_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the held worker should take its chunk and park inside it");

        waiting_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the host source should saturate the budget and park behind the held chunk");

        gate_sender.send(()).ok();
    })
    .expect("a run with a held worker should still finish once it is released");

    let peak = engine::SOURCE_BUDGET_PEAK.load(Ordering::SeqCst);

    assert!(
        peak <= budget,
        "{property}: outstanding bytes reached {peak}, past the {budget} byte budget",
    );
    assert!(
        peak > frame_bytes,
        "{property}: outstanding bytes only reached {peak}, so the budget was never exercised",
    );
    assert_eq!(report.frames, 32, "{property}: not every chunk reached the sink");
    engine::assert_every_thread_exited(property);
}

/// Two sources, a budget one alone could exhaust, and a stage joining them.
///
/// The whole budget is one frame's worth, split into half a frame's share each. That is small
/// enough that a single source, left to run alone against the entire budget, could not even hold
/// its own second frame outstanding, let alone all six of its chunks. If the two sources instead
/// drew on one shared pool, whichever ran first would fill it before the other ever got a frame,
/// and the join could never see a pair from both sides. The run finishing at all is only possible
/// because each source's memory is its own, fixed share, not something the other can exhaust.
#[test]
fn two_sources_feeding_one_join_both_make_progress() {
    let property = "two_sources_feeding_one_join_both_make_progress";
    engine::clear_recorded();

    let frame_bytes = one_frame_bytes();
    let chunks: Vec<u64> = (0..6).collect();
    let specs = vec![
        engine::source_spec("base_input", &chunks, 0),
        engine::source_spec("overlay_input", &chunks, 1),
        engine::join_spec("join", "base_input", "overlay_input"),
        engine::sink_spec("out", "join"),
    ];
    let run_engine = engine::build_engine(
        &specs,
        Settings {
            frame_budget: Some(frame_bytes),
            ..Settings::default()
        },
    );

    let report = engine::run_with_watchdog(run_engine, property);

    let mut seen = engine::recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..6).collect::<Vec<u64>>());
    assert_eq!(report.frames, 6);
    engine::assert_every_thread_exited(property);
}

/// A budget of one frame and a sink that holds each frame until the test releases it.
///
/// The source can only ever have one frame outstanding, so producing its second chunk has to
/// wait for the first to come back. [engine::SOURCE_WAITING] fires from inside that wait, proving
/// the source is genuinely parked on its own budget rather than merely slow, before the test lets
/// the held frame go. Releasing each frame in turn is what lets the source keep going, and every
/// chunk reaching the sink is what proves it did.
#[test]
fn a_source_parked_on_its_share_resumes_when_frames_return() {
    let property = "a_source_parked_on_its_share_resumes_when_frames_return";

    let (waiting_sender, waiting_receiver) = mpsc::channel();
    let (got_sender, got_receiver) = mpsc::channel();
    let (release_sender, release_receiver) = mpsc::channel();

    *engine::SOURCE_WAITING.lock().expect("not poisoned") = Some(waiting_sender);
    *engine::GATED_GOT.lock().expect("not poisoned") = Some(got_sender);
    *engine::GATED_RELEASE.lock().expect("not poisoned") = Some(release_receiver);

    // Widened so the sink lands in a region of its own, on its own thread, separate from the
    // source's. Left at the source's own width, the sink's blocking `consume` would block the
    // very thread that has to call the source's `next` again, deadlocking the test rather than
    // the source.
    let frame_bytes = one_frame_bytes();
    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        engine::source_spec("input", &chunks, 0),
        engine::filter_spec("filter", "input", Some(2)),
        engine::gated_spec("out", "filter"),
    ];
    let run_engine = engine::build_engine(
        &specs,
        Settings {
            frame_budget: Some(frame_bytes),
            ..Settings::default()
        },
    );

    let report = engine::run_driven_with_watchdog(run_engine, property, |_cancel| {
        got_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the sink should take the first frame");
        waiting_receiver
            .recv_timeout(engine::WATCHDOG)
            .expect("the source should now be parked on its own budget");
        release_sender.send(()).ok();

        for _ in 0..2 {
            got_receiver
                .recv_timeout(engine::WATCHDOG)
                .expect("releasing a frame should let the source resume and reach the sink");
            release_sender.send(()).ok();
        }
    })
    .expect("a source that resumes once its budget clears should finish normally");

    assert_eq!(report.frames, 3, "{property}: not every chunk reached the sink");
    engine::assert_every_thread_exited(property);
}
