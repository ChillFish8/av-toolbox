use std::sync::mpsc;

use av_toolbox_graph::Routing;

use crate::Settings;
use crate::engine::FRAME_UNITS_IN_FLIGHT;
use crate::engine::tests::{
    GATE_HOLDS,
    GATE_PROCESSED,
    GATE_REACHED,
    NOTHING_MORE_COMING,
    WATCHDOG,
    achieved_width,
    assert_every_thread_exited,
    clear_assigned,
    clear_matched,
    clear_recorded,
    engine_from,
    filter_spec,
    matched,
    matching_spec,
    planned_graph,
    planned_width,
    recorded,
    routing_of,
    run_driven_with_watchdog,
    run_with_watchdog,
    sink_spec,
    source_spec,
    tracked_spec,
};

/// The channels a test uses to hold some of a region's workers inside their own stage.
///
/// A held worker is never drained, so it is never marked free, so the dispatch policy cannot
/// choose it again. That is what turns an assertion about spreading from a hope that the
/// scheduler cooperated into something the placement rules force.
struct Held {
    /// One signal per held worker, sent as it arrives at its gate.
    arrivals: mpsc::Receiver<()>,
    /// One release per held worker, in worker order.
    releases: Vec<mpsc::Sender<()>>,
    /// One signal per frame any worker of the gated stage finished.
    processed: mpsc::Receiver<()>,
}

/// Holds the first `workers` workers of the gated stage inside their own first `process` call.
fn hold_workers(workers: usize) -> Held {
    let (arrived_sender, arrivals) = mpsc::channel();
    let (processed_sender, processed) = mpsc::channel();
    let mut releases = Vec::with_capacity(workers);

    for worker in 0..workers {
        let (release_sender, release_receiver) = mpsc::channel();

        *GATE_REACHED[worker].lock().expect("not poisoned") = Some(arrived_sender.clone());
        *GATE_HOLDS[worker].lock().expect("not poisoned") = Some(release_receiver);
        releases.push(release_sender);
    }

    *GATE_PROCESSED.lock().expect("not poisoned") = Some(processed_sender);

    Held {
        arrivals,
        releases,
        processed,
    }
}

impl Held {
    /// Waits for every held worker to be given a frame, then lets them all go.
    ///
    /// Answers how many arrived rather than asserting, so the caller can say what a shortfall
    /// means, and so a run that only ever reaches one worker costs one watchdog rather than one
    /// per worker that never came.
    fn release_once_all_arrived(&self) -> usize {
        let mut arrived = 0;

        while arrived < self.releases.len() && self.arrivals.recv_timeout(WATCHDOG).is_ok() {
            arrived += 1;
        }

        self.release();

        arrived
    }

    fn release(&self) {
        for release in &self.releases {
            release.send(()).ok();
        }
    }
}

/// Eight frames, every one of them in chunk zero, through a frame mode stage four workers wide,
/// with all four held inside their own stage at once.
///
/// Frame routing is the only thing that can spread a stream with no boundaries in it, so this is
/// the run that used to happen entirely on worker zero with nothing noticing. Holding every worker
/// is what makes that certain rather than likely: none of them can be drained, so none of them can
/// be chosen twice, and a fourth arrival can only be a fourth worker.
#[test]
fn a_frame_mode_stage_uses_every_worker_when_nothing_is_chunked() {
    let property = "a_frame_mode_stage_uses_every_worker_when_nothing_is_chunked";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(4);
    let chunks = vec![0; 8];
    let specs = vec![
        source_spec("input", &chunks, 0),
        tracked_spec("tracked", "input", 4, false),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "tracked"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });

    let report = result.expect("the run should finish once the held workers are released");
    let width = achieved_width();

    assert_eq!(
        arrived, 4,
        "{property}: only {arrived} of the four workers were ever given a frame",
    );
    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert!(
        width > 1,
        "{property}: all eight frames ran on {width} worker, so nothing was spread",
    );
    assert_every_thread_exited(property);
}

/// One chunk is one worker's work, so a width of one here is correct rather than a fault. It is
/// also why the run warns about it, and pinning it keeps the two modes from quietly merging.
#[test]
fn a_chunk_mode_stage_gets_one_worker_when_nothing_is_chunked() {
    let property = "a_chunk_mode_stage_gets_one_worker_when_nothing_is_chunked";
    clear_recorded();
    clear_assigned();

    let chunks = vec![0; 8];
    let specs = vec![
        source_spec("input", &chunks, 0),
        tracked_spec("tracked", "input", 4, true),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "tracked"), Routing::ByChunk);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);
    let width = achieved_width();

    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert_eq!(
        width, 1,
        "{property}: a chunk routed region spread a stream with one chunk in it across {width} workers",
    );
    assert_every_thread_exited(property);
}

/// The same chunk mode stage over four chunks, which is the stream it was always meant for.
///
/// A chunk routed worker holds one unit, so holding all four is what stops the disperse handing
/// the second chunk straight back to the worker that has just finished the first.
#[test]
fn a_chunk_mode_stage_uses_a_worker_per_chunk() {
    let property = "a_chunk_mode_stage_uses_a_worker_per_chunk";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(4);
    let chunks = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let specs = vec![
        source_spec("input", &chunks, 0),
        tracked_spec("tracked", "input", 4, true),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "tracked"), Routing::ByChunk);

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });

    let report = result.expect("the run should finish once the held workers are released");
    let width = achieved_width();

    assert_eq!(
        arrived, 4,
        "{property}: only {arrived} of the four chunks reached a worker of its own",
    );
    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert!(width > 1, "{property}: four chunks all ran on {width} worker");
    assert_every_thread_exited(property);
}

/// Two frames, both in chunk zero, through a frame mode stage two workers wide, with the first
/// worker held inside its own `process` call while the test watches for progress.
///
/// Chunk routing gives one chunk to one worker, so holding that worker holds the entire run and
/// nothing else can come through. Frame routing hands the second frame to the other worker, which
/// finishes it while the first is still held. This never asks which worker got what, only that one
/// of them got on with it, so ties going to the lowest index cannot make it flaky.
#[test]
fn a_held_frame_mode_worker_does_not_hold_up_the_others() {
    let property = "a_held_frame_mode_worker_does_not_hold_up_the_others";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(1);
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        tracked_spec("tracked", "input", 2, false),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "tracked"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        held.arrivals
            .recv_timeout(WATCHDOG)
            .expect("the first worker should reach its gate");

        let progressed = held.processed.recv_timeout(WATCHDOG);
        held.release();

        assert!(
            progressed.is_ok(),
            "{property}: nothing was processed while the first worker was held, so the whole \
             region was waiting on one worker",
        );
    });

    let report = result.expect("the run should finish once the held worker is released");
    let width = achieved_width();

    assert_eq!(recorded(), vec![0, 1]);
    assert_eq!(report.frames, 2);
    assert_eq!(
        width, 2,
        "{property}: two frames at width two reached {width} worker",
    );
    assert_every_thread_exited(property);
}

/// Two sources into a frame mode stage of two inputs, four workers wide.
///
/// Every bundle has to carry the two frames that belong together, whichever worker it lands on. A
/// bundle that crossed its slots would still produce a frame and still come back in order, so the
/// pairs the stage was handed are the only place the mistake shows. Holding all four workers is
/// what puts four different bundles on four different workers to be checked.
#[test]
fn a_frame_routed_region_keeps_its_two_inputs_paired_by_index() {
    let property = "a_frame_routed_region_keeps_its_two_inputs_paired_by_index";
    clear_recorded();
    clear_assigned();
    clear_matched();

    let held = hold_workers(4);
    let chunks = vec![0; 8];
    let specs = vec![
        source_spec("base_input", &chunks, 0),
        source_spec("overlay_input", &chunks, 1),
        matching_spec("matching", "base_input", "overlay_input", 4),
        sink_spec("out", "matching"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "matching"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });

    let report = result.expect("the run should finish once the held workers are released");
    let width = achieved_width();

    let mut pairs = matched();
    pairs.sort_unstable();
    let expected: Vec<(u64, u64)> = (0..8).map(|second| (second, second)).collect();

    assert_eq!(
        arrived, 4,
        "{property}: only {arrived} of the four workers were ever given a bundle",
    );
    assert_eq!(
        pairs, expected,
        "{property}: a bundle paired two different frames"
    );
    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert!(
        width > 1,
        "{property}: a frame routed region of two inputs ran {width} worker wide",
    );
    assert_every_thread_exited(property);
}

/// The execution view advertises a width, and a run must be able to reach it rather than only
/// claim it.
///
/// Every worker is held at once, so the run reaching the advertised width is the placement rules
/// working rather than the scheduler obliging.
#[test]
fn the_width_a_plan_advertises_is_a_width_the_run_can_reach() {
    let property = "the_width_a_plan_advertises_is_a_width_the_run_can_reach";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(4);
    let chunks = vec![0; 8];
    let specs = vec![
        source_spec("input", &chunks, 0),
        tracked_spec("tracked", "input", 4, false),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let advertised = planned_width(&graph, "tracked");

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });

    let report = result.expect("the run should finish once the held workers are released");
    let width = achieved_width();

    assert_eq!(advertised, 4);
    assert_eq!(
        arrived, advertised,
        "{property}: the plan advertises {advertised} workers and only {arrived} were given a frame",
    );
    assert_eq!(
        width, advertised,
        "{property}: the plan advertises {advertised} workers and the run reached {width}",
    );
    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert_every_thread_exited(property);
}

/// A chunked stream through a frame routed region and on into a chunk routed one.
///
/// The second region can only reach a worker per chunk if the chunk ids the source set are still
/// on the frames when they get there, so holding all four of its workers and counting how many
/// were given something is the evidence they survived.
///
/// All of that evidence lives downstream, in the chunk routed region, so this test is not
/// sensitive to how the region before it routes. It holds chunk id propagation, not frame
/// routing, and the other tests in this file are what hold the routing.
///
/// The two stages run at different widths deliberately. A node joins its producer's region when
/// their widths match, and a region takes the most demanding granularity in it, so two stages of
/// equal width would be one chunk routed region with no frame routing anywhere in it.
#[test]
fn a_frame_routed_region_passes_chunk_ids_through_to_the_region_after_it() {
    let property = "a_frame_routed_region_passes_chunk_ids_through_to_the_region_after_it";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(4);
    let chunks = vec![0, 0, 1, 1, 2, 2, 3, 3];
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("frame_stage", "input", Some(2)),
        tracked_spec("chunk_stage", "frame_stage", 4, true),
        sink_spec("out", "chunk_stage"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "frame_stage"), Routing::ByFrame);
    assert_eq!(routing_of(&graph, "chunk_stage"), Routing::ByChunk);

    let engine = engine_from(&graph, &solved, Settings::default());
    let mut arrived = 0;
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        arrived = held.release_once_all_arrived();
    });

    let report = result.expect("the run should finish once the held workers are released");

    assert_eq!(
        arrived, 4,
        "{property}: the chunk routed region downstream saw {arrived} of the four chunks the \
         source set, so the frame routed region before it did not pass them all through",
    );
    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert_every_thread_exited(property);
}

/// Sixteen frames through a four wide frame routed region, all four workers held at once, then
/// only the first one released.
///
/// The other three never give up the one unit they were first given, so the aggregate can never
/// drain past it and the disperse can never mark anything but the first worker free again. What
/// the first worker goes on to receive is therefore its own headroom alone, not a property of how
/// the whole run happened to spread out.
#[test]
fn a_frame_routed_worker_takes_no_more_than_its_own_headroom() {
    let property = "a_frame_routed_worker_takes_no_more_than_its_own_headroom";
    clear_recorded();
    clear_assigned();

    let held = hold_workers(4);
    let workers = 4;
    let total_units = workers * FRAME_UNITS_IN_FLIGHT;
    let chunks = vec![0; total_units];
    let specs = vec![
        source_spec("input", &chunks, 0),
        tracked_spec("tracked", "input", workers, false),
        sink_spec("out", "tracked"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "tracked"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        for _ in 0..workers {
            held.arrivals
                .recv_timeout(WATCHDOG)
                .expect("every worker should reach its gate");
        }

        held.releases[0].send(()).ok();

        for unit in 0..FRAME_UNITS_IN_FLIGHT {
            held.processed.recv_timeout(WATCHDOG).unwrap_or_else(|_| {
                panic!("{property}: the released worker never reached unit {unit} of its own headroom")
            });
        }

        let beyond_headroom = held.processed.recv_timeout(NOTHING_MORE_COMING);
        assert!(
            beyond_headroom.is_err(),
            "{property}: the released worker was handed a unit beyond the {FRAME_UNITS_IN_FLIGHT} \
             its headroom allows, while the other three workers still held theirs",
        );

        held.release();
    });

    let report = result.expect("the run should finish once every worker is released");

    assert_eq!(recorded().len(), total_units);
    assert_eq!(report.frames, total_units as u64);
    assert_every_thread_exited(property);
}
