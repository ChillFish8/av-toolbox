use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_gpu::{AnyItem, Attachments, Frame, FrameContext, FrameMeta, StreamItem, testing_context};
use av_toolbox_graph::{NodeId, Routing, StageBox};

use crate::engine::tests::fixtures::filters::{PASS_ENDS, PASS_STARTS, PassInstance};
use crate::engine::tests::fixtures::shaping::MultiplyingInstance;
use crate::engine::tests::harness::{
    WATCHDOG,
    assert_every_thread_exited,
    clear_recorded,
    engine_from,
    planned_graph,
    recorded,
    routing_of,
    run_with_watchdog,
};
use crate::engine::tests::specs::{filter_spec, sink_spec, source_spec};
use crate::engine::tests::support::format;
use crate::engine::{
    AggregateSide,
    FRAME_UNITS_IN_FLIGHT,
    HEADROOM_ITEMS_PER_WORKER,
    WorkerFreedom,
    WorkerSlot,
    run_aggregate,
    run_worker,
};
use crate::tests::routes::{meta_in, route};
use crate::{
    Aggregate,
    Assignment,
    Cancel,
    ExchangeError,
    RegionEdges,
    RegionRunner,
    RunError,
    Settings,
    WorkItem,
    WorkResult,
};

/// A one-slot bundle carrying a single frame in `chunk` of layer 0, for driving [run_worker]
/// directly with a [WorkItem::Unit] rather than through a disperse.
fn unit_bundle(context: &FrameContext, index: u64, chunk: u64) -> Vec<Option<AnyItem>> {
    let meta = meta_in(index, 0, chunk);
    vec![Some(
        StreamItem::Frame(context.alloc_frame(&format(), meta)).into(),
    )]
}

/// A worker slot around one fixture stage reading its chunk from layer 0, with no disperse or
/// aggregate in front of it, for sending [WorkItem]s to [run_worker] directly.
fn one_node_worker(
    stage: StageBox,
    work_receiver: mpsc::Receiver<WorkItem>,
    result_sender: mpsc::SyncSender<WorkResult>,
) -> WorkerSlot {
    let head = NodeId::new(0);
    let nodes = vec![(head, stage)];
    let routes = HashMap::from([(head, route(0))]);
    let edges = RegionEdges::new(vec![], vec![head]).with_routes(routes);

    let runner = RegionRunner::new(nodes, edges).routed_on(Some(route(0)));

    WorkerSlot {
        runner,
        context: testing_context(),
        work_receiver,
        result_sender,
    }
}

/// Fails every unit it is given.
struct AlwaysFailing;

impl StageInstance for AlwaysFailing {}

impl Filter for AlwaysFailing {
    fn process(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        Err(StageError::msg("this filter always fails"))
    }
}

#[test]
fn a_frame_routed_worker_starts_its_region_once_for_the_whole_run() {
    PASS_STARTS.store(0, Ordering::SeqCst);
    PASS_ENDS.store(0, Ordering::SeqCst);

    let (work_sender, work_receiver) = mpsc::channel();
    let (result_sender, result_receiver) = mpsc::sync_channel(64);
    let worker = one_node_worker(
        StageBox::Filter(Box::new(PassInstance::new())),
        work_receiver,
        result_sender,
    );
    let build_context = testing_context();
    let handle = thread::spawn(move || run_worker(worker));

    for unit in 0..8 {
        let bundle = unit_bundle(&build_context, unit, unit);
        work_sender
            .send(WorkItem::Unit(unit, bundle))
            .expect("the worker should still be receiving");
    }
    work_sender
        .send(WorkItem::EndOfStream)
        .expect("the worker should still be receiving");

    let result = handle.join().expect("worker thread should not panic");
    result.expect("a frame routed worker with no failures should not error");
    drop(result_receiver);

    let starts = PASS_STARTS.load(Ordering::SeqCst);
    let ends = PASS_ENDS.load(Ordering::SeqCst);

    assert_eq!(
        starts, 1,
        "a frame routed worker started its region {starts} times over eight units",
    );
    assert_eq!(
        ends, 1,
        "a frame routed worker ended its region {ends} times over eight units",
    );
}

/// Copies the unit's timestamp modulo three, so a third of the units produce nothing at all.
/// An `EndOfUnit` is only worth trusting if it arrives whether or not its unit produced an item.
#[test]
fn a_frame_routed_worker_closes_every_unit_it_is_given() {
    let (work_sender, work_receiver) = mpsc::channel();
    let (result_sender, result_receiver) = mpsc::sync_channel(64);
    let worker = one_node_worker(
        StageBox::Filter(Box::new(MultiplyingInstance::new())),
        work_receiver,
        result_sender,
    );
    let build_context = testing_context();
    let handle = thread::spawn(move || run_worker(worker));

    let unit_ids: Vec<u64> = (100..108).collect();
    for &unit in &unit_ids {
        let chunked = meta_in(unit, 0, unit + 900);
        let meta = FrameMeta {
            index: unit,
            timestamp: Some(Duration::from_secs(unit)),
            provided: chunked.provided,
            attachments: Attachments::new(),
        };
        let item = AnyItem::from(StreamItem::Frame(build_context.alloc_frame(&format(), meta)));
        work_sender
            .send(WorkItem::Unit(unit, vec![Some(item)]))
            .expect("the worker should still be receiving");
    }
    work_sender
        .send(WorkItem::EndOfStream)
        .expect("the worker should still be receiving");

    handle
        .join()
        .expect("worker thread should not panic")
        .expect("a frame routed worker with no failures should not error");

    let mut closed_units = Vec::new();
    let mut item_count = 0;
    let mut saw_end_of_stream = false;

    while let Ok(result) = result_receiver.recv_timeout(WATCHDOG) {
        match result {
            WorkResult::Item(_) => item_count += 1,
            WorkResult::EndOfUnit(unit) => closed_units.push(unit),
            WorkResult::EndOfStream => {
                saw_end_of_stream = true;
                break;
            },
        }
    }

    assert_eq!(
        closed_units, unit_ids,
        "every unit should close exactly once, in order"
    );
    assert_eq!(
        item_count, 9,
        "a third of the units should have produced nothing, so items should not equal units",
    );
    assert!(
        saw_end_of_stream,
        "the worker should still report the end of the stream"
    );
}

/// Chunk 42 comes from the failing unit's own bundle, not from its unit id of zero, so a worker
/// that named the chunk from the unit id instead would report the wrong number here.
#[test]
fn a_frame_routed_worker_names_the_chunk_a_failing_units_error_came_from() {
    let (work_sender, work_receiver) = mpsc::channel();
    let (result_sender, _result_receiver) = mpsc::sync_channel(64);
    let worker = one_node_worker(
        StageBox::Filter(Box::new(AlwaysFailing)),
        work_receiver,
        result_sender,
    );
    let build_context = testing_context();
    let handle = thread::spawn(move || run_worker(worker));

    let bundle = unit_bundle(&build_context, 0, 42);
    work_sender
        .send(WorkItem::Unit(0, bundle))
        .expect("the worker should still be receiving");

    let result = handle.join().expect("worker thread should not panic");
    let Err(error) = result else {
        panic!("a failing stage should have failed the worker");
    };

    assert_eq!(error.chunk(), Some(42), "the run reported {error}");
}

/// A worker ending while it still owes a chunk, on a run nothing has stopped.
///
/// The tolerance for this is guarded on the run having stopped, and that is a race rather than an
/// invariant: a thread that fails drops its senders a moment before the stop it raised is
/// recorded, and a worker can flush and end inside that window. Reporting rather than panicking
/// is what leaves the real cause the first error of the run.
#[test]
fn an_aggregate_whose_worker_ends_owing_a_chunk_reports_rather_than_panics() {
    let (result_sender, result_receiver) = mpsc::sync_channel(1);
    let (assignment_sender, assignment_receiver) = mpsc::channel();
    let mut aggregate = Aggregate::new();
    aggregate.assign(Assignment { unit: 0, worker: 0 });

    result_sender.send(WorkResult::EndOfStream).expect("should send");
    drop(assignment_sender);

    let side = AggregateSide {
        aggregate,
        assignment_receiver,
    };
    let freedom = WorkerFreedom::new(1, 1);
    let cancel = Cancel::new();

    let result = run_aggregate(side, vec![result_receiver], &freedom, &cancel, Vec::new());

    let Err(error) = result else {
        panic!("a worker ending while it owed a chunk should have been reported");
    };
    assert!(
        matches!(
            error,
            RunError::Exchange {
                source: ExchangeError::WorkerEndedOwingUnit { .. }
            }
        ),
        "the aggregate reported {error}",
    );
}

#[test]
fn a_capacity_of_one_claims_each_worker_once() {
    let freedom = WorkerFreedom::new(2, 1);

    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), Some(1));
    assert_eq!(freedom.try_claim(), None);
}

#[test]
fn a_worker_below_its_capacity_can_be_claimed_again() {
    let freedom = WorkerFreedom::new(1, 3);

    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), None);
}

/// A frame routed worker takes more than one unit before it is full, which is the entire
/// difference between it and a chunk routed one, and it takes exactly as many as the source
/// budget's per worker headroom was sized for.
#[test]
fn a_frame_routed_worker_holds_the_units_its_headroom_was_sized_for() {
    assert_eq!(
        FRAME_UNITS_IN_FLIGHT, HEADROOM_ITEMS_PER_WORKER,
        "the budget counts this many items per worker, so a worker holding more goes uncounted",
    );

    let freedom = WorkerFreedom::new(1, FRAME_UNITS_IN_FLIGHT);

    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(
        freedom.try_claim(),
        Some(0),
        "a frame routed worker that may hold only one unit is a chunk routed worker",
    );

    for unit in 2..FRAME_UNITS_IN_FLIGHT {
        assert_eq!(freedom.try_claim(), Some(0), "unit {unit} should have had room");
    }

    assert_eq!(freedom.try_claim(), None);
}

#[test]
fn draining_a_unit_returns_one_place_and_no_more() {
    let freedom = WorkerFreedom::new(1, 2);

    freedom.try_claim();
    freedom.try_claim();
    assert_eq!(freedom.try_claim(), None);

    freedom.mark_free(0);

    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), None);
}

/// With three workers at capacity two, the first three claims must land on three different
/// workers rather than stacking two onto worker zero. A frame routed region that stacked would
/// be serial with extra steps.
#[test]
fn claims_spread_across_workers_before_stacking_on_one() {
    let freedom = WorkerFreedom::new(3, 2);

    assert_eq!(freedom.try_claim(), Some(0));
    assert_eq!(freedom.try_claim(), Some(1));
    assert_eq!(freedom.try_claim(), Some(2));
    assert_eq!(freedom.try_claim(), Some(0));
}

/// Eight frames, every one of them in chunk zero, through a frame mode stage four workers wide.
///
/// Chunk routing keeps a whole chunk on one worker, so this is the shape that used to run all
/// eight on worker zero. Which worker ran what is not asserted here, only that the run finishes
/// and hands its frames back in the order they went in.
#[test]
fn a_frame_mode_stage_spreads_across_workers_with_no_chunk_ids_at_all() {
    let property = "a_frame_mode_stage_spreads_across_workers_with_no_chunk_ids_at_all";
    clear_recorded();

    let chunks = vec![0; 8];
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "filter"), Routing::ByFrame);

    let engine = engine_from(&graph, &solved, Settings::default());
    let report = run_with_watchdog(engine, property);

    assert_eq!(recorded(), (0..8).collect::<Vec<u64>>());
    assert_eq!(report.frames, 8);
    assert_every_thread_exited(property);
}
