use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;

use av_toolbox_gpu::AnyItem;

use super::channel::{AggregateSignal, ExitMessage, send_routed};
use super::freedom::{FirstIdle, WorkerFreedom};
use super::region::DisperseSide;
use crate::disperse::Dropped;
use crate::{Cancel, Disperse, ExchangeError, RunError, WorkItem};

/// Drives a region's disperse from its one merged, slot-tagged upstream channel.
///
/// `Disperse::offer` accepts at most one pending item per slot before raising `SlotRanAhead`, so
/// this cannot offer a message the instant it arrives: a slot fed by a faster producer would run
/// straight into that the moment the two producers ran at different speeds. Instead this keeps
/// its own queue per bound slot, pushes every arrival onto the queue for the slot it names, and
/// only offers once every slot's queue has something waiting. It then takes one item from each
/// slot's queue, in slot order, offering each in turn to complete one whole bundle before
/// checking again, and while [Disperse::is_holding] loops [Disperse::place], sending what that
/// returns and retrying when it answers `None`. A fast slot's queue simply grows rather than
/// panicking, so how far it may run ahead of a slow one is bounded by the frame budget rather
/// than capped at one item.
///
/// Owning the disperse outright, on one thread, is what keeps the state machine's own
/// lock-step assumption true: `Disperse` assumes whoever calls `offer` and `place` does so in
/// one sequence, and a second thread offering into the same instance from an independent
/// upstream would have no ordering against this one at all.
///
/// Waiting for a free worker is the one place here a stop has to reach by hand, since every
/// other wait is a receive that ends when its senders do. Once it gives up waiting, whatever is
/// still queued or still held has missed its run, so it is counted into `dropped` rather than
/// going quietly.
pub(super) fn drive_disperse(
    side: DisperseSide,
    cancel: &Cancel,
    dropped: &AtomicUsize,
) -> Result<(), RunError> {
    let DisperseSide {
        mut disperse,
        upstream,
        slots,
        freedom,
        work_senders,
        assignment_sender,
    } = side;

    let mut queues: HashMap<usize, VecDeque<AnyItem>> =
        slots.iter().map(|&slot| (slot, VecDeque::new())).collect();
    let mut done: HashSet<usize> = HashSet::new();

    'routing: loop {
        if cancel.is_stopped() {
            break;
        }

        while slots.iter().all(|slot| !queues[slot].is_empty()) {
            for &slot in &slots {
                let item = queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .pop_front()
                    .expect("this slot's queue was just confirmed non-empty");

                let offered = disperse.offer(slot, item);
                let routed = offered.map_err(refused_offer)?;
                send_routed(&work_senders, routed);

                let placed =
                    drain_holding(&mut disperse, &freedom, cancel, &work_senders, &assignment_sender);
                if !placed {
                    break 'routing;
                }
            }
        }

        if done.len() == slots.len() {
            break;
        }

        match upstream.recv() {
            Ok((slot, ExitMessage::Item(item))) => {
                queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .push_back(item);
            },
            Ok((slot, ExitMessage::Done)) => {
                done.insert(slot);
            },
            Err(_) => break,
        }
    }

    let finished = disperse.finish();
    for assignment in finished.assignments {
        assignment_sender
            .send(AggregateSignal::Assignment(assignment))
            .ok();
    }

    send_routed(&work_senders, finished.routed);
    assignment_sender.send(AggregateSignal::Finished).ok();

    let queued: usize = queues.values().map(VecDeque::len).sum();
    dropped.fetch_add(queued + dropped_items(finished.dropped), Ordering::SeqCst);

    Ok(())
}

/// What a refused offer means for the run.
///
/// Two inputs carrying different frames, and a chunk id going backwards, are things the data did,
/// and both fail the run. Every other refusal is about how the disperse was driven, and this
/// driver gathers a whole bundle before it offers any of it, so none of them can happen here.
fn refused_offer(error: ExchangeError) -> RunError {
    match error {
        ExchangeError::OutOfStep { .. } | ExchangeError::ChunkWentBackwards { .. } => {
            RunError::Exchange { source: error }
        },
        other => panic!("a slot only offers once every slot's queue has an item ready: {other}"),
    }
}

/// How many items a bundle the stream ended on was carrying.
fn dropped_items(dropped: Option<Dropped>) -> usize {
    match dropped {
        Some(Dropped::Partial(bundle) | Dropped::Unplaced(bundle)) => {
            bundle.iter().filter(|slot| slot.is_some()).count()
        },
        None => 0,
    }
}

/// Places whatever the disperse is holding, waiting for a worker to free up when it has to.
///
/// Answers false when the run stopped, leaving the bundle held, which tells its caller to stop
/// offering rather than run into a placement that is never coming. The stop is read before every
/// placement rather than only where this waits, so a stopped run never starts one more chunk off
/// on a worker that happens to free up as everything is winding down.
fn drain_holding(
    disperse: &mut Disperse<FirstIdle>,
    freedom: &WorkerFreedom,
    cancel: &Cancel,
    work_senders: &[Sender<WorkItem>],
    assignment_sender: &Sender<AggregateSignal>,
) -> bool {
    while disperse.is_holding() {
        if cancel.is_stopped() {
            return false;
        }

        match disperse.place() {
            Some(placed) => {
                assignment_sender
                    .send(AggregateSignal::Assignment(placed.assignment))
                    .ok();
                send_routed(work_senders, vec![placed.routed]);
            },
            None => freedom.wait_for_a_free_worker(cancel),
        }
    }

    true
}
