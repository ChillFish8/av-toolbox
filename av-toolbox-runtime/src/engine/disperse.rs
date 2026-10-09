use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::Sender;

use av_toolbox_gpu::{MetaRecord, RecordId};

use super::channel::{AggregateSignal, ExitMessage, send_routed};
use super::freedom::{FirstIdle, WorkerFreedom};
use super::region::DisperseSide;
use crate::disperse::Dropped;
use crate::{Cancel, Disperse, ExchangeError, Flow, RunError, WorkItem};

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
/// Records never join a bundle. Every record at the front of a slot's queue is handed to the disperse
/// before the next bundle forms, so it goes to that bundle's worker ahead of it, and a bundle only
/// forms once every slot's front is an item. A record reaching this by more than one slot is handed
/// over once. Whatever records are left when the stream ends are handed over before the disperse
/// finishes.
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

    let mut queues: HashMap<usize, VecDeque<Flow>> =
        slots.iter().map(|&slot| (slot, VecDeque::new())).collect();
    let mut done: HashSet<usize> = HashSet::new();
    let mut seen: HashSet<RecordId> = HashSet::new();

    'routing: loop {
        if cancel.is_stopped() {
            break;
        }

        loop {
            let records = take_front_records(&slots, &mut queues, &mut seen);
            disperse.offer_records(records);

            let ready = slots.iter().all(|slot| front_is_item(&queues[slot]));
            if !ready {
                break;
            }

            for &slot in &slots {
                let item = queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .pop_front()
                    .and_then(Flow::into_item)
                    .expect("this slot's front was just confirmed an item");

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
                    .push_back(Flow::Item(item));
            },
            Ok((slot, ExitMessage::Record(record))) => {
                queues
                    .get_mut(&slot)
                    .expect("every bound slot keeps its own queue")
                    .push_back(Flow::Record(record));
            },
            Ok((slot, ExitMessage::Done)) => {
                done.insert(slot);
            },
            Err(_) => break,
        }
    }

    let leftover_records = take_every_record(&slots, &mut queues, &mut seen);
    disperse.offer_records(leftover_records);

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

/// Takes every record at the front of each slot's queue, in slot order, leaving out any already seen.
fn take_front_records(
    slots: &[usize],
    queues: &mut HashMap<usize, VecDeque<Flow>>,
    seen: &mut HashSet<RecordId>,
) -> Vec<MetaRecord> {
    let mut records = Vec::new();

    for slot in slots {
        let queue = queues
            .get_mut(slot)
            .expect("every bound slot keeps its own queue");

        while let Some(flow) = queue.pop_front() {
            let Flow::Record(record) = flow else {
                queue.push_front(flow);
                break;
            };

            keep_first_sighting(record, seen, &mut records);
        }
    }

    records
}

/// Takes every record still queued on any slot, in slot order, leaving the items behind.
fn take_every_record(
    slots: &[usize],
    queues: &mut HashMap<usize, VecDeque<Flow>>,
    seen: &mut HashSet<RecordId>,
) -> Vec<MetaRecord> {
    let mut records = Vec::new();

    for slot in slots {
        let queue = queues
            .get_mut(slot)
            .expect("every bound slot keeps its own queue");
        let mut items = VecDeque::with_capacity(queue.len());

        for flow in queue.drain(..) {
            match flow {
                Flow::Item(item) => items.push_back(Flow::Item(item)),
                Flow::Record(record) => keep_first_sighting(record, seen, &mut records),
            }
        }

        *queue = items;
    }

    records
}

fn keep_first_sighting(record: MetaRecord, seen: &mut HashSet<RecordId>, records: &mut Vec<MetaRecord>) {
    let first_sighting = seen.insert(record.id());
    if first_sighting {
        records.push(record);
    }
}

fn front_is_item(queue: &VecDeque<Flow>) -> bool {
    matches!(queue.front(), Some(Flow::Item(_)))
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
                send_routed(work_senders, placed.in_order());
            },
            None => freedom.wait_for_a_free_worker(cancel),
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{AnyItem, FrameMeta, StreamItem, testing_context};

    use super::*;
    use crate::engine::tests::format;

    fn item(index: u64) -> Flow {
        let context = testing_context();
        let meta = FrameMeta::at(index);
        let frame = context.alloc_frame(&format(), meta);
        let item: AnyItem = StreamItem::Frame(frame).into();

        Flow::Item(item)
    }

    fn take_bundle(slots: &[usize], queues: &mut HashMap<usize, VecDeque<Flow>>) {
        for slot in slots {
            let front = queues.get_mut(slot).expect("every slot has a queue").pop_front();
            assert!(matches!(front, Some(Flow::Item(_))));
        }
    }

    /// The second slot is fed through a stage holding one frame back, so it passes the record on
    /// ahead of frame 0 while the first slot carries it behind frame 0.
    #[test]
    fn a_record_on_a_lagging_slot_is_taken_once_across_two_bundles() {
        let slots = [0, 1];
        let record = MetaRecord::new("mark");
        let mut queues: HashMap<usize, VecDeque<Flow>> = HashMap::new();
        let first_slot = VecDeque::from([item(0), Flow::Record(record.clone()), item(1)]);
        let second_slot = VecDeque::from([Flow::Record(record.clone()), item(0)]);
        queues.insert(0, first_slot);
        queues.insert(1, second_slot);
        let mut seen = HashSet::new();

        let before_first = take_front_records(&slots, &mut queues, &mut seen);
        take_bundle(&slots, &mut queues);
        let before_second = take_front_records(&slots, &mut queues, &mut seen);

        let ids: Vec<RecordId> = before_first.iter().map(MetaRecord::id).collect();
        assert_eq!(ids, vec![record.id()]);
        assert!(before_second.is_empty());
        assert!(front_is_item(&queues[&0]));
    }

    #[test]
    fn records_left_at_the_end_leave_out_those_already_taken() {
        let slots = [0, 1];
        let taken = MetaRecord::new("mark");
        let fresh = MetaRecord::new("chunk");
        let mut queues: HashMap<usize, VecDeque<Flow>> = HashMap::new();
        let first_slot = VecDeque::from([item(0), Flow::Record(taken.clone())]);
        let second_slot = VecDeque::from([Flow::Record(fresh.clone()), Flow::Record(taken.clone())]);
        queues.insert(0, first_slot);
        queues.insert(1, second_slot);
        let mut seen = HashSet::from([taken.id()]);

        let leftover = take_every_record(&slots, &mut queues, &mut seen);

        let ids: Vec<RecordId> = leftover.iter().map(MetaRecord::id).collect();
        assert_eq!(ids, vec![fresh.id()]);
        assert_eq!(queues[&0].len(), 1);
        assert!(queues[&1].is_empty());
    }
}
