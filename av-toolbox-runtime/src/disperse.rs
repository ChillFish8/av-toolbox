use av_toolbox_gpu::StreamItem;

use crate::{Assignment, ExchangeError, WorkItem};

/// Chooses which worker takes the next chunk.
pub trait Dispatch {
    /// Picks a worker for `chunk`, or `None` if no worker is free yet.
    ///
    /// A policy that hands out workers by availability answers `None` until one frees up. The
    /// caller is expected to try again later rather than treat `None` as an error.
    fn choose(&mut self, chunk: u64) -> Option<usize>;

    /// How many workers this policy can choose between.
    ///
    /// A disperse needs this to end every worker when the stream finishes, including one that
    /// was never assigned a chunk.
    fn workers(&self) -> usize;
}

/// Hands each chunk to the next worker in turn.
pub struct RoundRobin {
    workers: usize,
    next: usize,
}

impl RoundRobin {
    pub fn new(workers: usize) -> Self {
        assert!(workers > 0, "a region always has at least one worker");

        Self { workers, next: 0 }
    }
}

impl Dispatch for RoundRobin {
    fn choose(&mut self, _chunk: u64) -> Option<usize> {
        let chosen = self.next;
        self.next = (self.next + 1) % self.workers;
        Some(chosen)
    }

    fn workers(&self) -> usize {
        self.workers
    }
}

/// One message, and the worker it is for.
#[derive(Debug)]
pub struct Routed {
    pub worker: usize,
    pub item: WorkItem,
}

/// What placing a held bundle produced.
pub struct Placed {
    pub assignment: Assignment,
    pub routed: Routed,
}

/// What `finish` produced, including anything the stream ending left unrouted.
pub struct Finished {
    pub routed: Vec<Routed>,
    pub dropped: Option<Dropped>,
}

/// A bundle that never made it to a worker because the stream ended first.
#[derive(Debug)]
pub enum Dropped {
    /// Some slots held items but the bundle never completed.
    Partial(Vec<Option<StreamItem>>),
    /// The bundle completed but no worker was ever chosen for it.
    Unplaced(Vec<Option<StreamItem>>),
}

/// The chunk currently being routed, and the worker holding it.
#[derive(Debug, Copy, Clone)]
struct Current {
    chunk: u64,
    worker: usize,
}

/// A completed bundle waiting for `place` to give it a worker.
struct HeldBundle {
    chunk: u64,
    bundle: Vec<Option<StreamItem>>,
}

/// Gathers one item per bound input slot into a bundle, then routes each chunk to one worker.
///
/// A slot the signature never bound stays `None` in every bundle. A bundle is complete once every
/// bound slot holds an item, and it is only routed once a worker has been chosen for its chunk.
pub struct Disperse<D> {
    slots: usize,
    bound: Vec<bool>,
    authority: usize,
    dispatch: D,
    pending: Vec<Option<StreamItem>>,
    current: Option<Current>,
    held: Option<HeldBundle>,
    assignments: Vec<Assignment>,
    finished: bool,
}

impl<D: Dispatch> Disperse<D> {
    /// Builds a disperse for a signature whose bound slots are marked `true` in `bound`.
    ///
    /// `bound.len()` sets how many slots every bundle carries, one per signature param and one
    /// per variadic input, with `false` for a slot no input was ever bound to. `authority` must
    /// name a bound slot.
    pub fn new(bound: &[bool], authority: usize, dispatch: D) -> Self {
        assert!(
            authority < bound.len(),
            "the chunk authority must be one of the disperse's slots"
        );
        assert!(bound[authority], "the chunk authority slot must be bound");

        Self {
            slots: bound.len(),
            bound: bound.to_vec(),
            authority,
            dispatch,
            pending: vec![None; bound.len()],
            current: None,
            held: None,
            assignments: Vec::new(),
            finished: false,
        }
    }

    /// The chunk given to each worker, in the order it was dispatched.
    pub fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }

    /// Whether a completed bundle is waiting for `place` to give it a worker.
    pub fn is_holding(&self) -> bool {
        self.held.is_some()
    }

    /// Holds `item` against `slot`, and returns whatever it just made ready for a worker.
    ///
    /// A slot already holding an item means that input ran ahead of the others before the bundle
    /// it belongs to could complete. The new item is refused rather than overwriting the one
    /// already held, so the caller learns of it instead of a frame silently disappearing.
    ///
    /// Completing a bundle never chooses its worker here, not even the very first one. It is held
    /// instead, so [Disperse::place] can choose once a worker is free without a boundary's close
    /// sitting undelivered while a policy decides. Offering again before placing that bundle is an
    /// error.
    pub fn offer(&mut self, slot: usize, item: StreamItem) -> Result<Vec<Routed>, ExchangeError> {
        if self.finished {
            return Err(ExchangeError::AlreadyFinished);
        }

        if self.held.is_some() {
            return Err(ExchangeError::PlacementPending);
        }

        if slot >= self.slots {
            return Err(ExchangeError::SlotOutOfRange {
                slot,
                slots: self.slots,
            });
        }

        if !self.bound[slot] {
            return Err(ExchangeError::UnboundSlot { slot });
        }

        if let Some(held) = &self.pending[slot] {
            return Err(ExchangeError::SlotRanAhead {
                slot,
                held: held.meta().index,
                offered: item.meta().index,
            });
        }

        self.pending[slot] = Some(item);

        if !self.is_complete() {
            return Ok(Vec::new());
        }

        self.route_bundle()
    }

    /// Gives the held bundle a worker, once one is free.
    ///
    /// Returns `None` when nothing is held, or when [Dispatch::choose] has no worker free yet. A
    /// caller sees `None` for the second reason as "not yet" and is expected to try again later.
    pub fn place(&mut self) -> Option<Placed> {
        let held = self.held.as_ref()?;
        let worker = self.dispatch.choose(held.chunk)?;
        let held = self.held.take().expect("just confirmed a bundle is held");

        let assignment = Assignment {
            chunk: held.chunk,
            worker,
        };
        self.assignments.push(assignment);
        self.current = Some(Current {
            chunk: held.chunk,
            worker,
        });

        Some(Placed {
            assignment,
            routed: Routed {
                worker,
                item: WorkItem::Bundle(held.bundle),
            },
        })
    }

    /// Closes the chunk in flight and ends every worker, once the input streams are over.
    ///
    /// A second call is a no-op, since the first already closed everything. Whatever the stream
    /// ended in the middle of, a partially gathered bundle or a complete one still waiting for
    /// `place`, is reported as [Dropped] rather than silently discarded.
    pub fn finish(&mut self) -> Finished {
        if self.finished {
            return Finished {
                routed: Vec::new(),
                dropped: None,
            };
        }

        self.finished = true;

        let mut routed = Vec::new();

        if let Some(current) = self.current {
            routed.push(Routed {
                worker: current.worker,
                item: WorkItem::EndOfChunk(current.chunk),
            });
        }

        for worker in 0..self.dispatch.workers() {
            routed.push(Routed {
                worker,
                item: WorkItem::EndOfStream,
            });
        }

        let dropped = self.take_dropped();

        Finished { routed, dropped }
    }

    fn take_dropped(&mut self) -> Option<Dropped> {
        if let Some(held) = self.held.take() {
            return Some(Dropped::Unplaced(held.bundle));
        }

        if self.pending.iter().any(Option::is_some) {
            let partial = std::mem::replace(&mut self.pending, vec![None; self.slots]);
            return Some(Dropped::Partial(partial));
        }

        None
    }

    fn is_complete(&self) -> bool {
        self.bound
            .iter()
            .zip(&self.pending)
            .all(|(&bound, item)| !bound || item.is_some())
    }

    fn route_bundle(&mut self) -> Result<Vec<Routed>, ExchangeError> {
        check_frame_indices(&self.pending, &self.bound)?;

        let chunk = self.pending[self.authority]
            .as_ref()
            .expect("the authority slot is bound and filled once a bundle is complete")
            .meta()
            .chunk;

        let bundle = std::mem::replace(&mut self.pending, vec![None; self.slots]);

        let Some(current) = self.current else {
            self.held = Some(HeldBundle { chunk, bundle });
            return Ok(Vec::new());
        };

        if chunk < current.chunk {
            return Err(ExchangeError::ChunkWentBackwards {
                previous: current.chunk,
                found: chunk,
            });
        }

        if chunk == current.chunk {
            return Ok(vec![Routed {
                worker: current.worker,
                item: WorkItem::Bundle(bundle),
            }]);
        }

        self.held = Some(HeldBundle { chunk, bundle });

        Ok(vec![Routed {
            worker: current.worker,
            item: WorkItem::EndOfChunk(current.chunk),
        }])
    }
}

/// Checks every bound, filled slot in `pending` carries the same frame index.
fn check_frame_indices(pending: &[Option<StreamItem>], bound: &[bool]) -> Result<(), ExchangeError> {
    let mut expected: Option<(usize, u64)> = None;

    for (slot, item) in pending.iter().enumerate() {
        if !bound[slot] {
            continue;
        }

        let frame_index = item
            .as_ref()
            .expect("every bound slot is filled once a bundle is complete")
            .meta()
            .index;

        let Some((expected_slot, expected_index)) = expected else {
            expected = Some((slot, frame_index));
            continue;
        };

        if frame_index != expected_index {
            return Err(ExchangeError::OutOfStep {
                slot,
                found: frame_index,
                expected_slot,
                expected: expected_index,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use av_toolbox_gpu::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        SampleFormat,
        StreamItem,
        Subsampling,
        testing_context,
    };

    use super::*;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        }
    }

    fn item(index: u64, chunk: u64) -> StreamItem {
        let context = testing_context();
        let meta = FrameMeta::at(index).in_chunk(chunk);
        StreamItem::Frame(context.alloc_frame(&format(), meta))
    }

    fn item_in_chunk(index: u64, chunk: u64) -> StreamItem {
        item(index, chunk)
    }

    /// Answers exactly what `script` says, one answer per `choose` call, in order.
    struct ScriptedChoice {
        script: VecDeque<Option<usize>>,
        workers: usize,
    }

    impl ScriptedChoice {
        fn new(script: &[Option<usize>], workers: usize) -> Self {
            Self {
                script: script.iter().copied().collect(),
                workers,
            }
        }
    }

    impl Dispatch for ScriptedChoice {
        fn choose(&mut self, _chunk: u64) -> Option<usize> {
            self.script
                .pop_front()
                .expect("the script names an answer for every choose call")
        }

        fn workers(&self) -> usize {
            self.workers
        }
    }

    #[test]
    fn a_bundle_is_held_back_until_every_slot_has_an_item() {
        let mut disperse = Disperse::new(&[true, true], 0, RoundRobin::new(2));

        let ready = disperse.offer(0, item(0, 0)).expect("should accept");
        assert!(ready.is_empty());

        let ready = disperse.offer(1, item(0, 0)).expect("should accept");
        assert!(ready.is_empty());
        assert!(disperse.is_holding());

        let placed = disperse.place().expect("round robin always has a free worker");

        assert!(matches!(placed.routed.item, WorkItem::Bundle(_)));
    }

    #[test]
    fn inputs_that_disagree_about_the_frame_are_an_error() {
        let mut disperse = Disperse::new(&[true, true], 0, RoundRobin::new(2));
        disperse.offer(0, item(4, 0)).expect("should accept");

        let result = disperse.offer(1, item(5, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected inputs that disagree about the frame");
        };

        assert_eq!(
            error,
            ExchangeError::OutOfStep {
                slot: 1,
                found: 5,
                expected_slot: 0,
                expected: 4,
            }
        );
    }

    #[test]
    fn every_frame_of_one_chunk_goes_to_one_worker() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(4));
        disperse.offer(0, item(0, 7)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");

        let second = disperse.offer(0, item(1, 7)).expect("should accept");

        assert_eq!(first.routed.worker, second[0].worker);
    }

    #[test]
    fn a_new_chunk_closes_the_last_one_and_moves_worker() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(4));
        disperse.offer(0, item(0, 7)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");
        let first_worker = first.routed.worker;

        let routed = disperse.offer(0, item(1, 8)).expect("should accept");

        assert!(matches!(routed[0].item, WorkItem::EndOfChunk(7)));
        assert_eq!(routed[0].worker, first_worker);
        assert_eq!(routed.len(), 1);

        let placed = disperse.place().expect("round robin always has a free worker");

        assert_ne!(placed.routed.worker, first_worker);
        assert!(matches!(placed.routed.item, WorkItem::Bundle(_)));
    }

    #[test]
    fn a_boundary_hands_back_the_close_before_a_worker_is_chosen() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let before = disperse.assignments().len();
        let routed = disperse.offer(0, item(1, 8)).expect("should accept");

        assert!(matches!(routed[0].item, WorkItem::EndOfChunk(7)));
        assert_eq!(disperse.assignments().len(), before);
        assert!(disperse.is_holding());
    }

    #[test]
    fn offering_again_before_placing_a_held_bundle_is_an_error() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");

        let result = disperse.offer(0, item(1, 8));

        let Err(error) = result else {
            panic!("the disperse should have refused an offer while a bundle is held");
        };

        assert_eq!(error, ExchangeError::PlacementPending);
    }

    /// This is the shape the next phase relies on: every completed bundle, the first one
    /// included, defers `choose` to `place`, and a policy that has no free worker yet can say so
    /// without panicking or blocking, then succeed once one frees up.
    #[test]
    fn a_dispatch_with_no_free_worker_at_a_boundary_can_decline_without_panicking() {
        let script = [Some(0), None, Some(1)];
        let mut disperse = Disperse::new(&[true], 0, ScriptedChoice::new(&script, 2));

        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse
            .place()
            .expect("the policy frees a worker on its first try");

        let boundary = disperse.offer(0, item(1, 8)).expect("should accept");

        assert!(matches!(boundary[0].item, WorkItem::EndOfChunk(7)));
        assert_eq!(disperse.assignments().len(), 1);

        assert!(disperse.place().is_none());
        assert_eq!(disperse.assignments().len(), 1);

        let placed = disperse
            .place()
            .expect("the policy frees a worker on its second try");

        assert_eq!(placed.assignment, Assignment { chunk: 8, worker: 1 });
        assert_eq!(disperse.assignments().len(), 2);
        assert!(!disperse.is_holding());
    }

    #[test]
    fn the_first_chunk_can_also_defer_when_no_worker_is_free_yet() {
        let script = [None, Some(0)];
        let mut disperse = Disperse::new(&[true], 0, ScriptedChoice::new(&script, 1));

        let ready = disperse.offer(0, item(0, 7)).expect("should accept");

        assert!(ready.is_empty());
        assert!(disperse.is_holding());

        assert!(disperse.place().is_none());
        assert!(disperse.is_holding());

        let placed = disperse
            .place()
            .expect("the policy frees a worker on its second try");

        assert_eq!(placed.assignment, Assignment { chunk: 7, worker: 0 });
    }

    #[test]
    fn routing_follows_the_authority_slot_and_not_the_first_one() {
        // Slot 1 is the authority. Slot 0 carries a different chunking that must be ignored.
        let mut disperse = Disperse::new(&[true, true], 1, RoundRobin::new(4));

        disperse.offer(0, item_in_chunk(0, 99)).expect("should accept");
        disperse.offer(1, item_in_chunk(0, 7)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item_in_chunk(1, 55)).expect("should accept");
        let second = disperse.offer(1, item_in_chunk(1, 7)).expect("should accept");

        assert_eq!(first.routed.worker, second[0].worker);
    }

    #[test]
    fn a_hole_before_the_authority_slot_still_selects_the_right_input() {
        // Slot 0 is unbound. Slot 2, the authority, must still be found past the hole.
        let mut disperse = Disperse::new(&[false, true, true], 2, RoundRobin::new(4));

        let ready = disperse.offer(1, item_in_chunk(0, 9)).expect("should accept");
        assert!(ready.is_empty());

        let ready = disperse.offer(2, item_in_chunk(0, 9)).expect("should accept");
        assert!(ready.is_empty());

        let placed = disperse.place().expect("round robin always has a free worker");

        let WorkItem::Bundle(bundle) = &placed.routed.item else {
            panic!("a complete bundle should have routed");
        };
        assert!(bundle[0].is_none());
        assert!(bundle[1].is_some());
        assert!(bundle[2].is_some());
    }

    #[test]
    fn a_signature_with_a_hole_completes_its_bundles_and_routes_them() {
        let mut disperse = Disperse::new(&[true, false, true], 0, RoundRobin::new(2));

        let ready = disperse.offer(0, item(0, 0)).expect("should accept");
        assert!(ready.is_empty());

        let ready = disperse.offer(2, item(0, 0)).expect("should accept");
        assert!(ready.is_empty());

        let placed = disperse.place().expect("round robin always has a free worker");

        let WorkItem::Bundle(bundle) = &placed.routed.item else {
            panic!("a complete bundle should have routed");
        };
        assert_eq!(bundle.len(), 3);
        assert!(bundle[0].is_some());
        assert!(bundle[1].is_none());
        assert!(bundle[2].is_some());
    }

    #[test]
    fn offering_an_unbound_slot_is_an_error() {
        let mut disperse = Disperse::new(&[true, false], 0, RoundRobin::new(2));

        let result = disperse.offer(1, item(0, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected an item for an unbound slot");
        };

        assert_eq!(error, ExchangeError::UnboundSlot { slot: 1 });
    }

    #[test]
    fn offering_a_slot_out_of_range_is_an_error() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));

        let result = disperse.offer(4, item(0, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected a slot out of range");
        };

        assert_eq!(error, ExchangeError::SlotOutOfRange { slot: 4, slots: 1 });
    }

    #[test]
    #[should_panic(expected = "chunk authority must be one of the disperse's slots")]
    fn an_authority_out_of_range_panics_at_construction() {
        Disperse::new(&[true], 4, RoundRobin::new(2));
    }

    #[test]
    #[should_panic(expected = "chunk authority slot must be bound")]
    fn an_unbound_authority_panics_at_construction() {
        Disperse::new(&[true, false], 1, RoundRobin::new(2));
    }

    #[test]
    fn a_chunk_id_that_goes_backwards_is_an_error() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 8)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let result = disperse.offer(0, item(1, 7));

        let Err(error) = result else {
            panic!("the disperse should have rejected a chunk id that went backwards");
        };

        assert_eq!(
            error,
            ExchangeError::ChunkWentBackwards {
                previous: 8,
                found: 7,
            }
        );
    }

    #[test]
    fn a_chunk_id_may_skip() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 1)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let result = disperse.offer(0, item(1, 9));

        assert!(result.is_ok());
    }

    #[test]
    fn finishing_closes_the_chunk_in_flight_and_ends_every_worker() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(3));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let closing = disperse.finish();

        assert!(matches!(closing.routed[0].item, WorkItem::EndOfChunk(7)));
        let ends = closing
            .routed
            .iter()
            .filter(|routed| matches!(routed.item, WorkItem::EndOfStream))
            .count();
        assert_eq!(ends, 3);
        assert!(closing.dropped.is_none());
    }

    #[test]
    fn finishing_with_a_half_gathered_bundle_reports_it_instead_of_dropping_it() {
        let mut disperse = Disperse::new(&[true, true], 0, RoundRobin::new(1));
        disperse.offer(0, item(0, 0)).expect("should accept");

        let finished = disperse.finish();

        let Some(Dropped::Partial(bundle)) = finished.dropped else {
            panic!("a half-gathered bundle should have been reported as dropped");
        };
        assert!(bundle[0].is_some());
        assert!(bundle[1].is_none());
    }

    #[test]
    fn finishing_with_a_held_bundle_reports_it_as_unplaced() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(1));
        disperse.offer(0, item(0, 7)).expect("should accept");

        assert!(disperse.is_holding());

        let finished = disperse.finish();

        let Some(Dropped::Unplaced(bundle)) = finished.dropped else {
            panic!("a bundle still waiting for a worker should have been reported as dropped");
        };
        assert!(bundle[0].is_some());
    }

    #[test]
    fn finishing_twice_is_a_no_op() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");

        let first = disperse.finish();
        let second = disperse.finish();

        assert!(!first.routed.is_empty());
        assert!(second.routed.is_empty());
        assert!(second.dropped.is_none());
    }

    #[test]
    fn offering_after_finish_is_an_error() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.finish();

        let result = disperse.offer(0, item(1, 8));

        let Err(error) = result else {
            panic!("the disperse should have rejected an item offered after it finished");
        };

        assert_eq!(error, ExchangeError::AlreadyFinished);
    }

    #[test]
    fn the_assignments_name_every_chunk_in_the_order_it_was_dispatched() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(1, 8)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        assert_eq!(
            disperse.assignments(),
            &[
                Assignment { chunk: 7, worker: 0 },
                Assignment { chunk: 8, worker: 1 },
            ]
        );
    }

    #[test]
    fn a_slot_offered_a_second_item_before_the_bundle_completes_is_an_error() {
        let mut disperse = Disperse::new(&[true, true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 0)).expect("should accept");

        let result = disperse.offer(0, item(1, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected a second item on a slot still held");
        };

        assert_eq!(
            error,
            ExchangeError::SlotRanAhead {
                slot: 0,
                held: 0,
                offered: 1,
            }
        );
    }
}
