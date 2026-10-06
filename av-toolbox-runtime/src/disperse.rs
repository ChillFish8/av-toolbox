use av_toolbox_gpu::{AnyItem, ChunkRoute};

use crate::chunks::chunk_in;
use crate::{Assignment, ExchangeError, WorkItem};

/// Chooses which worker takes the next unit.
pub trait Dispatch {
    /// Picks a worker for `unit`, or `None` if no worker is free yet.
    ///
    /// A policy that hands out workers by availability answers `None` until one frees up. The
    /// caller is expected to try again later rather than treat `None` as an error.
    fn choose(&mut self, unit: u64) -> Option<usize>;

    /// How many workers this policy can choose between.
    ///
    /// A disperse needs this to end every worker when the stream finishes, including one that
    /// was never assigned a unit.
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
    fn choose(&mut self, _unit: u64) -> Option<usize> {
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
    /// The units `finish` handed out itself, which an aggregate has not been told about yet.
    pub assignments: Vec<Assignment>,
    pub dropped: Option<Dropped>,
}

/// A bundle that never made it to a worker because the stream ended first.
#[derive(Debug)]
pub enum Dropped {
    /// Some slots held items but the bundle never completed.
    Partial(Vec<Option<AnyItem>>),
    /// The bundle completed but no worker was ever chosen for it.
    Unplaced(Vec<Option<AnyItem>>),
}

/// The chunk currently being routed, and the worker holding it.
#[derive(Debug, Copy, Clone)]
struct Current {
    chunk: u64,
    worker: usize,
}

/// A completed bundle waiting for `place` to give it a worker.
struct HeldBundle {
    unit: u64,
    bundle: Vec<Option<AnyItem>>,
}

/// What a disperse needs to remember, which differs by how it routes.
enum RoutingState {
    /// Routes by the authority slot's chunk id, keeping one worker for a whole chunk.
    ByChunk {
        authority: usize,
        current: Option<Current>,
    },
    /// Routes each bundle on its own, numbering units as it goes.
    ByFrame { next_unit: u64 },
}

/// Gathers one item per bound input slot into a bundle, then routes it to one worker.
///
/// A slot the signature never bound stays `None` in every bundle. A bundle is complete once every
/// bound slot holds an item, and it is only routed once a worker has been chosen for it.
pub struct Disperse<D> {
    slots: usize,
    bound: Vec<bool>,
    routing: RoutingState,
    route: Option<ChunkRoute>,
    dispatch: D,
    pending: Vec<Option<AnyItem>>,
    held: Option<HeldBundle>,
    assignments: Vec<Assignment>,
    finished: bool,
}

impl<D: Dispatch> Disperse<D> {
    /// Builds a disperse for a signature whose bound slots are marked `true` in `bound`.
    ///
    /// `bound.len()` sets how many slots every bundle carries, one per signature param and one
    /// per variadic input, with `false` for a slot no input was ever bound to. `authority` must
    /// name a bound slot, and its chunk id decides which worker keeps a whole chunk.
    pub fn new(bound: &[bool], authority: usize, dispatch: D) -> Self {
        assert!(
            authority < bound.len(),
            "the chunk authority must be one of the disperse's slots"
        );
        assert!(bound[authority], "the chunk authority slot must be bound");

        Self::with_routing(
            bound,
            dispatch,
            RoutingState::ByChunk {
                authority,
                current: None,
            },
        )
    }

    /// Builds a disperse that routes each bundle to whichever worker has room.
    ///
    /// Frame routing reads no chunk id, so it needs no authority. An authority declared on the
    /// head node's signature is simply not used.
    pub fn by_frame(bound: &[bool], dispatch: D) -> Self {
        Self::with_routing(bound, dispatch, RoutingState::ByFrame { next_unit: 0 })
    }

    fn with_routing(bound: &[bool], dispatch: D, routing: RoutingState) -> Self {
        Self {
            slots: bound.len(),
            bound: bound.to_vec(),
            routing,
            route: None,
            dispatch,
            pending: vec![None; bound.len()],
            held: None,
            assignments: Vec::new(),
            finished: false,
        }
    }

    /// The same disperse, reading each bundle's chunk on `route`.
    pub fn with_route(mut self, route: Option<ChunkRoute>) -> Self {
        self.route = route;
        self
    }

    /// The unit given to each worker, in the order it was dispatched.
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
    pub fn offer(&mut self, slot: usize, item: AnyItem) -> Result<Vec<Routed>, ExchangeError> {
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
        let worker = self.dispatch.choose(held.unit)?;
        let held = self.held.take().expect("just confirmed a bundle is held");

        let assignment = Assignment {
            unit: held.unit,
            worker,
        };
        self.assignments.push(assignment);

        let item = match &mut self.routing {
            RoutingState::ByChunk { current, .. } => {
                *current = Some(Current {
                    chunk: held.unit,
                    worker,
                });
                WorkItem::Bundle(held.bundle)
            },
            RoutingState::ByFrame { .. } => WorkItem::Unit(held.unit, held.bundle),
        };

        Some(Placed {
            assignment,
            routed: Routed { worker, item },
        })
    }

    /// Closes whatever is in flight and ends every worker, once the input streams are over.
    ///
    /// A second call is a no-op, since the first already closed everything. Whatever the stream
    /// ended in the middle of, a partially gathered bundle or a complete one still waiting for
    /// `place`, is reported as [Dropped] rather than silently discarded. Under chunk routing, a
    /// chunk whose close was already sent when a held bundle arrived is not closed a second time.
    ///
    /// Frame routing has no chunk in flight to close, since every bundle already closed its own
    /// unit as it was placed. Every worker is given one last unit of its own instead, so whatever
    /// it emits as its stream ends arrives under an assignment its aggregate is still waiting on.
    /// A worker that was never given a bundle gets one too, since it was told its stream started
    /// and so has an end of its own to deliver.
    pub fn finish(&mut self) -> Finished {
        if self.finished {
            return Finished {
                routed: Vec::new(),
                assignments: Vec::new(),
                dropped: None,
            };
        }

        self.finished = true;

        let workers = self.dispatch.workers();
        let mut routed = Vec::new();
        let mut assignments = Vec::new();

        // A held bundle already closed the chunk in flight as it arrived, since that is what a
        // new chunk id does, so closing it again here would hand one worker two ends for the
        // same chunk.
        let chunk_in_flight = match &self.routing {
            RoutingState::ByChunk {
                current: Some(current),
                ..
            } if self.held.is_none() => Some(*current),
            _ => None,
        };

        if let Some(current) = chunk_in_flight {
            routed.push(Routed {
                worker: current.worker,
                item: WorkItem::EndOfChunk(current.chunk),
            });
        }

        if let RoutingState::ByFrame { next_unit } = &mut self.routing {
            for worker in 0..workers {
                let unit = *next_unit;
                *next_unit += 1;

                assignments.push(Assignment { unit, worker });
                routed.push(Routed {
                    worker,
                    item: WorkItem::EndOfChunk(unit),
                });
            }
        }

        self.assignments.extend(&assignments);

        for worker in 0..workers {
            routed.push(Routed {
                worker,
                item: WorkItem::EndOfStream,
            });
        }

        let dropped = self.take_dropped();

        Finished {
            routed,
            assignments,
            dropped,
        }
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

        match &mut self.routing {
            RoutingState::ByChunk { authority, current } => {
                let authoritative = self.pending[*authority]
                    .as_ref()
                    .expect("the authority slot is bound and filled once a bundle is complete");
                let chunk = chunk_in(authoritative.meta(), self.route);

                let bundle = std::mem::replace(&mut self.pending, vec![None; self.slots]);

                let Some(current) = *current else {
                    self.held = Some(HeldBundle { unit: chunk, bundle });
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

                self.held = Some(HeldBundle { unit: chunk, bundle });

                Ok(vec![Routed {
                    worker: current.worker,
                    item: WorkItem::EndOfChunk(current.chunk),
                }])
            },
            RoutingState::ByFrame { next_unit } => {
                let unit = *next_unit;
                *next_unit += 1;

                let bundle = std::mem::replace(&mut self.pending, vec![None; self.slots]);
                self.held = Some(HeldBundle { unit, bundle });

                Ok(Vec::new())
            },
        }
    }
}

/// Checks every bound, filled slot in `pending` carries the same frame index.
fn check_frame_indices(pending: &[Option<AnyItem>], bound: &[bool]) -> Result<(), ExchangeError> {
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

    use av_toolbox_core::testing::FixtureChunk;
    use av_toolbox_gpu::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        LayerId,
        SampleFormat,
        StreamItem,
        Subsampling,
        testing_context,
    };

    use super::*;
    use crate::tests::routes::{meta_in, route};

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        }
    }

    /// A frame at `index`, in `chunk` of layer 0.
    fn item(index: u64, chunk: u64) -> AnyItem {
        let context = testing_context();
        let meta = meta_in(index, 0, chunk);
        StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
    }

    /// A disperse routing on layer 0, the layer [item] puts its chunk in.
    fn chunked_disperse<D: Dispatch>(bound: &[bool], authority: usize, dispatch: D) -> Disperse<D> {
        Disperse::new(bound, authority, dispatch).with_route(Some(route(0)))
    }

    fn item_in_chunk(index: u64, chunk: u64) -> AnyItem {
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
        fn choose(&mut self, _unit: u64) -> Option<usize> {
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
        let mut disperse = chunked_disperse(&[true, true], 0, RoundRobin::new(2));

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
        let mut disperse = chunked_disperse(&[true, true], 0, RoundRobin::new(2));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(4));
        disperse.offer(0, item(0, 7)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");

        let second = disperse.offer(0, item(1, 7)).expect("should accept");

        assert_eq!(first.routed.worker, second[0].worker);
    }

    fn layered_item(index: u64, layer_zero: u64, layer_one: u64) -> AnyItem {
        let context = testing_context();
        let meta = meta_in(index, 0, layer_zero).with_provided(LayerId(1), FixtureChunk(layer_one));
        StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
    }

    #[test]
    fn a_disperse_routes_on_its_layer() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(2)).with_route(Some(route(1)));
        disperse.offer(0, layered_item(0, 0, 0)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let second = disperse.offer(0, layered_item(1, 0, 0)).expect("should accept");
        assert!(matches!(
            second[..],
            [Routed {
                item: WorkItem::Bundle(_),
                ..
            }]
        ));

        let third = disperse.offer(0, layered_item(2, 0, 1)).expect("should accept");
        assert!(matches!(
            third[..],
            [Routed {
                item: WorkItem::EndOfChunk(0),
                ..
            }]
        ));
        assert!(disperse.is_holding());
    }

    #[test]
    fn a_frame_without_the_routed_value_is_in_chunk_zero() {
        let mut disperse = Disperse::new(&[true], 0, RoundRobin::new(4)).with_route(Some(route(4)));
        disperse.offer(0, layered_item(0, 1, 1)).expect("should accept");
        let placed = disperse.place().expect("round robin always has a free worker");
        assert_eq!(placed.assignment.unit, 0);

        for index in 1..4 {
            let routed = disperse
                .offer(0, layered_item(index, 1, index + 1))
                .expect("should accept");

            assert!(matches!(
                routed[..],
                [Routed {
                    item: WorkItem::Bundle(_),
                    ..
                }]
            ));
        }
    }

    #[test]
    fn a_new_chunk_closes_the_last_one_and_moves_worker() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(4));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
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
        let mut disperse = chunked_disperse(&[true], 0, ScriptedChoice::new(&script, 2));

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

        assert_eq!(placed.assignment, Assignment { unit: 8, worker: 1 });
        assert_eq!(disperse.assignments().len(), 2);
        assert!(!disperse.is_holding());
    }

    #[test]
    fn the_first_chunk_can_also_defer_when_no_worker_is_free_yet() {
        let script = [None, Some(0)];
        let mut disperse = chunked_disperse(&[true], 0, ScriptedChoice::new(&script, 1));

        let ready = disperse.offer(0, item(0, 7)).expect("should accept");

        assert!(ready.is_empty());
        assert!(disperse.is_holding());

        assert!(disperse.place().is_none());
        assert!(disperse.is_holding());

        let placed = disperse
            .place()
            .expect("the policy frees a worker on its second try");

        assert_eq!(placed.assignment, Assignment { unit: 7, worker: 0 });
    }

    #[test]
    fn routing_follows_the_authority_slot_and_not_the_first_one() {
        // Slot 1 is the authority. Slot 0 carries a different chunking that must be ignored.
        let mut disperse = chunked_disperse(&[true, true], 1, RoundRobin::new(4));

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
        let mut disperse = chunked_disperse(&[false, true, true], 2, RoundRobin::new(4));

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
        let mut disperse = chunked_disperse(&[true, false, true], 0, RoundRobin::new(2));

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
        let mut disperse = chunked_disperse(&[true, false], 0, RoundRobin::new(2));

        let result = disperse.offer(1, item(0, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected an item for an unbound slot");
        };

        assert_eq!(error, ExchangeError::UnboundSlot { slot: 1 });
    }

    #[test]
    fn offering_a_slot_out_of_range_is_an_error() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));

        let result = disperse.offer(4, item(0, 0));

        let Err(error) = result else {
            panic!("the disperse should have rejected a slot out of range");
        };

        assert_eq!(error, ExchangeError::SlotOutOfRange { slot: 4, slots: 1 });
    }

    #[test]
    #[should_panic(expected = "chunk authority must be one of the disperse's slots")]
    fn an_authority_out_of_range_panics_at_construction() {
        chunked_disperse(&[true], 4, RoundRobin::new(2));
    }

    #[test]
    #[should_panic(expected = "chunk authority slot must be bound")]
    fn an_unbound_authority_panics_at_construction() {
        chunked_disperse(&[true, false], 1, RoundRobin::new(2));
    }

    #[test]
    fn a_chunk_id_that_goes_backwards_is_an_error() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 1)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let result = disperse.offer(0, item(1, 9));

        assert!(result.is_ok());
    }

    #[test]
    fn finishing_closes_the_chunk_in_flight_and_ends_every_worker() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(3));
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
        let mut disperse = chunked_disperse(&[true, true], 0, RoundRobin::new(1));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(1));
        disperse.offer(0, item(0, 7)).expect("should accept");

        assert!(disperse.is_holding());

        let finished = disperse.finish();

        let Some(Dropped::Unplaced(bundle)) = finished.dropped else {
            panic!("a bundle still waiting for a worker should have been reported as dropped");
        };
        assert!(bundle[0].is_some());
    }

    /// The chunk in flight was closed the moment the held bundle's own chunk arrived, so
    /// finishing on top of that must not close it again.
    #[test]
    fn finishing_while_holding_a_bundle_does_not_close_a_chunk_twice() {
        let mut disperse = chunked_disperse(&[true], 0, ScriptedChoice::new(&[Some(0), None], 1));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("the script hands out worker zero");

        let closing = disperse.offer(0, item(1, 8)).expect("should accept");
        assert!(matches!(closing[0].item, WorkItem::EndOfChunk(7)));
        assert!(disperse.place().is_none(), "the script has no worker free");

        let finished = disperse.finish();

        let closes = finished
            .routed
            .iter()
            .filter(|routed| matches!(routed.item, WorkItem::EndOfChunk(_)))
            .count();
        assert_eq!(closes, 0);
        assert!(matches!(finished.dropped, Some(Dropped::Unplaced(_))));
    }

    #[test]
    fn finishing_twice_is_a_no_op() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");

        let first = disperse.finish();
        let second = disperse.finish();

        assert!(!first.routed.is_empty());
        assert!(second.routed.is_empty());
        assert!(second.dropped.is_none());
    }

    #[test]
    fn offering_after_finish_is_an_error() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
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
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(2));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(1, 8)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        assert_eq!(
            disperse.assignments(),
            &[
                Assignment { unit: 7, worker: 0 },
                Assignment { unit: 8, worker: 1 },
            ]
        );
    }

    #[test]
    fn a_slot_offered_a_second_item_before_the_bundle_completes_is_an_error() {
        let mut disperse = chunked_disperse(&[true, true], 0, RoundRobin::new(2));
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

    #[test]
    fn frame_routing_gives_consecutive_bundles_to_different_workers() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(3));

        disperse.offer(0, item(0, 0)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(1, 0)).expect("should accept");
        let second = disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(2, 0)).expect("should accept");
        let third = disperse.place().expect("round robin always has a free worker");

        let workers = [first.routed.worker, second.routed.worker, third.routed.worker];
        assert_ne!(workers[0], workers[1]);
        assert_ne!(workers[1], workers[2]);
        assert_ne!(workers[0], workers[2]);
    }

    #[test]
    fn frame_routing_numbers_its_units_from_zero() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(3));

        disperse.offer(0, item(0, 40)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(1, 41)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        disperse.offer(0, item(2, 9)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let units: Vec<u64> = disperse
            .assignments()
            .iter()
            .map(|assignment| assignment.unit)
            .collect();
        assert_eq!(units, vec![0, 1, 2]);
    }

    #[test]
    fn frame_routing_ignores_a_chunk_id_that_goes_backwards() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(2));

        disperse.offer(0, item(0, 5)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let result = disperse.offer(0, item(1, 1));

        assert!(result.is_ok());
    }

    #[test]
    fn frame_routing_sends_a_unit_rather_than_a_bare_bundle() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(1));
        disperse.offer(0, item(0, 0)).expect("should accept");

        let placed = disperse.place().expect("round robin always has a free worker");

        let WorkItem::Unit(unit, bundle) = &placed.routed.item else {
            panic!("frame routing should have sent a unit rather than a bare bundle");
        };
        assert_eq!(*unit, 0);
        assert!(bundle[0].is_some());
    }

    /// A frame routed worker starts once and ends once, so finishing gives every one of them a
    /// last unit, which is the assignment whatever it flushes is delivered under.
    #[test]
    fn frame_routing_gives_every_worker_a_last_unit_to_close() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(3));
        disperse.offer(0, item(0, 0)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let finished = disperse.finish();

        let closed: Vec<(usize, u64)> = finished
            .routed
            .iter()
            .filter_map(|routed| match routed.item {
                WorkItem::EndOfChunk(unit) => Some((routed.worker, unit)),
                _ => None,
            })
            .collect();
        assert_eq!(closed, vec![(0, 1), (1, 2), (2, 3)]);

        let expected = vec![
            Assignment { unit: 1, worker: 0 },
            Assignment { unit: 2, worker: 1 },
            Assignment { unit: 3, worker: 2 },
        ];
        assert_eq!(finished.assignments, expected);

        let ends = finished
            .routed
            .iter()
            .filter(|routed| matches!(routed.item, WorkItem::EndOfStream))
            .count();
        assert_eq!(ends, 3);
        assert_eq!(finished.routed.len(), 6);
    }

    /// A worker that was never given a bundle was still told its stream started, so it has an end
    /// of its own to deliver and needs a unit to deliver it under.
    #[test]
    fn frame_routing_gives_a_worker_that_ran_nothing_a_last_unit_too() {
        let mut disperse = Disperse::by_frame(&[true], RoundRobin::new(2));

        let finished = disperse.finish();

        let expected = vec![
            Assignment { unit: 0, worker: 0 },
            Assignment { unit: 1, worker: 1 },
        ];
        assert_eq!(finished.assignments, expected);
    }

    /// Chunk routing closes the chunk in flight under the unit it was already assigned, so
    /// finishing invents no unit of its own the way frame routing does.
    #[test]
    fn chunk_routing_invents_no_unit_when_it_finishes() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(3));
        disperse.offer(0, item(0, 7)).expect("should accept");
        disperse.place().expect("round robin always has a free worker");

        let finished = disperse.finish();

        assert!(finished.assignments.is_empty());
        assert_eq!(disperse.assignments(), &[Assignment { unit: 7, worker: 0 }]);
    }

    /// Guards the split between `new` and `by_frame`: the chunk path must still behave exactly
    /// as it did before frame routing existed.
    #[test]
    fn chunk_routing_is_unchanged_by_the_frame_path() {
        let mut disperse = chunked_disperse(&[true], 0, RoundRobin::new(4));
        disperse.offer(0, item(0, 7)).expect("should accept");
        let first = disperse.place().expect("round robin always has a free worker");

        let second = disperse.offer(0, item(1, 7)).expect("should accept");
        assert_eq!(second.len(), 1);
        assert!(matches!(second[0].item, WorkItem::Bundle(_)));
        assert_eq!(first.routed.worker, second[0].worker);

        let boundary = disperse.offer(0, item(2, 8)).expect("should accept");
        let WorkItem::EndOfChunk(closed) = &boundary[0].item else {
            panic!("a new chunk should still close the one before it");
        };
        assert_eq!(*closed, 7);

        let placed = disperse.place().expect("round robin always has a free worker");
        assert_ne!(placed.routed.worker, first.routed.worker);
        assert!(matches!(placed.routed.item, WorkItem::Bundle(_)));
    }
}
