use std::fmt;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

/// How long a wait for room lasts before the budget gives up on it.
const DEFAULT_STALL_DEADLINE: Duration = Duration::from_secs(30);

/// How a wait for room ended.
#[derive(Debug, Copy, Clone)]
pub enum Room {
    /// The request fits, and the caller should charge it.
    Available,
    /// The budget was released, so the run is over and nothing more is charged.
    Released,
    /// Nothing was credited back for the whole deadline, so the wait was given up on.
    Stalled,
}

/// Called around every wait a budget makes for room.
pub trait BudgetWaits: Send + Sync {
    /// Called on the waiting thread just before it starts waiting.
    fn before_wait(&self);

    /// Called on the same thread once the wait has ended.
    fn after_wait(&self);
}

/// How much frame memory a source may have outstanding at once.
///
/// A source waits here before emitting a frame that would take it past its limit, and is woken
/// when a frame it emitted earlier returns to the pool. Nothing else waits.
pub struct FrameBudget {
    limit: usize,
    stall_deadline: Duration,
    state: Mutex<BudgetState>,
    room: Condvar,
    waits: OnceLock<Arc<dyn BudgetWaits>>,
}

impl fmt::Debug for FrameBudget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FrameBudget")
            .field("limit", &self.limit)
            .field("stall_deadline", &self.stall_deadline)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct BudgetState {
    outstanding: usize,
    released: bool,
    stalled: Option<usize>,
}

impl FrameBudget {
    /// A budget allowing up to `bytes` of frame memory outstanding at once.
    pub fn new(bytes: usize) -> Arc<Self> {
        Self::with_stall_deadline(bytes, DEFAULT_STALL_DEADLINE)
    }

    /// The same, giving up on a wait that lasts `deadline` without room appearing.
    pub fn with_stall_deadline(bytes: usize, deadline: Duration) -> Arc<Self> {
        Arc::new(Self {
            limit: bytes,
            stall_deadline: deadline,
            state: Mutex::new(BudgetState {
                outstanding: 0,
                released: false,
                stalled: None,
            }),
            room: Condvar::new(),
            waits: OnceLock::new(),
        })
    }

    /// Calls `waits` around every wait for room from now on. A second call keeps the first hook.
    pub fn set_waits(&self, waits: Arc<dyn BudgetWaits>) {
        let _ = self.waits.set(waits);
    }

    pub fn outstanding(&self) -> usize {
        self.lock().outstanding
    }

    pub const fn limit(&self) -> usize {
        self.limit
    }

    /// What was outstanding when this budget gave up waiting for room, if it ever did.
    pub fn stalled(&self) -> Option<usize> {
        self.lock().stalled
    }

    /// Waits until charging `bytes` would fit.
    ///
    /// A request larger than the whole limit is treated as having room as soon as nothing else is
    /// outstanding, since it would otherwise wait for room that can never exist. A wait lasting
    /// the whole stall deadline without the outstanding total falling is one nothing is going to
    /// end, so the budget records what it was holding, answers [Room::Stalled], and never blocks
    /// again.
    ///
    /// When it has to wait, the hook given to [FrameBudget::set_waits] is called either side of
    /// the wait with the budget unlocked.
    pub fn wait_for_room(&self, bytes: usize) -> Room {
        let state = self.lock();
        let has_to_wait = must_wait(&state, bytes, self.limit);
        let waits = self.waits.get();

        let Some(waits) = waits.filter(|_| has_to_wait) else {
            return self.wait_locked(state, bytes);
        };

        drop(state);
        waits.before_wait();

        let state = self.lock();
        let room = self.wait_locked(state, bytes);

        waits.after_wait();

        room
    }

    /// Waits for room while holding `guard`, which is dropped once the answer is known.
    fn wait_locked(&self, guard: MutexGuard<'_, BudgetState>, bytes: usize) -> Room {
        let waited = self.room.wait_timeout_while(guard, self.stall_deadline, |state| {
            must_wait(state, bytes, self.limit)
        });
        let (mut state, timing) = waited.unwrap_or_else(|error| error.into_inner());

        if state.released {
            return Room::Released;
        }

        let outstanding = state.outstanding;
        if timing.timed_out() && state.stalled.is_none() && !has_room(&state, bytes, self.limit) {
            state.stalled = Some(outstanding);
        }

        match state.stalled {
            Some(_) => Room::Stalled,
            None => Room::Available,
        }
    }

    /// Wakes every waiter permanently. A released budget never blocks again.
    pub fn release(&self) {
        self.lock().released = true;
        self.room.notify_all();
    }

    /// Adds `bytes` to what is outstanding.
    pub(crate) fn charge(&self, bytes: usize) {
        self.lock().outstanding += bytes;
    }

    /// Returns `bytes` to the budget and wakes anyone waiting for room.
    pub(crate) fn credit(&self, bytes: usize) {
        self.lock().outstanding -= bytes;
        self.room.notify_all();
    }

    fn lock(&self) -> MutexGuard<'_, BudgetState> {
        self.state.lock().unwrap_or_else(|error| error.into_inner())
    }
}

/// Whether a request for `bytes` has to wait, because the budget is still live and it does not fit.
fn must_wait(state: &BudgetState, bytes: usize, limit: usize) -> bool {
    !state.released && state.stalled.is_none() && !has_room(state, bytes, limit)
}

/// Whether `bytes` more would still fit under `limit`, given `state`.
///
/// Nothing outstanding always has room, even for a request larger than `limit`, or such a
/// request would wait forever for room that can never exist.
fn has_room(state: &BudgetState, bytes: usize, limit: usize) -> bool {
    state.outstanding == 0 || state.outstanding + bytes <= limit
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;

    use super::*;
    use crate::testing::testing_context;
    use crate::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        HostPlanes,
        LoadItem,
        MaskFormat,
        RingSpec,
        SampleFormat,
        StreamFormat,
        Subsampling,
    };

    fn frame_format() -> FrameFormat {
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

    fn mask_format() -> MaskFormat {
        MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
        }
    }

    fn has_room_now(budget: &FrameBudget, bytes: usize) -> bool {
        matches!(budget.wait_for_room(bytes), Room::Available)
    }

    #[test]
    fn an_empty_budget_has_room() {
        let budget = FrameBudget::new(1024);

        assert_eq!(budget.outstanding(), 0);
        assert!(has_room_now(&budget, 512));
    }

    #[test]
    fn a_frame_charges_the_budget_and_dropping_it_credits_back() {
        let budget = FrameBudget::new(1 << 20);
        let context = testing_context().with_budget(budget.clone());

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(0));
        let charged = budget.outstanding();
        assert!(charged > 0);

        drop(frame);

        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn loading_a_frame_charges_the_budget_and_dropping_it_credits_back() {
        let budget = FrameBudget::new(1 << 20);
        let context = testing_context().with_budget(budget.clone());
        let planes = HostPlanes::zeroed(&StreamFormat::Frames(frame_format()));
        let item = LoadItem::new(&planes, FrameMeta::at(0));

        let loaded = context.load_one(&item).expect("should load");
        let charged = budget.outstanding();
        assert!(charged > 0);

        drop(loaded);

        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn loading_a_mask_charges_the_budget_and_dropping_it_credits_back() {
        let budget = FrameBudget::new(1 << 20);
        let context = testing_context().with_budget(budget.clone());
        let planes = HostPlanes::zeroed(&StreamFormat::Mask(mask_format()));
        let item = LoadItem::new(&planes, FrameMeta::at(0));

        let loaded = context.load_one(&item).expect("should load");
        let charged = budget.outstanding();
        assert!(charged > 0);

        drop(loaded);

        assert_eq!(budget.outstanding(), 0);
    }

    #[test]
    fn a_ring_does_not_charge_the_budget() {
        let budget = FrameBudget::new(1 << 20);
        let context = testing_context().with_budget(budget.clone());

        let ring = context.ring(RingSpec {
            slots: 4,
            format: frame_format(),
        });

        assert_eq!(budget.outstanding(), 0);
        drop(ring);
    }

    #[test]
    fn scratch_does_not_charge_the_budget() {
        let budget = FrameBudget::new(1 << 20);
        let context = testing_context().with_budget(budget.clone());

        let scratch = context.scratch(4096);

        assert_eq!(budget.outstanding(), 0);
        drop(scratch);
    }

    #[test]
    fn a_context_with_no_budget_allocates_freely() {
        let context = testing_context();

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(0));

        assert!(context.budget().is_none());
        drop(frame);
    }

    #[test]
    fn releasing_a_budget_stops_it_blocking() {
        let budget = FrameBudget::new(16);
        budget.release();

        assert!(matches!(budget.wait_for_room(1024), Room::Released));
    }

    /// A request bigger than the whole limit must not wait forever once the budget is idle.
    #[test]
    fn a_request_larger_than_the_limit_still_gets_room_when_idle() {
        let budget = FrameBudget::new(16);

        assert!(has_room_now(&budget, 1024));
    }

    /// The wait runs on its own thread so this one can fail the test rather than hang with it if
    /// the deadline is ever dropped.
    #[test]
    fn a_wait_nothing_credits_gives_up_at_its_deadline() {
        let budget = FrameBudget::with_stall_deadline(64, Duration::from_millis(50));
        budget.charge(64);

        let waiting = budget.clone();
        let (answer_sender, answer_receiver) = mpsc::channel();
        thread::spawn(move || {
            answer_sender.send(waiting.wait_for_room(64)).ok();
        });

        let answer = answer_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the wait should have given up long before this");

        assert!(matches!(answer, Room::Stalled));
        assert_eq!(budget.stalled(), Some(64));
    }

    #[test]
    fn a_budget_that_gave_up_never_waits_again() {
        let budget = FrameBudget::with_stall_deadline(64, Duration::from_millis(50));
        budget.charge(64);

        let first = budget.wait_for_room(64);
        let second = budget.wait_for_room(64);

        assert!(matches!(first, Room::Stalled));
        assert!(matches!(second, Room::Stalled));
    }

    #[test]
    fn a_wait_that_gets_room_before_its_deadline_does_not_give_up() {
        let budget = FrameBudget::with_stall_deadline(64, Duration::from_secs(10));
        budget.charge(64);

        let waiting = budget.clone();
        let (answer_sender, answer_receiver) = mpsc::channel();
        thread::spawn(move || {
            answer_sender.send(waiting.wait_for_room(64)).ok();
        });

        budget.credit(64);

        let answer = answer_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("crediting the budget back should have woken the wait");

        assert!(matches!(answer, Room::Available));
        assert_eq!(budget.stalled(), None);
    }

    #[derive(Default)]
    struct CountingWaits {
        before: AtomicUsize,
        after: AtomicUsize,
    }

    impl BudgetWaits for CountingWaits {
        fn before_wait(&self) {
            self.before.fetch_add(1, Ordering::SeqCst);
        }

        fn after_wait(&self) {
            self.after.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_request_with_room_does_not_call_the_waits_hook() {
        let budget = FrameBudget::new(1024);
        let waits = Arc::new(CountingWaits::default());
        budget.set_waits(waits.clone());

        assert!(has_room_now(&budget, 512));
        assert_eq!(waits.before.load(Ordering::SeqCst), 0);
        assert_eq!(waits.after.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_wait_for_room_calls_the_waits_hook_once_either_side() {
        let budget = FrameBudget::with_stall_deadline(64, Duration::from_secs(10));
        let waits = Arc::new(CountingWaits::default());
        budget.set_waits(waits.clone());
        budget.charge(64);

        let waiting = budget.clone();
        let (answer_sender, answer_receiver) = mpsc::channel();
        thread::spawn(move || {
            answer_sender.send(waiting.wait_for_room(64)).ok();
        });

        while waits.before.load(Ordering::SeqCst) == 0 {
            thread::yield_now();
        }
        budget.credit(64);

        let answer = answer_receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("crediting the budget back should have woken the wait");

        assert!(matches!(answer, Room::Available));
        assert_eq!(waits.before.load(Ordering::SeqCst), 1);
        assert_eq!(waits.after.load(Ordering::SeqCst), 1);
    }
}
