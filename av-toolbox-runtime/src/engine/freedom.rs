use std::sync::{Arc, Condvar, Mutex};

use crate::{Cancel, Dispatch};

/// Tracks how many units each of a region's workers currently holds, so a disperse can ask for
/// one without blocking and a caller can wait for the answer to change.
///
/// A worker is claimable while it holds fewer units than its capacity. Held means the worker's
/// aggregate has not yet drained the unit, not merely that the worker has sent it.
/// [WorkerFreedom::mark_free] is called from the aggregate thread for exactly that reason.
pub(crate) struct WorkerFreedom {
    /// How many units each worker holds that its aggregate has not drained yet.
    outstanding: Mutex<Vec<usize>>,
    capacity: usize,
    changed: Condvar,
}

impl WorkerFreedom {
    pub(crate) fn new(workers: usize, capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "a worker that may hold nothing can never be given work"
        );

        Self {
            outstanding: Mutex::new(vec![0; workers]),
            capacity,
            changed: Condvar::new(),
        }
    }

    fn workers(&self) -> usize {
        self.outstanding.lock().expect("not poisoned").len()
    }

    /// Claims the least loaded worker with room and charges it one unit, or answers `None` when
    /// every worker is full. Never blocks.
    ///
    /// Ties go to the lowest index. At capacity one this is exactly the first free worker, since
    /// every worker with room ties at zero held units. At a higher capacity it spreads claims
    /// across workers before stacking a second unit onto one that already has one.
    pub(crate) fn try_claim(&self) -> Option<usize> {
        let mut outstanding = self.outstanding.lock().expect("not poisoned");
        let worker = outstanding
            .iter()
            .enumerate()
            .filter(|&(_, held)| *held < self.capacity)
            .min_by_key(|&(_, held)| held)
            .map(|(worker, _)| worker)?;
        outstanding[worker] += 1;
        Some(worker)
    }

    pub(crate) fn mark_free(&self, worker: usize) {
        let mut outstanding = self.outstanding.lock().expect("not poisoned");
        outstanding[worker] = outstanding[worker].saturating_sub(1);
        self.changed.notify_all();
    }

    /// Wakes everyone waiting here, whether or not a worker actually freed up.
    ///
    /// Cancellation registers this, because a thread waiting on workers that have stopped has no
    /// other way out. The notify goes out under the same lock the wait holds, so a waiter that
    /// has checked its condition but not yet slept cannot miss it.
    pub(super) fn wake_everyone(&self) {
        let held = self.outstanding.lock().expect("not poisoned");
        self.changed.notify_all();
        drop(held);
    }

    /// Blocks until at least one worker has room, or the run stops.
    ///
    /// A stop ends this wait as surely as a worker freeing up does, because a thread waiting on
    /// workers that have stopped has nothing else coming. The wait re-checks its condition every
    /// time it wakes, so a `mark_free` call landing between a failed claim and this call is never
    /// lost, and neither is a stop.
    pub(super) fn wait_for_a_free_worker(&self, cancel: &Cancel) {
        let outstanding = self.outstanding.lock().expect("not poisoned");
        drop(
            self.changed
                .wait_while(outstanding, |outstanding| {
                    !cancel.is_stopped() && outstanding.iter().all(|held| *held >= self.capacity)
                })
                .expect("not poisoned"),
        );
    }
}

/// Hands a unit to whichever worker holds the fewest units its aggregate has not yet drained.
///
/// A worker stays claimable up to its capacity, so this can still hand out a unit to a worker
/// that is already holding one. Answers `None` rather than blocking when every worker is full,
/// which is exactly what [Disperse::place] needs to defer choosing without stalling the exchange
/// it drives.
///
/// [Disperse::place]: crate::Disperse::place
pub(super) struct FirstIdle {
    pub(super) freedom: Arc<WorkerFreedom>,
}

impl Dispatch for FirstIdle {
    fn choose(&mut self, _unit: u64) -> Option<usize> {
        self.freedom.try_claim()
    }

    fn workers(&self) -> usize {
        self.freedom.workers()
    }
}
