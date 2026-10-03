use std::cell::Cell;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

use av_toolbox_gpu::{AnyItem, BudgetWaits, DeviceSync, FrameContext, RuntimeError};
use av_toolbox_graph::NodeId;

use crate::RunError;

thread_local! {
    /// When this thread gave up the profiling lock to wait for budget room.
    static WAIT_STARTED: Cell<Option<Instant>> = const { Cell::new(None) };
    /// How long this thread's current call has spent waiting for budget room.
    static WAITED: Cell<Duration> = const { Cell::new(Duration::ZERO) };
}

/// Times every stage call of a run.
#[derive(Clone)]
pub struct Profiling {
    /// Blocks until the calling thread's device work has finished.
    pub sync: DeviceSync,
    /// How many frames each node produces before its time is counted.
    pub warmup: u64,
}

impl fmt::Debug for Profiling {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Profiling")
            .field("warmup", &self.warmup)
            .finish_non_exhaustive()
    }
}

/// One node's counted calls and the time they took.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodeTiming {
    pub node: NodeId,
    /// Frames counted after the warmup.
    pub frames: u64,
    /// Time spent inside the stage's own calls.
    pub host: Duration,
    /// Time from each call starting until its device work had finished.
    pub busy: Duration,
}

impl NodeTiming {
    /// The time spent waiting on the device after the stage's own calls returned.
    pub fn device(&self) -> Duration {
        self.busy.saturating_sub(self.host)
    }
}

/// Whether a timed call counts as one of its node's frames.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Counted {
    /// The call always counts as a frame.
    Frame,
    /// The call counts as a frame only when it produced something.
    FrameIfProduced,
    /// The call never counts as a frame.
    Never,
}

impl Counted {
    fn counts(self, produced: &[AnyItem]) -> bool {
        match self {
            Counted::Frame => true,
            Counted::FrameIfProduced => !produced.is_empty(),
            Counted::Never => false,
        }
    }
}

#[derive(Default)]
struct Totals {
    warmed: u64,
    frames: u64,
    host: Duration,
    busy: Duration,
}

/// A lock that records which thread holds it, so that thread can hand it over and take it back.
struct HandoffLock {
    holder: Mutex<Option<ThreadId>>,
    freed: Condvar,
}

impl HandoffLock {
    fn new() -> Self {
        Self {
            holder: Mutex::new(None),
            freed: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let current = thread::current().id();
        let holder = self.holder();
        let waited = self.freed.wait_while(holder, |holder| holder.is_some());
        let mut holder = waited.unwrap_or_else(PoisonError::into_inner);

        *holder = Some(current);
    }

    /// Frees the lock when the current thread holds it, waking one waiter.
    fn release(&self) {
        let current = thread::current().id();
        let mut holder = self.holder();
        if *holder != Some(current) {
            return;
        }

        *holder = None;
        drop(holder);

        self.freed.notify_one();
    }

    fn held_by_current(&self) -> bool {
        let current = thread::current().id();

        *self.holder() == Some(current)
    }

    /// Acquires the lock until the returned guard is dropped.
    fn hold(&self) -> HeldLock<'_> {
        self.acquire();

        HeldLock { lock: self }
    }

    fn holder(&self) -> MutexGuard<'_, Option<ThreadId>> {
        self.holder.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

struct HeldLock<'a> {
    lock: &'a HandoffLock,
}

impl Drop for HeldLock<'_> {
    fn drop(&mut self) {
        self.lock.release();
    }
}

pub(crate) struct Profiler {
    profiling: Profiling,
    lock: HandoffLock,
    totals: Mutex<HashMap<NodeId, Totals>>,
    wait_error: Mutex<Option<(ThreadId, RuntimeError)>>,
}

impl Profiler {
    pub(crate) fn new(profiling: Profiling) -> Self {
        Self {
            profiling,
            lock: HandoffLock::new(),
            totals: Mutex::new(HashMap::new()),
            wait_error: Mutex::new(None),
        }
    }

    /// Runs `call` for `node` under the lock, waits for its device work, and records both times.
    ///
    /// Time the call spends waiting for budget room is left out of both.
    pub(crate) fn time<F>(
        &self,
        node: NodeId,
        counted: Counted,
        context: &mut FrameContext,
        call: F,
    ) -> Result<Vec<AnyItem>, RunError>
    where
        F: FnOnce(&mut FrameContext) -> Result<Vec<AnyItem>, RunError>,
    {
        let held = self.lock.hold();
        WAITED.set(Duration::ZERO);

        let started = Instant::now();
        let result = call(context);
        let host = started.elapsed();

        context.flush();
        let synced = (self.profiling.sync)();
        let busy = started.elapsed();

        drop(held);

        let wait_error = self.take_wait_error();
        let produced = result?;
        if let Err(source) = synced {
            return Err(RunError::Sync { node, source });
        }
        if let Some(source) = wait_error {
            return Err(RunError::Sync { node, source });
        }

        let waited = WAITED.get();
        let host = host.saturating_sub(waited);
        let busy = busy.saturating_sub(waited);
        let counts_frame = counted.counts(&produced);
        self.record(node, counts_frame, host, busy);

        Ok(produced)
    }

    fn record(&self, node: NodeId, counts_frame: bool, host: Duration, busy: Duration) {
        let mut totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);
        let entry = totals.entry(node).or_default();

        if entry.warmed < self.profiling.warmup {
            if counts_frame {
                entry.warmed += 1;
            }
            return;
        }

        if counts_frame {
            entry.frames += 1;
        }
        entry.host += host;
        entry.busy += busy;
    }

    /// Keeps the first failed sync, for the call on this thread to return once it ends.
    fn keep_wait_error(&self, source: RuntimeError) {
        let current = thread::current().id();
        let mut kept = self.wait_error.lock().unwrap_or_else(PoisonError::into_inner);

        if kept.is_none() {
            *kept = Some((current, source));
        }
    }

    /// The failed sync kept for this thread's call, if there is one.
    fn take_wait_error(&self) -> Option<RuntimeError> {
        let current = thread::current().id();
        let mut kept = self.wait_error.lock().unwrap_or_else(PoisonError::into_inner);

        let kept_here = matches!(&*kept, Some((thread, _)) if *thread == current);
        if !kept_here {
            return None;
        }

        kept.take().map(|(_, source)| source)
    }

    /// Every timed node's counted calls, in node order.
    pub(crate) fn timings(&self) -> Vec<NodeTiming> {
        let totals = self.totals.lock().unwrap_or_else(PoisonError::into_inner);

        let mut timings: Vec<NodeTiming> = totals
            .iter()
            .map(|(node, totals)| NodeTiming {
                node: *node,
                frames: totals.frames,
                host: totals.host,
                busy: totals.busy,
            })
            .collect();
        timings.sort_by_key(|timing| timing.node.index());

        timings
    }
}

/// Hands the lock over while a call waits for budget room, so the stages that free that room can run.
impl BudgetWaits for Profiler {
    /// Syncs the device for the work queued so far, then gives up the lock if this thread holds it.
    fn before_wait(&self) {
        if !self.lock.held_by_current() {
            return;
        }

        let synced = (self.profiling.sync)();
        if let Err(source) = synced {
            self.keep_wait_error(source);
        }

        WAIT_STARTED.set(Some(Instant::now()));
        self.lock.release();
    }

    /// Takes the lock back after a wait [BudgetWaits::before_wait] gave it up for.
    fn after_wait(&self) {
        let Some(started) = WAIT_STARTED.take() else {
            return;
        };

        self.lock.acquire();

        let waited = started.elapsed();
        let total = WAITED.get() + waited;
        WAITED.set(total);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
    use std::time::Duration;

    use av_toolbox_core::StageError;
    use av_toolbox_gpu::{DeviceSync, RuntimeError, testing_context};
    use av_toolbox_graph::NodeId;

    use super::*;

    fn counting_sync(count: Arc<AtomicUsize>) -> DeviceSync {
        Arc::new(move || {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
    }

    fn sleeping_sync() -> DeviceSync {
        Arc::new(|| {
            thread::sleep(Duration::from_millis(5));
            Ok(())
        })
    }

    fn failing_sync() -> DeviceSync {
        Arc::new(|| {
            Err(RuntimeError::ReadShape {
                expected: 1,
                returned: 0,
            })
        })
    }

    #[test]
    fn every_call_syncs_and_the_warmup_is_not_counted() {
        let syncs = Arc::new(AtomicUsize::new(0));
        let sync = counting_sync(syncs.clone());
        let profiler = Profiler::new(Profiling { sync, warmup: 2 });
        let mut context = testing_context();
        let node = NodeId::new(4);

        for _ in 0..5 {
            profiler
                .time(node, Counted::Frame, &mut context, |_context| Ok(Vec::new()))
                .expect("runs");
        }

        let timings = profiler.timings();
        assert_eq!(syncs.load(Ordering::SeqCst), 5);
        assert_eq!(timings.len(), 1);
        assert_eq!(timings[0].node, node);
        assert_eq!(timings[0].frames, 3);
    }

    #[test]
    fn the_device_time_covers_the_wait_for_the_device() {
        let profiler = Profiler::new(Profiling {
            sync: sleeping_sync(),
            warmup: 0,
        });
        let mut context = testing_context();

        profiler
            .time(NodeId::new(0), Counted::Frame, &mut context, |_context| {
                Ok(Vec::new())
            })
            .expect("runs");

        let timings = profiler.timings();
        assert!(timings[0].device() >= Duration::from_millis(5));
    }

    #[test]
    fn a_failed_sync_names_the_node() {
        let profiler = Profiler::new(Profiling {
            sync: failing_sync(),
            warmup: 0,
        });
        let mut context = testing_context();
        let node = NodeId::new(2);

        let result = profiler.time(node, Counted::Frame, &mut context, |_context| Ok(Vec::new()));

        let error = result.expect_err("the sync fails");
        assert_eq!(error.node(), Some(node));
    }

    #[test]
    fn a_failed_call_releases_the_lock() {
        let profiler = Profiler::new(Profiling {
            sync: Arc::new(|| Ok(())),
            warmup: 0,
        });
        let mut context = testing_context();
        let node = NodeId::new(1);

        let failed = profiler.time(node, Counted::Frame, &mut context, |_context| {
            Err(RunError::Stage {
                node,
                source: StageError::msg("broken"),
            })
        });
        let next = profiler.time(node, Counted::Frame, &mut context, |_context| Ok(Vec::new()));

        assert!(failed.is_err());
        assert!(next.is_ok());
    }

    #[test]
    fn time_spent_waiting_for_room_is_not_counted() {
        let profiler = Profiler::new(Profiling {
            sync: Arc::new(|| Ok(())),
            warmup: 0,
        });
        let mut context = testing_context();

        profiler
            .time(NodeId::new(0), Counted::Frame, &mut context, |_context| {
                profiler.before_wait();
                assert!(!profiler.lock.held_by_current());
                thread::sleep(Duration::from_millis(50));
                profiler.after_wait();
                Ok(Vec::new())
            })
            .expect("runs");

        let timings = profiler.timings();
        assert_eq!(timings[0].frames, 1);
        assert!(timings[0].host < Duration::from_millis(50));
        assert!(timings[0].busy < Duration::from_millis(50));
    }

    #[test]
    fn a_call_never_counted_adds_time_but_no_frame() {
        let profiler = Profiler::new(Profiling {
            sync: Arc::new(|| Ok(())),
            warmup: 0,
        });
        let mut context = testing_context();

        profiler
            .time(NodeId::new(3), Counted::Never, &mut context, |_context| {
                thread::sleep(Duration::from_millis(5));
                Ok(Vec::new())
            })
            .expect("runs");

        let timings = profiler.timings();
        assert_eq!(timings[0].frames, 0);
        assert!(timings[0].host >= Duration::from_millis(5));
    }

    #[test]
    fn a_failed_sync_while_waiting_fails_the_call() {
        let profiler = Profiler::new(Profiling {
            sync: failing_sync(),
            warmup: 0,
        });
        let mut context = testing_context();
        let node = NodeId::new(5);

        let result = profiler.time(node, Counted::Frame, &mut context, |_context| {
            profiler.before_wait();
            profiler.after_wait();
            Ok(Vec::new())
        });

        let error = result.expect_err("the sync fails");
        assert_eq!(error.node(), Some(node));
        assert!(profiler.take_wait_error().is_none());
    }
}
