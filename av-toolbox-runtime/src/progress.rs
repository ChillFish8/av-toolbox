use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use av_toolbox_graph::NodeId;

/// A live view of a run, readable from any thread while it goes.
#[derive(Debug, Clone)]
pub struct Progress {
    state: Arc<ProgressState>,
}

#[derive(Debug)]
struct ProgressState {
    started: OnceLock<Instant>,
    read: Arc<AtomicU64>,
    sinks: Vec<(NodeId, Arc<AtomicU64>)>,
    first_output: FirstOutput,
}

/// An action waiting for the first output of a run.
#[derive(Default)]
struct FirstOutput(Mutex<Option<Box<dyn FnOnce() + Send>>>);

impl FirstOutput {
    fn set(&self, action: Box<dyn FnOnce() + Send>) {
        let mut slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let replaced = slot.replace(action);
        drop(slot);

        drop(replaced);
    }

    /// Takes the action out, so it is run or dropped without the lock held.
    fn take(&self) -> Option<Box<dyn FnOnce() + Send>> {
        let mut slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
        slot.take()
    }
}

impl fmt::Debug for FirstOutput {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slot = self.0.lock().unwrap_or_else(|error| error.into_inner());
        let pending = slot.is_some();
        drop(slot);

        formatter
            .debug_struct("FirstOutput")
            .field("pending", &pending)
            .finish()
    }
}

/// How many steps one sink has run so far.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SinkProgress {
    pub node: NodeId,
    pub steps: u64,
}

impl Progress {
    pub(crate) fn new(sinks: impl IntoIterator<Item = NodeId>) -> Self {
        let sinks = sinks
            .into_iter()
            .map(|node| (node, Arc::new(AtomicU64::new(0))))
            .collect();
        let state = ProgressState {
            started: OnceLock::new(),
            read: Arc::new(AtomicU64::new(0)),
            sinks,
            first_output: FirstOutput::default(),
        };

        Self {
            state: Arc::new(state),
        }
    }

    /// Marks the moment the run started. Only the first call counts.
    pub(crate) fn start(&self) {
        let _ = self.state.started.set(Instant::now());
    }

    /// The counter every source adds its frames to.
    pub(crate) fn read_counter(&self) -> Arc<AtomicU64> {
        self.state.read.clone()
    }

    /// The counter the sink at `node` adds its steps to.
    pub(crate) fn sink_counter(&self, node: NodeId) -> Arc<AtomicU64> {
        self.state
            .sinks
            .iter()
            .find(|(id, _)| *id == node)
            .map(|(_, counter)| counter.clone())
            .expect("every sink in the graph has a counter")
    }

    /// Runs `action` on the sink's thread when the first sink step of the run has consumed a frame.
    ///
    /// It runs at most once, and replaces any action set before. When the run ends without output it
    /// is dropped without running.
    pub fn on_first_output(&self, action: impl FnOnce() + Send + 'static) {
        let boxed = Box::new(action);
        self.state.first_output.set(boxed);
    }

    /// Runs the first-output action, if it is still waiting.
    pub(crate) fn output_reached(&self) {
        let action = self.state.first_output.take();
        if let Some(action) = action {
            action();
        }
    }

    /// Drops the first-output action without running it.
    pub(crate) fn drop_first_output(&self) {
        let action = self.state.first_output.take();
        drop(action);
    }

    /// How long the run has been going, zero before it starts.
    pub fn elapsed(&self) -> Duration {
        match self.state.started.get() {
            Some(started) => started.elapsed(),
            None => Duration::ZERO,
        }
    }

    /// How many frames every source has read so far, added together.
    pub fn read(&self) -> u64 {
        self.state.read.load(Ordering::Relaxed)
    }

    /// How many steps each sink has run so far, in node order.
    pub fn sinks(&self) -> Vec<SinkProgress> {
        self.state
            .sinks
            .iter()
            .map(|(node, counter)| SinkProgress {
                node: *node,
                steps: counter.load(Ordering::Relaxed),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    #[test]
    fn the_first_output_action_runs_once() {
        let progress = Progress::new([NodeId::new(0)]);
        let runs = Arc::new(AtomicUsize::new(0));
        let counted = runs.clone();

        progress.on_first_output(move || {
            counted.fetch_add(1, Ordering::SeqCst);
        });
        progress.output_reached();
        progress.output_reached();

        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn an_unused_first_output_action_is_dropped() {
        let progress = Progress::new([NodeId::new(0)]);
        let held = Arc::new(());
        let captured = held.clone();

        progress.on_first_output(move || drop(captured));
        assert_eq!(Arc::strong_count(&held), 2);

        drop(progress);
        assert_eq!(Arc::strong_count(&held), 1);
    }

    #[test]
    fn dropping_the_first_output_action_leaves_clones_without_it() {
        let progress = Progress::new([NodeId::new(0)]);
        let clone = progress.clone();
        let held = Arc::new(());
        let captured = held.clone();

        progress.on_first_output(move || drop(captured));
        progress.drop_first_output();

        assert_eq!(Arc::strong_count(&held), 1);
        drop(clone);
    }
}
