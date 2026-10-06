use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
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
