use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use av_toolbox_core::{Inputs, MetaInputs, MetaSink, Sink, Source, StageError, StageInstance};
use av_toolbox_gpu::{FrameContext, Side};
use av_toolbox_graph::{NodeId, StageBox};

/// Wraps a source so the engine can tell, from outside, when it has produced its last frame.
///
/// `Source::next` only ever returns `None` once, per its own contract, so storing that once is
/// enough to know the source will never be called again.
pub(super) struct WatchedSource<S: Side> {
    inner: Box<dyn Source<S>>,
    exhausted: Arc<AtomicBool>,
    limit: Option<u64>,
    produced: u64,
}

impl<S: Side> StageInstance for WatchedSource<S> {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl<S: Side> Source<S> for WatchedSource<S> {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<S::Frame>, StageError> {
        let reached = self.limit.is_some_and(|limit| self.produced >= limit);
        if reached {
            self.exhausted.store(true, Ordering::SeqCst);
            return Ok(None);
        }

        let frame = self.inner.next(context)?;
        match frame {
            Some(_) => self.produced += 1,
            None => self.exhausted.store(true, Ordering::SeqCst),
        }

        Ok(frame)
    }
}

/// What reached the sinks, and when the counted part of the run began.
pub(super) struct SinkTally {
    frames: AtomicU64,
    warmup: u64,
    counted_from: OnceLock<Instant>,
}

impl SinkTally {
    pub(super) fn new(warmup: u64) -> Self {
        Self {
            frames: AtomicU64::new(0),
            warmup,
            counted_from: OnceLock::new(),
        }
    }

    /// Adds `seen` items, marking the moment the warmup is passed.
    fn add(&self, seen: u64) {
        let before = self.frames.fetch_add(seen, Ordering::SeqCst);
        let after = before + seen;

        let crossed = before < self.warmup && after >= self.warmup;
        if crossed {
            let _ = self.counted_from.set(Instant::now());
        }
    }

    pub(super) fn frames(&self) -> u64 {
        self.frames.load(Ordering::SeqCst)
    }

    /// How long the run took after its warmup, given when it started and ended.
    pub(super) fn counted_wall(&self, started: Instant, ended: Instant) -> Duration {
        if self.warmup == 0 {
            return ended.duration_since(started);
        }

        match self.counted_from.get() {
            Some(from) => ended.duration_since(*from),
            None => Duration::ZERO,
        }
    }
}

/// Wraps a sink so the engine can count what reached it, for
/// [RunReport::frames](crate::engine::RunReport::frames).
struct CountingSink<S: Side> {
    inner: Box<dyn Sink<S>>,
    tally: Arc<SinkTally>,
}

impl<S: Side> StageInstance for CountingSink<S> {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl<S: Side> Sink<S> for CountingSink<S> {
    fn consume(&mut self, context: &mut FrameContext, mut inputs: Inputs<S>) -> Result<(), StageError> {
        let mut slots = Vec::with_capacity(inputs.len());
        let mut seen = 0u64;

        for slot in 0..inputs.len() {
            let item = inputs.take(slot);
            if item.is_some() {
                seen += 1;
            }
            slots.push(item);
        }

        self.tally.add(seen);
        self.inner.consume(context, Inputs::from_slots(slots))
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Wraps a metadata-only sink so the engine can count what reached it.
struct CountingMetaSink {
    inner: Box<dyn MetaSink>,
    tally: Arc<SinkTally>,
}

impl StageInstance for CountingMetaSink {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl MetaSink for CountingMetaSink {
    fn consume(&mut self, context: &mut FrameContext, inputs: MetaInputs) -> Result<(), StageError> {
        let seen = inputs.filled() as u64;
        self.tally.add(seen);

        self.inner.consume(context, inputs)
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Replaces the region head's source with a watched one, leaving every other stage alone.
pub(super) fn wrap_head_source(
    nodes: Vec<(NodeId, StageBox)>,
    exhausted: Arc<AtomicBool>,
    limit: Option<u64>,
) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (id, stage))| {
            if index == 0 {
                match stage {
                    StageBox::Source(inner) => {
                        let exhausted = exhausted.clone();
                        return (
                            id,
                            StageBox::Source(Box::new(WatchedSource {
                                inner,
                                exhausted,
                                limit,
                                produced: 0,
                            })),
                        );
                    },
                    StageBox::HostSource(inner) => {
                        let exhausted = exhausted.clone();
                        return (
                            id,
                            StageBox::HostSource(Box::new(WatchedSource {
                                inner,
                                exhausted,
                                limit,
                                produced: 0,
                            })),
                        );
                    },
                    other => return (id, other),
                }
            }

            (id, stage)
        })
        .collect()
}

/// Replaces every sink in a region's stages with a counting one.
pub(super) fn wrap_sinks(nodes: Vec<(NodeId, StageBox)>, tally: Arc<SinkTally>) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .map(|(id, stage)| match stage {
            StageBox::Sink(inner) => (
                id,
                StageBox::Sink(Box::new(CountingSink {
                    inner,
                    tally: tally.clone(),
                })),
            ),
            StageBox::HostSink(inner) => (
                id,
                StageBox::HostSink(Box::new(CountingSink {
                    inner,
                    tally: tally.clone(),
                })),
            ),
            StageBox::MetaSink(inner) => (
                id,
                StageBox::MetaSink(Box::new(CountingMetaSink {
                    inner,
                    tally: tally.clone(),
                })),
            ),
            other => (id, other),
        })
        .collect()
}
