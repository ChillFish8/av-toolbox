use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use av_toolbox_core::{Inputs, Sink, Source, StageError, StageInstance};
use av_toolbox_gpu::{Frame, FrameContext};
use av_toolbox_graph::{NodeId, StageBox};

/// Wraps a source so the engine can tell, from outside, when it has produced its last frame.
///
/// `Source::next` only ever returns `None` once, per its own contract, so storing that once is
/// enough to know the source will never be called again.
pub(super) struct WatchedSource {
    inner: Box<dyn Source>,
    exhausted: Arc<AtomicBool>,
}

impl StageInstance for WatchedSource {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl Source for WatchedSource {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        let frame = self.inner.next(context)?;
        if frame.is_none() {
            self.exhausted.store(true, Ordering::SeqCst);
        }

        Ok(frame)
    }
}

/// Wraps a sink so the engine can count what reached it, for [RunReport::frames].
///
/// [RunReport::frames]: super::RunReport::frames
struct CountingSink {
    inner: Box<dyn Sink>,
    frames: Arc<AtomicU64>,
}

impl StageInstance for CountingSink {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.start_of_stream(context)
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let mut slots = Vec::with_capacity(inputs.len());
        let mut seen = 0u64;

        for slot in 0..inputs.len() {
            let item = inputs.take(slot);
            if item.is_some() {
                seen += 1;
            }
            slots.push(item);
        }

        self.frames.fetch_add(seen, Ordering::SeqCst);
        self.inner.consume(context, Inputs::new(slots))
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        self.inner.end_of_stream(context)
    }
}

/// Replaces the region head's source with a watched one, leaving every other stage alone.
pub(super) fn wrap_head_source(
    nodes: Vec<(NodeId, StageBox)>,
    exhausted: Arc<AtomicBool>,
) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .enumerate()
        .map(|(index, (id, stage))| {
            if index == 0
                && let StageBox::Source(inner) = stage
            {
                return (
                    id,
                    StageBox::Source(Box::new(WatchedSource {
                        inner,
                        exhausted: exhausted.clone(),
                    })),
                );
            }

            (id, stage)
        })
        .collect()
}

/// Replaces every sink in a region's stages with a counting one.
pub(super) fn wrap_sinks(nodes: Vec<(NodeId, StageBox)>, frames: Arc<AtomicU64>) -> Vec<(NodeId, StageBox)> {
    nodes
        .into_iter()
        .map(|(id, stage)| match stage {
            StageBox::Sink(inner) => (
                id,
                StageBox::Sink(Box::new(CountingSink {
                    inner,
                    frames: frames.clone(),
                })),
            ),
            other => (id, other),
        })
        .collect()
}
