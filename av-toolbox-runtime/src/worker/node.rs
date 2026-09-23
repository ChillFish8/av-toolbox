use av_toolbox_core::{Filter, Inputs, Mask, Sink, Source, Transform};
use av_toolbox_gpu::{FrameContext, StreamItem};
use av_toolbox_graph::{NodeId, StageBox};

use crate::RunError;

/// Runs one stage against one bundle of inputs, producing what it emits.
///
/// A source ignores `inputs` and produces at most one item, a filter and a mask each produce
/// exactly one, a transform produces however many it returns, and a sink produces nothing.
pub fn run_node(
    node: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    inputs: Inputs,
) -> Result<Vec<StreamItem>, RunError> {
    match stage {
        StageBox::Source(source) => run_source(node, source.as_mut(), context),
        StageBox::Filter(filter) => run_filter(node, filter.as_mut(), context, inputs),
        StageBox::Mask(mask) => run_mask(node, mask.as_mut(), context, inputs),
        StageBox::Transform(transform) => run_transform(node, transform.as_mut(), context, inputs),
        StageBox::Sink(sink) => run_sink(node, sink.as_mut(), context, inputs),
    }
}

/// Signals the start of a chunk to whichever of the five kinds `stage` is.
pub fn start_node(node: NodeId, stage: &mut StageBox, context: &mut FrameContext) -> Result<(), RunError> {
    let result = match stage {
        StageBox::Source(source) => source.start_of_stream(context),
        StageBox::Filter(filter) => filter.start_of_stream(context),
        StageBox::Mask(mask) => mask.start_of_stream(context),
        StageBox::Transform(transform) => transform.start_of_stream(context),
        StageBox::Sink(sink) => sink.start_of_stream(context),
    };

    result.map_err(|source| RunError::Stage { node, source })
}

/// Signals the end of a chunk, returning whatever the stage flushes.
///
/// A source and a mask have no end of stream signal at all, so this never calls into them and
/// always produces nothing on their behalf.
pub fn end_node(
    node: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    match stage {
        StageBox::Source(_) | StageBox::Mask(_) => Ok(Vec::new()),
        StageBox::Filter(filter) => end_filter(node, filter.as_mut(), context),
        StageBox::Transform(transform) => end_transform(node, transform.as_mut(), context),
        StageBox::Sink(sink) => end_sink(node, sink.as_mut(), context),
    }
}

fn run_source(
    node: NodeId,
    source: &mut dyn Source,
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    let frame = source
        .next(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(frame.into_iter().map(StreamItem::Frame).collect())
}

fn run_filter(
    node: NodeId,
    filter: &mut dyn Filter,
    context: &mut FrameContext,
    inputs: Inputs,
) -> Result<Vec<StreamItem>, RunError> {
    let frame = filter
        .process(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(vec![StreamItem::Frame(frame)])
}

fn run_mask(
    node: NodeId,
    mask: &mut dyn Mask,
    context: &mut FrameContext,
    inputs: Inputs,
) -> Result<Vec<StreamItem>, RunError> {
    let selected = mask
        .select(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(vec![StreamItem::Mask(selected)])
}

fn run_transform(
    node: NodeId,
    transform: &mut dyn Transform,
    context: &mut FrameContext,
    inputs: Inputs,
) -> Result<Vec<StreamItem>, RunError> {
    transform
        .process(context, inputs)
        .map_err(|source| RunError::Stage { node, source })
}

fn run_sink(
    node: NodeId,
    sink: &mut dyn Sink,
    context: &mut FrameContext,
    inputs: Inputs,
) -> Result<Vec<StreamItem>, RunError> {
    sink.consume(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(Vec::new())
}

fn end_filter(
    node: NodeId,
    filter: &mut dyn Filter,
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    let flushed = filter
        .end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(flushed.into_iter().map(StreamItem::Frame).collect())
}

fn end_transform(
    node: NodeId,
    transform: &mut dyn Transform,
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    transform
        .end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })
}

fn end_sink(
    node: NodeId,
    sink: &mut dyn Sink,
    context: &mut FrameContext,
) -> Result<Vec<StreamItem>, RunError> {
    sink.end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use av_toolbox_core::{StageError, StageInstance};
    use av_toolbox_gpu::{
        ColourInfo,
        Frame,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        Mask as GpuMask,
        MaskFormat,
        SampleFormat,
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

    fn item(context: &FrameContext, index: u64, chunk: u64) -> StreamItem {
        let meta = FrameMeta::at(index).in_chunk(chunk);
        StreamItem::Frame(context.alloc_frame(&format(), meta))
    }

    fn mask_format() -> MaskFormat {
        MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
        }
    }

    /// A fresh context for a call, since `FrameContext` cannot be cloned or shared mutably
    /// across the several calls one test makes.
    fn context_of(_seed: &FrameContext) -> FrameContext {
        testing_context()
    }

    struct PassThrough;

    impl StageInstance for PassThrough {}

    impl Filter for PassThrough {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
        }
    }

    /// A filter that keeps a copy of what it is given and flushes it again when its chunk ends.
    struct Holding {
        held: Option<Frame>,
    }

    impl StageInstance for Holding {}

    impl Filter for Holding {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
            self.held = Some(frame.clone());
            Ok(frame)
        }

        fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
            Ok(self.held.take().into_iter().collect())
        }
    }

    /// Emits a fixed number of copies of what it was given.
    struct Multiply {
        copies: usize,
    }

    impl StageInstance for Multiply {}

    impl Transform for Multiply {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<StreamItem>, StageError> {
            let item = inputs
                .take(0)
                .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

            Ok((0..self.copies).map(|_| item.clone()).collect())
        }
    }

    /// Records what it was asked to consume.
    struct Recording {
        seen: Arc<Mutex<Vec<u64>>>,
    }

    impl StageInstance for Recording {}

    impl Sink for Recording {
        fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
            let mut seen = self.seen.lock().expect("not poisoned");

            for slot in 0..inputs.len() {
                if let Some(item) = inputs.take(slot) {
                    seen.push(item.meta().index);
                }
            }

            Ok(())
        }
    }

    /// Yields a fixed list of frames, then ends.
    struct Listed {
        remaining: VecDeque<Frame>,
    }

    impl StageInstance for Listed {}

    impl Source for Listed {
        fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
            Ok(self.remaining.pop_front())
        }
    }

    /// Produces a fixed mask.
    struct FixedMask;

    impl StageInstance for FixedMask {}

    impl Mask for FixedMask {
        fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
            Ok(context.alloc_mask(&mask_format(), FrameMeta::at(2)))
        }
    }

    struct Failing;

    impl StageInstance for Failing {}

    impl Filter for Failing {
        fn process(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<Frame, StageError> {
            Err(StageError::msg("this filter always fails"))
        }
    }

    /// Records whether it was started.
    struct Starting {
        started: Arc<AtomicBool>,
    }

    impl StageInstance for Starting {
        fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
            self.started.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    impl Sink for Starting {
        fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
            Ok(())
        }
    }

    #[test]
    fn a_filter_produces_one_item_from_its_inputs() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(PassThrough));
        let inputs = Inputs::new(vec![Some(item(&context, 4, 0))]);

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert_eq!(produced.len(), 1);
        assert_eq!(produced[0].meta().index, 4);
    }

    #[test]
    fn a_transform_produces_what_it_returns() {
        let context = testing_context();
        let mut stage = StageBox::Transform(Box::new(Multiply { copies: 3 }));
        let inputs = Inputs::new(vec![Some(item(&context, 0, 0))]);

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert_eq!(produced.len(), 3);
    }

    #[test]
    fn a_transform_producing_nothing_produces_nothing() {
        let context = testing_context();
        let mut stage = StageBox::Transform(Box::new(Multiply { copies: 0 }));
        let inputs = Inputs::new(vec![Some(item(&context, 0, 0))]);

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert!(produced.is_empty());
    }

    #[test]
    fn a_sink_produces_nothing_and_sees_what_it_was_given() {
        let context = testing_context();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = StageBox::Sink(Box::new(Recording { seen: seen.clone() }));
        let inputs = Inputs::new(vec![Some(item(&context, 7, 0))]);

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert!(produced.is_empty());
        assert_eq!(*seen.lock().expect("not poisoned"), vec![7]);
    }

    #[test]
    fn a_mask_produces_one_item_as_a_mask() {
        let context = testing_context();
        let mut stage = StageBox::Mask(Box::new(FixedMask));
        let inputs = Inputs::new(vec![Some(item(&context, 0, 0))]);

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert_eq!(produced.len(), 1);
        assert!(produced[0].is_mask());
    }

    #[test]
    fn a_source_produces_until_it_ends() {
        let context = testing_context();
        let frames = VecDeque::from(vec![
            context.alloc_frame(&format(), FrameMeta::at(0)),
            context.alloc_frame(&format(), FrameMeta::at(1)),
        ]);
        let mut stage = StageBox::Source(Box::new(Listed { remaining: frames }));

        let first = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context_of(&context),
            Inputs::new(vec![]),
        )
        .expect("should run");
        let second = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context_of(&context),
            Inputs::new(vec![]),
        )
        .expect("should run");
        let third = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context_of(&context),
            Inputs::new(vec![]),
        )
        .expect("should run");

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert!(third.is_empty());
    }

    #[test]
    fn a_stage_failure_names_the_node_it_came_from() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Failing));
        let inputs = Inputs::new(vec![Some(item(&context, 0, 0))]);

        let result = run_node(NodeId::new(3), &mut stage, &mut context_of(&context), inputs);

        let Err(error) = result else {
            panic!("the failing stage should have failed");
        };

        assert_eq!(error.node(), Some(NodeId::new(3)));
    }

    /// A filter's `process` always returns exactly one item, since it is declared as
    /// `Result<Frame, StageError>` rather than an optional one. Holding cannot withhold its
    /// input during `process`, only keep a copy to flush again once its chunk ends.
    #[test]
    fn ending_a_filter_flushes_what_it_held() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Holding { held: None }));
        let inputs = Inputs::new(vec![Some(item(&context, 9, 0))]);

        let during =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");
        let flushed = end_node(NodeId::new(0), &mut stage, &mut context_of(&context)).expect("should end");

        assert_eq!(during.len(), 1);
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].meta().index, 9);
    }

    #[test]
    fn ending_a_source_produces_nothing_and_does_not_fail() {
        let context = testing_context();
        let mut stage = StageBox::Source(Box::new(Listed {
            remaining: VecDeque::new(),
        }));

        let flushed = end_node(NodeId::new(0), &mut stage, &mut context_of(&context)).expect("should end");

        assert!(flushed.is_empty());
    }

    #[test]
    fn ending_a_mask_produces_nothing_and_does_not_fail() {
        let context = testing_context();
        let mut stage = StageBox::Mask(Box::new(FixedMask));

        let flushed = end_node(NodeId::new(0), &mut stage, &mut context_of(&context)).expect("should end");

        assert!(flushed.is_empty());
    }

    #[test]
    fn starting_a_stage_reaches_it() {
        let context = testing_context();
        let started = Arc::new(AtomicBool::new(false));
        let mut stage = StageBox::Sink(Box::new(Starting {
            started: started.clone(),
        }));

        start_node(NodeId::new(0), &mut stage, &mut context_of(&context)).expect("should start");

        assert!(started.load(Ordering::SeqCst));
    }
}
