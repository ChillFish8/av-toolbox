use av_toolbox_core::{Crossing, Filter, Inputs, Mask, Provider, Sink, Source, end_meta_sink, run_meta_sink};
use av_toolbox_gpu::{AnyItem, FrameContext, Side};
use av_toolbox_graph::{NodeId, StageBox};

use crate::RunError;

/// Runs one stage against one bundle of inputs, producing what it emits.
///
/// A source ignores `slots` and produces at most one item, a mask produces exactly one, a filter
/// and a provider produce however many they return, and a sink produces nothing.
pub fn run_node(
    node: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
) -> Result<Vec<AnyItem>, RunError> {
    match stage {
        StageBox::Source(source) => run_source(node, source.as_mut(), context),
        StageBox::Filter(filter) => {
            let inputs = narrow(node, slots);
            run_filter(node, filter.as_mut(), context, inputs)
        },
        StageBox::Mask(mask) => {
            let inputs = narrow(node, slots);
            run_mask(node, mask.as_mut(), context, inputs)
        },
        StageBox::Provider(provider) => {
            let inputs = narrow(node, slots);
            run_provider(node, provider.as_mut(), context, inputs)
        },
        StageBox::Sink(sink) => {
            let inputs = narrow(node, slots);
            run_sink(node, sink.as_mut(), context, inputs)
        },
        StageBox::MetaSink(sink) => {
            run_meta_sink(sink.as_mut(), context, slots)
                .map_err(|source| RunError::Stage { node, source })?;

            Ok(Vec::new())
        },
        StageBox::HostSource(source) => run_source(node, source.as_mut(), context),
        StageBox::HostFilter(filter) => {
            let inputs = narrow(node, slots);
            run_filter(node, filter.as_mut(), context, inputs)
        },
        StageBox::HostMask(mask) => {
            let inputs = narrow(node, slots);
            run_mask(node, mask.as_mut(), context, inputs)
        },
        StageBox::HostProvider(provider) => {
            let inputs = narrow(node, slots);
            run_provider(node, provider.as_mut(), context, inputs)
        },
        StageBox::HostSink(sink) => {
            let inputs = narrow(node, slots);
            run_sink(node, sink.as_mut(), context, inputs)
        },
        StageBox::Crossing(crossing) => run_crossing(node, *crossing, context, slots),
    }
}

/// Narrows the engine's items to the ones a stage on side `S` takes.
///
/// Panics if handed an item from the other side, since a crossing should have caught that first.
fn narrow<S: Side>(node: NodeId, slots: Vec<Option<AnyItem>>) -> Inputs<S> {
    let slots = slots
        .into_iter()
        .map(|slot| {
            slot.map(|item| {
                S::from_any(item).unwrap_or_else(|_| {
                    panic!(
                        "node {node:?} runs on the {} side but was handed an item from the other side, so a crossing is missing in front of it",
                        S::NAME,
                    )
                })
            })
        })
        .collect();

    Inputs::from_slots(slots)
}

/// Signals the start of a chunk to whichever kind `stage` is.
pub fn start_node(node: NodeId, stage: &mut StageBox, context: &mut FrameContext) -> Result<(), RunError> {
    match stage {
        StageBox::Source(source) => source.start_of_stream(context),
        StageBox::Filter(filter) => filter.start_of_stream(context),
        StageBox::Mask(mask) => mask.start_of_stream(context),
        StageBox::Provider(provider) => provider.start_of_stream(context),
        StageBox::Sink(sink) => sink.start_of_stream(context),
        StageBox::MetaSink(sink) => sink.start_of_stream(context),
        StageBox::HostSource(source) => source.start_of_stream(context),
        StageBox::HostFilter(filter) => filter.start_of_stream(context),
        StageBox::HostMask(mask) => mask.start_of_stream(context),
        StageBox::HostProvider(provider) => provider.start_of_stream(context),
        StageBox::HostSink(sink) => sink.start_of_stream(context),
        StageBox::Crossing(_) => Ok(()),
    }
    .map_err(|source| RunError::Stage { node, source })
}

/// Signals the end of a chunk, returning whatever the stage flushes.
///
/// A source and a mask have no end of stream signal at all, so this never calls into them and
/// always produces nothing on their behalf.
pub fn end_node(
    node: NodeId,
    stage: &mut StageBox,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    match stage {
        StageBox::Source(_)
        | StageBox::Mask(_)
        | StageBox::HostSource(_)
        | StageBox::HostMask(_)
        | StageBox::Crossing(_) => Ok(Vec::new()),
        StageBox::Filter(filter) => end_filter(node, filter.as_mut(), context),
        StageBox::Provider(provider) => end_provider(node, provider.as_mut(), context),
        StageBox::Sink(sink) => end_sink(node, sink.as_mut(), context),
        StageBox::MetaSink(sink) => {
            end_meta_sink(sink.as_mut(), context).map_err(|source| RunError::Stage { node, source })?;

            Ok(Vec::new())
        },
        StageBox::HostFilter(filter) => end_filter(node, filter.as_mut(), context),
        StageBox::HostProvider(provider) => end_provider(node, provider.as_mut(), context),
        StageBox::HostSink(sink) => end_sink(node, sink.as_mut(), context),
    }
}

/// Moves one item across to the side the crossing leads to.
///
/// Panics if the item is already on the far side, since the graph placed the crossing wrongly.
fn run_crossing(
    node: NodeId,
    crossing: Crossing,
    context: &mut FrameContext,
    mut slots: Vec<Option<AnyItem>>,
) -> Result<Vec<AnyItem>, RunError> {
    let item = slots
        .first_mut()
        .and_then(Option::take)
        .expect("a crossing reads exactly one input, in slot 0");

    let crossed = match (crossing, item) {
        (Crossing::Upload, AnyItem::Host(item)) => {
            let loaded = context
                .load_host(&item)
                .map_err(|source| RunError::Transfer { node, source })?;

            AnyItem::Device(loaded)
        },
        (Crossing::Unload, AnyItem::Device(item)) => {
            let landed = context
                .unload_host(item)
                .map_err(|source| RunError::Transfer { node, source })?;

            AnyItem::Host(landed)
        },
        (crossing, _) => panic!(
            "node {node:?} is an {} crossing but was handed an item already on the far side, so the graph placed it wrongly",
            crossing.verb(),
        ),
    };

    Ok(vec![crossed])
}

fn emitted<S: Side>(items: impl IntoIterator<Item = S::Item>) -> Vec<AnyItem> {
    items.into_iter().map(S::into_any).collect()
}

fn run_source<S: Side>(
    node: NodeId,
    source: &mut dyn Source<S>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    let frame = source
        .next(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>(frame.map(S::frame_item)))
}

fn run_filter<S: Side>(
    node: NodeId,
    filter: &mut dyn Filter<S>,
    context: &mut FrameContext,
    inputs: Inputs<S>,
) -> Result<Vec<AnyItem>, RunError> {
    let frames = filter
        .process(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>(frames.into_iter().map(S::frame_item)))
}

fn run_mask<S: Side>(
    node: NodeId,
    mask: &mut dyn Mask<S>,
    context: &mut FrameContext,
    inputs: Inputs<S>,
) -> Result<Vec<AnyItem>, RunError> {
    let selected = mask
        .select(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>([S::mask_item(selected)]))
}

fn run_provider<S: Side>(
    node: NodeId,
    provider: &mut dyn Provider<S>,
    context: &mut FrameContext,
    inputs: Inputs<S>,
) -> Result<Vec<AnyItem>, RunError> {
    let items = provider
        .process(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>(items))
}

fn run_sink<S: Side>(
    node: NodeId,
    sink: &mut dyn Sink<S>,
    context: &mut FrameContext,
    inputs: Inputs<S>,
) -> Result<Vec<AnyItem>, RunError> {
    sink.consume(context, inputs)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(Vec::new())
}

fn end_filter<S: Side>(
    node: NodeId,
    filter: &mut dyn Filter<S>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    let flushed = filter
        .end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>(flushed.into_iter().map(S::frame_item)))
}

fn end_provider<S: Side>(
    node: NodeId,
    provider: &mut dyn Provider<S>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    let flushed = provider
        .end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(emitted::<S>(flushed))
}

fn end_sink<S: Side>(
    node: NodeId,
    sink: &mut dyn Sink<S>,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, RunError> {
    sink.end_of_stream(context)
        .map_err(|source| RunError::Stage { node, source })?;

    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};

    use av_toolbox_core::{Crossing, MetaInputs, MetaSink, StageError, StageInstance};
    use av_toolbox_gpu::{
        AnyItem,
        ColourInfo,
        Frame,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        HostFrame,
        HostSide,
        HostStreamItem,
        LayerId,
        Mask as GpuMask,
        MaskFormat,
        SampleFormat,
        StreamFormat,
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
            frame_rate: None,
            field_order: None,
        }
    }

    fn item(context: &FrameContext, index: u64, chunk: u64) -> AnyItem {
        let meta = FrameMeta::at(index).in_layer(LayerId(0), chunk);
        StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
    }

    fn mask_format() -> MaskFormat {
        MaskFormat {
            width: 64,
            height: 64,
            sample: SampleFormat::U8,
            chroma: None,
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
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

            Ok(vec![frame])
        }
    }

    struct HostPassThrough;

    impl StageInstance for HostPassThrough {}

    impl Filter<HostSide> for HostPassThrough {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs<HostSide>,
        ) -> Result<Vec<HostFrame>, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

            Ok(vec![frame])
        }
    }

    fn host_item(context: &FrameContext, index: u64) -> AnyItem {
        HostStreamItem::Frame(context.alloc_host_frame(&format(), FrameMeta::at(index)).freeze()).into()
    }

    /// A filter that keeps a copy of what it is given and flushes it again when its chunk ends.
    struct Holding {
        held: Option<Frame>,
    }

    impl StageInstance for Holding {}

    impl Filter for Holding {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
            self.held = Some(frame.clone());
            Ok(vec![frame])
        }

        fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
            Ok(self.held.take().into_iter().collect())
        }
    }

    /// Emits `copies` copies of the frame it was given.
    struct Copying {
        copies: usize,
    }

    impl StageInstance for Copying {}

    impl Filter for Copying {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
            let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
            let copies = vec![frame; self.copies];

            Ok(copies)
        }
    }

    fn missing_frame() -> StageError {
        StageError::msg("expected a frame in slot 0")
    }

    /// Holds each item back until the next one arrives, handing the last on at the end.
    #[derive(Default)]
    struct HoldOne {
        held: Option<HostStreamItem>,
    }

    impl StageInstance for HoldOne {}

    impl Provider<HostSide> for HoldOne {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs<HostSide>,
        ) -> Result<Vec<HostStreamItem>, StageError> {
            let item = inputs
                .take(0)
                .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
            let previous = self.held.replace(item);

            Ok(previous.into_iter().collect())
        }

        fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<HostStreamItem>, StageError> {
            Ok(self.held.take().into_iter().collect())
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

    /// Records the index of every item it consumes, without touching any samples.
    struct MetaRecording {
        seen: Arc<Mutex<Vec<u64>>>,
    }

    impl StageInstance for MetaRecording {}

    impl MetaSink for MetaRecording {
        fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
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
        fn process(
            &mut self,
            _context: &mut FrameContext,
            _inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
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
        let inputs = vec![Some(item(&context, 4, 0))];

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert_eq!(produced.len(), 1);
        assert_eq!(produced[0].meta().index, 4);
    }

    #[test]
    fn a_sink_produces_nothing_and_sees_what_it_was_given() {
        let context = testing_context();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = StageBox::Sink(Box::new(Recording { seen: seen.clone() }));
        let inputs = vec![Some(item(&context, 7, 0))];

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");

        assert!(produced.is_empty());
        assert_eq!(*seen.lock().expect("not poisoned"), vec![7]);
    }

    #[test]
    fn a_mask_produces_one_item_as_a_mask() {
        let context = testing_context();
        let mut stage = StageBox::Mask(Box::new(FixedMask));
        let inputs = vec![Some(item(&context, 0, 0))];

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

        let first =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), vec![]).expect("should run");
        let second =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), vec![]).expect("should run");
        let third =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), vec![]).expect("should run");

        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);
        assert!(third.is_empty());
    }

    #[test]
    fn a_stage_failure_names_the_node_it_came_from() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Failing));
        let inputs = vec![Some(item(&context, 0, 0))];

        let result = run_node(NodeId::new(3), &mut stage, &mut context_of(&context), inputs);

        let Err(error) = result else {
            panic!("the failing stage should have failed");
        };

        assert_eq!(error.node(), Some(NodeId::new(3)));
    }

    /// Holding emits its input during `process` and keeps a copy to flush again once its chunk ends.
    #[test]
    fn ending_a_filter_flushes_what_it_held() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Holding { held: None }));
        let inputs = vec![Some(item(&context, 9, 0))];

        let during =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), inputs).expect("should run");
        let flushed = end_node(NodeId::new(0), &mut stage, &mut context_of(&context)).expect("should end");

        assert_eq!(during.len(), 1);
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].meta().index, 9);
    }

    #[test]
    fn a_filter_emitting_nothing_produces_nothing() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Copying { copies: 0 }));
        let slots = vec![Some(item(&context, 3, 0))];

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), slots).expect("should run");

        assert!(produced.is_empty());
    }

    #[test]
    fn a_filter_emitting_one_frame_produces_it() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Copying { copies: 1 }));
        let slots = vec![Some(item(&context, 3, 0))];

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), slots).expect("should run");

        assert_eq!(produced.len(), 1);
        assert_eq!(produced[0].meta().index, 3);
    }

    #[test]
    fn a_filter_emitting_several_frames_produces_them_all() {
        let context = testing_context();
        let mut stage = StageBox::Filter(Box::new(Copying { copies: 3 }));
        let slots = vec![Some(item(&context, 3, 0))];

        let produced =
            run_node(NodeId::new(0), &mut stage, &mut context_of(&context), slots).expect("should run");

        assert_eq!(produced.len(), 3);
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
    #[should_panic(expected = "crossing is missing")]
    fn a_device_stage_handed_a_host_item_is_a_planner_bug() {
        let context = testing_context();
        let host = context.alloc_host_frame(&format(), FrameMeta::at(0)).freeze();
        let mut stage = StageBox::Filter(Box::new(PassThrough));

        let mut context = context_of(&context);
        let _ = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context,
            vec![Some(AnyItem::from(HostStreamItem::Frame(host)))],
        );
    }

    #[test]
    fn a_host_filter_runs_on_host_items_and_emits_host_items() {
        let context = testing_context();
        let mut stage = StageBox::HostFilter(Box::new(HostPassThrough));

        let mut run_context = context_of(&context);
        let out = run_node(
            NodeId::new(0),
            &mut stage,
            &mut run_context,
            vec![Some(host_item(&context, 3))],
        )
        .expect("runs");

        assert_eq!(out.len(), 1);
        assert!(out[0].is_host());
        assert_eq!(out[0].meta().index, 3);
    }

    #[test]
    #[should_panic(expected = "crossing is missing")]
    fn a_host_stage_handed_a_device_item_is_a_planner_bug() {
        let context = testing_context();
        let mut stage = StageBox::HostFilter(Box::new(HostPassThrough));

        let mut run_context = context_of(&context);
        let _ = run_node(
            NodeId::new(0),
            &mut stage,
            &mut run_context,
            vec![Some(item(&context, 0, 0))],
        );
    }

    #[test]
    fn a_meta_sink_is_handed_items_from_either_side_and_emits_nothing() {
        let context = testing_context();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut stage = StageBox::MetaSink(Box::new(MetaRecording { seen: seen.clone() }));
        let device_item = item(&context, 3, 0);
        let host_side_item = host_item(&context, 5);

        let from_device = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context_of(&context),
            vec![Some(device_item)],
        )
        .expect("should run");
        let from_host = run_node(
            NodeId::new(0),
            &mut stage,
            &mut context_of(&context),
            vec![Some(host_side_item)],
        )
        .expect("should run");

        assert!(from_device.is_empty());
        assert!(from_host.is_empty());
        assert_eq!(*seen.lock().expect("not poisoned"), vec![3, 5]);
    }

    #[test]
    fn an_upload_turns_a_host_item_into_a_device_item_in_the_same_place() {
        let context = testing_context();
        let mut stage = StageBox::Crossing(Crossing::Upload);
        let mut run_context = context_of(&context);

        let out = run_node(
            NodeId::new(0),
            &mut stage,
            &mut run_context,
            vec![Some(host_item(&context, 6))],
        )
        .expect("uploads");

        assert!(!out[0].is_host());
        assert_eq!(out[0].meta().index, 6);
    }

    #[test]
    fn an_unload_turns_a_device_item_into_a_host_item_in_the_same_place() {
        let context = testing_context();
        let mut stage = StageBox::Crossing(Crossing::Unload);
        let mut run_context = context_of(&context);

        let out = run_node(
            NodeId::new(0),
            &mut stage,
            &mut run_context,
            vec![Some(item(&context, 9, 2))],
        )
        .expect("unloads");

        assert!(out[0].is_host());
        assert_eq!(out[0].meta().index, 9);
        assert_eq!(out[0].meta().chunks.get(LayerId(0)), 2);
        assert_eq!(out[0].format(), StreamFormat::Frames(format()));
    }

    #[test]
    #[should_panic(expected = "already on the far side")]
    fn an_unload_handed_a_host_item_is_a_planner_bug() {
        let context = testing_context();
        let mut stage = StageBox::Crossing(Crossing::Unload);
        let mut run_context = context_of(&context);

        let _ = run_node(
            NodeId::new(0),
            &mut stage,
            &mut run_context,
            vec![Some(host_item(&context, 0))],
        );
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

    #[test]
    fn a_host_provider_runs_on_host_items_and_flushes_at_the_end() {
        let context = testing_context();
        let mut stage = StageBox::HostProvider(Box::<HoldOne>::default());
        let mut run_context = context_of(&context);

        let first = vec![Some(host_item(&context, 1))];
        let held = run_node(NodeId::new(0), &mut stage, &mut run_context, first).expect("runs");
        let second = vec![Some(host_item(&context, 2))];
        let passed = run_node(NodeId::new(0), &mut stage, &mut run_context, second).expect("runs");
        let flushed = end_node(NodeId::new(0), &mut stage, &mut run_context).expect("ends");

        assert!(held.is_empty());
        assert_eq!(passed.len(), 1);
        assert!(passed[0].is_host());
        assert_eq!(passed[0].meta().index, 1);
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].meta().index, 2);
    }
}
