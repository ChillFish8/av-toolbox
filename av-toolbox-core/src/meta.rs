use std::fmt;

use av_toolbox_gpu::{AnyItem, FrameContext, FrameMeta, StreamFormat};

use crate::{StageError, StageInstance};

/// An item a metadata-only transform holds, with its place in the stream reachable and its samples
/// not.
pub struct MetaItem {
    item: AnyItem,
}

impl MetaItem {
    pub const fn meta(&self) -> &FrameMeta {
        self.item.meta()
    }

    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        self.item.meta_mut()
    }

    pub const fn is_frame(&self) -> bool {
        self.item.is_frame()
    }

    pub const fn is_mask(&self) -> bool {
        self.item.is_mask()
    }

    pub fn format(&self) -> StreamFormat {
        self.item.format()
    }
}

impl fmt::Debug for MetaItem {
    /// Shows only the metadata, not the samples.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MetaItem")
            .field("meta", self.meta())
            .finish()
    }
}

/// What a metadata-only transform was handed this step, in the same slots [Inputs](crate::Inputs)
/// uses.
#[derive(Debug)]
pub struct MetaInputs {
    slots: Vec<Option<MetaItem>>,
}

impl MetaInputs {
    pub fn take(&mut self, index: usize) -> Option<MetaItem> {
        self.slots.get_mut(index)?.take()
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// How many slots hold an item.
    pub fn filled(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_some()).count()
    }
}

/// A transform that works only on where items sit in the stream, never on their samples.
///
/// Having no way to reach a plane is what lets it run on whichever side its input already sits on.
pub trait MetaTransform: StageInstance {
    /// What this emits for one call, which may be nothing, once, or several items if it holds
    /// some back or replays earlier ones.
    fn process(
        &mut self,
        context: &mut FrameContext,
        inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError>;

    /// What this emits once the stream has ended, which may be nothing.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
        Ok(Vec::new())
    }
}

pub fn run_meta_transform(
    transform: &mut dyn MetaTransform,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
) -> Result<Vec<AnyItem>, StageError> {
    let slots = slots
        .into_iter()
        .map(|slot| slot.map(|item| MetaItem { item }))
        .collect();

    let emitted = transform.process(context, MetaInputs { slots })?;

    Ok(emitted.into_iter().map(|meta| meta.item).collect())
}

pub fn end_meta_transform(
    transform: &mut dyn MetaTransform,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, StageError> {
    let flushed = transform.end_of_stream(context)?;

    Ok(flushed.into_iter().map(|meta| meta.item).collect())
}

/// A sink that reads only where items sit in the stream, never their samples.
///
/// Having no way to reach a plane is what lets it run on whichever side its input already sits on.
pub trait MetaSink: StageInstance {
    fn consume(&mut self, context: &mut FrameContext, inputs: MetaInputs) -> Result<(), StageError>;

    /// Called once the stream has ended.
    ///
    /// A sink that is given one chunk at a time is called at the end of every chunk rather than once
    /// at the end of the run.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        Ok(())
    }
}

pub fn run_meta_sink(
    sink: &mut dyn MetaSink,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
) -> Result<(), StageError> {
    let slots = slots
        .into_iter()
        .map(|slot| slot.map(|item| MetaItem { item }))
        .collect();

    sink.consume(context, MetaInputs { slots })
}

pub fn end_meta_sink(sink: &mut dyn MetaSink, context: &mut FrameContext) -> Result<(), StageError> {
    sink.end_of_stream(context)
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        HostStreamItem,
        SampleFormat,
        StreamItem,
        testing_context,
    };

    use super::*;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 16,
            height: 16,
            layout: FrameLayout::Grey,
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        }
    }

    struct Restamp;

    impl StageInstance for Restamp {}

    impl MetaTransform for Restamp {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: MetaInputs,
        ) -> Result<Vec<MetaItem>, StageError> {
            let mut item = inputs.take(0).expect("an item");
            item.meta_mut().chunk = 3;

            Ok(vec![item])
        }
    }

    struct Delay {
        held: Option<MetaItem>,
    }

    impl StageInstance for Delay {}

    impl MetaTransform for Delay {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: MetaInputs,
        ) -> Result<Vec<MetaItem>, StageError> {
            let arrived = inputs.take(0).expect("an item");

            Ok(self.held.replace(arrived).into_iter().collect())
        }

        fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<MetaItem>, StageError> {
            Ok(self.held.take().into_iter().collect())
        }
    }

    #[test]
    fn a_restamped_item_stays_on_the_side_it_arrived_on() {
        let mut context = testing_context();
        let device_frame = context.alloc_frame(&format(), FrameMeta::at(0));
        let device = AnyItem::from(StreamItem::Frame(device_frame));
        let host_frame = context.alloc_host_frame(&format(), FrameMeta::at(1)).freeze();
        let host = AnyItem::from(HostStreamItem::Frame(host_frame));

        let out_device = run_meta_transform(&mut Restamp, &mut context, vec![Some(device)]).expect("runs");
        let out_host = run_meta_transform(&mut Restamp, &mut context, vec![Some(host)]).expect("runs");

        assert!(!out_device[0].is_host());
        assert_eq!(out_device[0].meta().chunk, 3);
        assert!(out_host[0].is_host());
        assert_eq!(out_host[0].meta().chunk, 3);
    }

    #[test]
    fn a_meta_transform_can_hold_an_item_across_calls() {
        let mut context = testing_context();
        let mut delay = Delay { held: None };
        let first_frame = context.alloc_frame(&format(), FrameMeta::at(0));
        let first = AnyItem::from(StreamItem::Frame(first_frame));
        let second_frame = context.alloc_frame(&format(), FrameMeta::at(1));
        let second = AnyItem::from(StreamItem::Frame(second_frame));

        let after_first = run_meta_transform(&mut delay, &mut context, vec![Some(first)]).expect("runs");
        let after_second = run_meta_transform(&mut delay, &mut context, vec![Some(second)]).expect("runs");
        let flushed = end_meta_transform(&mut delay, &mut context).expect("flushes");

        assert!(after_first.is_empty());
        assert_eq!(after_second[0].meta().index, 0);
        assert_eq!(flushed[0].meta().index, 1);
    }

    struct Counting {
        seen: Vec<u64>,
        ended: bool,
    }

    impl StageInstance for Counting {}

    impl MetaSink for Counting {
        fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
            let item = inputs.take(0).expect("an item");
            self.seen.push(item.meta().index);

            Ok(())
        }

        fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
            self.ended = true;

            Ok(())
        }
    }

    #[test]
    fn a_meta_sink_is_handed_items_from_either_side() {
        let context = testing_context();
        let mut sink = Counting {
            seen: Vec::new(),
            ended: false,
        };
        let device_frame = context.alloc_frame(&format(), FrameMeta::at(0));
        let device_item = AnyItem::from(StreamItem::Frame(device_frame));
        let host_frame = context.alloc_host_frame(&format(), FrameMeta::at(1)).freeze();
        let host_item = AnyItem::Host(HostStreamItem::from(host_frame));
        let mut stage_context = testing_context();

        run_meta_sink(&mut sink, &mut stage_context, vec![Some(device_item)]).expect("consumes");
        run_meta_sink(&mut sink, &mut stage_context, vec![Some(host_item)]).expect("consumes");
        end_meta_sink(&mut sink, &mut stage_context).expect("ends");

        assert_eq!(sink.seen, vec![0, 1]);
        assert!(sink.ended);
    }

    #[test]
    fn filled_counts_only_occupied_slots() {
        let context = testing_context();
        let host_frame = context.alloc_host_frame(&format(), FrameMeta::at(1)).freeze();
        let host_item = AnyItem::Host(HostStreamItem::from(host_frame));
        let slots = vec![None, Some(MetaItem { item: host_item })];

        let inputs = MetaInputs { slots };

        assert_eq!(inputs.filled(), 1);
    }
}
