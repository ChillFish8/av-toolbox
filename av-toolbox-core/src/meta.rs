use std::fmt;

use av_toolbox_gpu::{AnyItem, FrameContext, FrameMeta, StreamFormat};

use crate::{StageError, StageInstance};

/// An item a metadata-only transform holds, with its place in the stream reachable and its samples
/// not.
///
/// Nothing outside this crate can make one, so a metadata-only transform can only ever emit items
/// it was handed, on the side they were handed to it on.
pub struct MetaItem {
    item: AnyItem,
}

impl MetaItem {
    /// Where this item sits in its stream.
    pub const fn meta(&self) -> &FrameMeta {
        self.item.meta()
    }

    /// Where this item sits in its stream, to restamp, reorder or otherwise rewrite.
    pub fn meta_mut(&mut self) -> &mut FrameMeta {
        self.item.meta_mut()
    }

    /// Whether this item is a frame rather than a mask.
    pub const fn is_frame(&self) -> bool {
        self.item.is_frame()
    }

    /// Whether this item is a mask rather than a frame.
    pub const fn is_mask(&self) -> bool {
        self.item.is_mask()
    }

    /// The sample format of the item underneath, which a metadata-only transform may read to
    /// decide how to route or group items without ever touching their planes.
    pub fn format(&self) -> StreamFormat {
        self.item.format()
    }
}

impl fmt::Debug for MetaItem {
    /// Shows only the metadata, because a metadata-only transform has no business with the
    /// samples underneath.
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
    /// Takes the item out of `index`, leaving that slot empty for the rest of this call.
    pub fn take(&mut self, index: usize) -> Option<MetaItem> {
        self.slots.get_mut(index)?.take()
    }

    /// How many slots this call was given, whether or not each still holds an item.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether this call was given no slots at all.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }
}

/// A transform that works only on where items sit in the stream, never on their samples.
///
/// It can reorder, drop, hold and restamp items, which is what re-chunking needs. Having no way to
/// reach a plane is what lets it run on whichever side its input is already on, so placing one
/// never costs a transfer.
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

/// Runs `transform` over `slots`, which is how the engine calls a metadata-only transform.
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

/// Flushes `transform`, which is how the engine ends a metadata-only transform's stream.
pub fn end_meta_transform(
    transform: &mut dyn MetaTransform,
    context: &mut FrameContext,
) -> Result<Vec<AnyItem>, StageError> {
    let flushed = transform.end_of_stream(context)?;

    Ok(flushed.into_iter().map(|meta| meta.item).collect())
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
        }
    }

    /// Puts every item in chunk 3.
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

    /// Emits each item one call late, and whatever it still holds at the end.
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
        let device = AnyItem::from(StreamItem::Frame(
            context.alloc_frame(&format(), FrameMeta::at(0)),
        ));
        let host = AnyItem::from(HostStreamItem::Frame(
            context.alloc_host_frame(&format(), FrameMeta::at(1)).freeze(),
        ));

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
        let first = AnyItem::from(StreamItem::Frame(
            context.alloc_frame(&format(), FrameMeta::at(0)),
        ));
        let second = AnyItem::from(StreamItem::Frame(
            context.alloc_frame(&format(), FrameMeta::at(1)),
        ));

        let after_first = run_meta_transform(&mut delay, &mut context, vec![Some(first)]).expect("runs");
        let after_second = run_meta_transform(&mut delay, &mut context, vec![Some(second)]).expect("runs");
        let flushed = end_meta_transform(&mut delay, &mut context).expect("flushes");

        assert!(after_first.is_empty());
        assert_eq!(after_second[0].meta().index, 0);
        assert_eq!(flushed[0].meta().index, 1);
    }
}
