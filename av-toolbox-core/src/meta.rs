use std::fmt;

use av_toolbox_gpu::{AnyItem, FrameContext, FrameMeta, MetaRecord, StreamFormat};

use crate::{StageError, StageInstance};

/// An item a metadata-only stage is handed, with its place in the stream reachable and its samples
/// not.
pub struct MetaItem {
    item: AnyItem,
}

impl MetaItem {
    pub(crate) fn new(item: AnyItem) -> Self {
        Self { item }
    }

    pub(crate) fn into_any(self) -> AnyItem {
        self.item
    }

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

/// What a metadata-only sink was handed this step, in the same slots [Inputs](crate::Inputs)
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

    /// Called with each record that reaches this sink, in its place in the stream.
    fn consume_record(
        &mut self,
        _context: &mut FrameContext,
        _record: &MetaRecord,
    ) -> Result<(), StageError> {
        Ok(())
    }
}

pub fn run_meta_sink(
    sink: &mut dyn MetaSink,
    context: &mut FrameContext,
    slots: Vec<Option<AnyItem>>,
) -> Result<(), StageError> {
    let slots = slots.into_iter().map(|slot| slot.map(MetaItem::new)).collect();

    sink.consume(context, MetaInputs { slots })
}

pub fn end_meta_sink(sink: &mut dyn MetaSink, context: &mut FrameContext) -> Result<(), StageError> {
    sink.end_of_stream(context)
}

pub fn record_meta_sink(
    sink: &mut dyn MetaSink,
    context: &mut FrameContext,
    record: &MetaRecord,
) -> Result<(), StageError> {
    sink.consume_record(context, record)
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

    struct Counting {
        seen: Vec<u64>,
        records: Vec<String>,
        ended: bool,
    }

    impl StageInstance for Counting {}

    impl MetaSink for Counting {
        fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
            let item = inputs.take(0).expect("an item");
            self.seen.push(item.meta().index);

            Ok(())
        }

        fn consume_record(
            &mut self,
            _context: &mut FrameContext,
            record: &MetaRecord,
        ) -> Result<(), StageError> {
            self.records.push(record.kind().to_string());

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
            records: Vec::new(),
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
    fn a_meta_sink_is_handed_records_apart_from_items() {
        let mut sink = Counting {
            seen: Vec::new(),
            records: Vec::new(),
            ended: false,
        };
        let mut stage_context = testing_context();
        let record = MetaRecord::new("chunk");

        record_meta_sink(&mut sink, &mut stage_context, &record).expect("consumes");

        assert_eq!(sink.records, vec![String::from("chunk")]);
        assert!(sink.seen.is_empty());
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
