use av_toolbox_core::{
    BoundInputs,
    InputSignature,
    Inputs,
    Sink,
    SinkDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    HostSide,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

const INPUTS: &InputSignature = &InputSignature::with_variadic(&[]);

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct DiscardArgs {}

/// A sink dropping every frame and mask it is handed.
pub(crate) struct Discard;

impl StageDefinition for Discard {
    type Args = DiscardArgs;

    fn package() -> &'static str {
        "null"
    }

    fn identifier() -> &'static str {
        "discard"
    }

    fn description() -> &'static str {
        "Drops every frame and mask it is given.\n\nUseful when a run only needs its metadata or artefacts."
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }
}

impl SinkDefinition for Discard {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts(offered)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        Ok(Box::new(DiscardSink))
    }
}

fn accepts(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let sample = offered.sample();

    Ok(FormatSet::of([sample]))
}

struct DiscardSink;

impl StageInstance for DiscardSink {}

impl Sink<HostSide> for DiscardSink {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs<HostSide>) -> Result<(), StageError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::{Inputs, Sink};
    use av_toolbox_gpu::{
        FrameMeta,
        HostSide,
        HostStreamItem,
        MaskFormat,
        SampleFormat,
        StreamFormat,
        testing_context,
    };

    use super::*;
    use crate::shared::testing::{filled_frame, yuv420};

    fn mask_format() -> MaskFormat {
        MaskFormat {
            frame_rate: None,
            width: 4,
            height: 2,
            sample: SampleFormat::U8,
            chroma: None,
        }
    }

    #[test]
    fn discard_drops_frames_and_masks_together() {
        let mut context = testing_context();
        let format = yuv420(4, 2, SampleFormat::U8, None);
        let frame = filled_frame(&format, 0, 5);
        let mask = context.alloc_host_mask(&mask_format(), FrameMeta::at(0)).freeze();
        let frame_item = HostStreamItem::from(frame);
        let mask_item = HostStreamItem::from(mask);
        let inputs = Inputs::<HostSide>::from_slots(vec![Some(frame_item), Some(mask_item)]);
        let mut sink = DiscardSink;

        sink.consume(&mut context, inputs).expect("drops both");
        sink.end_of_stream(&mut context).expect("ends");
    }

    #[test]
    fn discard_takes_a_mask_in_its_own_sample() {
        let offered = StreamFormat::Mask(mask_format());

        let accepted = accepts(offered).expect("a mask is taken");

        assert_eq!(accepted, FormatSet::of([SampleFormat::U8]));
    }
}
