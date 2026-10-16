use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Parallelism,
    StageDefinition,
    StageError,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::depth::{Dither, checked_input, converter, not_frames, target_sample};
use crate::bits::bits_of;
use crate::checks::accepts_frames;

const STAGE: &str = "resample/min_depth";

/// Arguments for `resample/min_depth`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct MinDepthArgs {
    /// The lowest bit depth the output has, between 8 and 16.
    pub(crate) depth: u8,
}

/// The sample format `input` leaves as, raised to `depth` when it holds fewer bits.
fn raised_sample(input: &FrameFormat, depth: u8) -> Result<SampleFormat, StageError> {
    let minimum = target_sample(STAGE, depth)?;
    let input_bits = bits_of(input.sample);
    let minimum_bits = bits_of(minimum);

    if input_bits >= minimum_bits {
        return Ok(input.sample);
    }

    Ok(minimum)
}

/// Raises a clip to at least a given bit depth, leaving deeper clips as they are.
pub(crate) struct MinDepth;

impl StageDefinition for MinDepth {
    type Args = MinDepthArgs;

    fn package() -> &'static str {
        "resample"
    }

    fn identifier() -> &'static str {
        "min_depth"
    }

    fn description() -> &'static str {
        "Raises a clip to at least a given bit depth, leaving clips already that deep untouched."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for MinDepth {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(samples) = accepts_frames(offered) else {
            return Err(not_frames(STAGE));
        };

        Ok(samples)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        let target = raised_sample(&input, args.depth)?;
        let output = FrameFormat {
            sample: target,
            ..input
        };
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([target]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(STAGE, inputs)?;
        let target = raised_sample(&input, args.depth)?;

        // Raising the depth never loses precision, so there is nothing to dither.
        let converter = converter(context, STAGE, input, target, Dither::None);

        Ok(converter)
    }
}
