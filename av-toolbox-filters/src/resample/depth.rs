use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::depth::{
    Conversion,
    DepthPlanes,
    Pattern,
    Thresholds,
    convert,
    plane_shift,
    upload_pattern,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourRange,
    ComputeClient,
    Depth as BitDepth,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::bits_of;
use crate::sizing::Sizing;

const STAGE: &str = "resample/depth";

/// How a lowered depth rounds.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Dither {
    None,
    Ordered,
    #[default]
    BlueNoise,
}

/// Arguments for `resample/depth`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DepthArgs {
    /// The output bit depth, between 8 and 16.
    pub(crate) depth: u8,
    /// How a lowered depth rounds.
    #[serde(default)]
    pub(crate) dither: Dither,
}

pub(super) fn not_frames(stage: &str) -> StageError {
    let message = format!("{stage} converts frames, not masks");

    StageError::msg(message)
}

fn missing_input(stage: &str) -> StageError {
    let message = format!("{stage} needs its input bound");

    StageError::msg(message)
}

fn missing_channel(stage: &str) -> StageError {
    let message = format!("{stage} was handed a frame missing a channel");

    StageError::msg(message)
}

fn bad_depth(stage: &str, depth: u8) -> StageError {
    let message = format!("{stage} needs depth between 8 and 16, got {depth}");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
pub(super) fn checked_input(stage: &str, chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input(stage));
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames(stage));
    };

    Ok(*format)
}

/// The sample format `depth` describes, or an error when it is outside 8..=16.
pub(super) fn target_sample(stage: &str, depth: u8) -> Result<SampleFormat, StageError> {
    if depth == 8 {
        return Ok(SampleFormat::U8);
    }

    let Ok(depth) = BitDepth::new(depth) else {
        return Err(bad_depth(stage, depth));
    };

    Ok(SampleFormat::U16 { depth })
}

/// The thresholds a conversion rounds through.
///
/// Raising the depth always rounds plainly, whatever `dither` says, since dithering only matters
/// when precision is lost.
fn thresholds(dither: Dither, lowering: bool) -> Thresholds {
    if !lowering || dither == Dither::None {
        return Thresholds::rounding();
    }

    match dither {
        Dither::Ordered => Thresholds::ordered(),
        _ => Thresholds::blue_noise(),
    }
}

/// Changes every plane of a clip to a different bit depth.
pub(crate) struct Depth;

impl StageDefinition for Depth {
    type Args = DepthArgs;

    fn package() -> &'static str {
        "resample"
    }

    fn identifier() -> &'static str {
        "depth"
    }

    fn description() -> &'static str {
        "Converts a clip to a different bit depth, dithering when the depth is lowered."
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

impl FilterDefinition for Depth {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(not_frames(STAGE));
        };

        if !matches!(format.sample, SampleFormat::U8 | SampleFormat::U16 { .. }) {
            return Ok(FormatSet::new());
        }

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        let target = target_sample(STAGE, args.depth)?;
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
        let target = target_sample(STAGE, args.depth)?;
        let converter = converter(context, STAGE, input, target, args.dither);

        Ok(converter)
    }
}

/// A filter converting every frame of `input` to `target`, handing frames already there straight back.
pub(super) fn converter<R: Runtime>(
    context: &BuildContext<R>,
    stage: &'static str,
    input: FrameFormat,
    target: SampleFormat,
    dither: Dither,
) -> Box<dyn Filter> {
    let output = FrameFormat {
        sample: target,
        ..input
    };
    let identity = input.sample == output.sample;

    let input_bits = bits_of(input.sample);
    let output_bits = bits_of(output.sample);
    let full_range = input.colour.range == ColourRange::Full;
    let conversion = Conversion::new(input_bits, output_bits, full_range);

    let lowering = output_bits < input_bits;
    let thresholds = thresholds(dither, lowering);
    let client = context.client().clone();
    let pattern = upload_pattern(&client, &thresholds);
    let converter = Converter {
        client,
        stage,
        output,
        conversion,
        pattern,
        identity,
    };

    Box::new(converter)
}

/// Converts every plane of a frame from one bit depth to another.
struct Converter<R: Runtime> {
    client: ComputeClient<R>,
    stage: &'static str,
    output: FrameFormat,
    conversion: Conversion,
    pattern: Pattern,
    identity: bool,
}

impl<R: Runtime> Converter<R> {
    fn convert_plane(&self, context: &FrameContext, input: &Plane, output: &Plane, index: u32) {
        let (in_width, in_height) = input.dimensions();
        let (out_width, out_height) = output.dimensions();

        let input_handle = input.buffer().handle();
        let output_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let input_plane = unsafe { DevicePlane::new(input_handle, in_width, in_height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let output_plane = unsafe { DevicePlane::new(output_handle, out_width, out_height) };

        let planes = DepthPlanes {
            input: input_plane,
            output: output_plane,
        };
        let sizing = Sizing(context);
        let shift = plane_shift(index);

        convert(
            &self.client,
            &sizing,
            planes,
            &self.conversion,
            &self.pattern,
            shift,
        );
    }
}

impl<R: Runtime> StageInstance for Converter<R> {}

impl<R: Runtime> Filter for Converter<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input(self.stage));
        };

        if self.identity {
            return Ok(vec![input]);
        }

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);

        for (index, channel) in self.output.channels().iter().enumerate() {
            let Some(input_plane) = input.plane(*channel) else {
                return Err(missing_channel(self.stage));
            };

            let Some(output_plane) = output.plane(*channel) else {
                return Err(missing_channel(self.stage));
            };

            self.convert_plane(context, input_plane, output_plane, index as u32);
        }

        Ok(vec![output])
    }
}
