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
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
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
use crate::checks::accepts_integer_or_f32_frames;
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
        "Converts a clip to a different bit depth, dithering when the depth is lowered.\n\n\
         A float clip is read as running between 0 and 1, so 1 becomes the largest output value."
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
        let Some(samples) = accepts_integer_or_f32_frames(offered) else {
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
        let target = target_sample(STAGE, args.depth)?;
        let output = converted_format(input, target);
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
    let output = converted_format(input, target);
    let identity = input.sample == output.sample;

    let output_bits = bits_of(output.sample);
    let plan = plan_conversion(input, output_bits);
    let thresholds = thresholds(dither, plan.lowering);
    let client = context.client().clone();
    let pattern = upload_pattern(&client, &thresholds);
    let converter = Converter {
        client,
        stage,
        output,
        conversion: plan.conversion,
        chroma_conversion: plan.chroma_conversion,
        pattern,
        identity,
    };

    Box::new(converter)
}

/// The format of `input` converted to `target`.
///
/// Float samples are always full range, so a float clip comes out labelled full range.
fn converted_format(input: FrameFormat, target: SampleFormat) -> FrameFormat {
    let mut output = FrameFormat {
        sample: target,
        ..input
    };

    if input.sample == SampleFormat::F32 {
        output.colour.range = ColourRange::Full;
    }

    output
}

/// How each plane of a clip changes depth.
struct ConversionPlan {
    conversion: Conversion,
    chroma_conversion: Conversion,
    lowering: bool,
}

/// The conversions from `input` to `output_bits`, and whether they lose precision.
///
/// Float chroma is centred on zero, so it takes its own conversion. Float samples always lose
/// precision when they become integers.
fn plan_conversion(input: FrameFormat, output_bits: u32) -> ConversionPlan {
    if input.sample == SampleFormat::F32 {
        let conversion = Conversion::from_float(output_bits, SampleRange::Full, false);
        let chroma_conversion = Conversion::from_float(output_bits, SampleRange::Full, true);

        return ConversionPlan {
            conversion,
            chroma_conversion,
            lowering: true,
        };
    }

    let input_bits = bits_of(input.sample);
    let full_range = input.colour.range == ColourRange::Full;
    let conversion = Conversion::new(input_bits, output_bits, full_range);

    ConversionPlan {
        conversion,
        chroma_conversion: conversion,
        lowering: output_bits < input_bits,
    }
}

/// Converts every plane of a frame from one bit depth to another.
struct Converter<R: Runtime> {
    client: ComputeClient<R>,
    stage: &'static str,
    output: FrameFormat,
    conversion: Conversion,
    chroma_conversion: Conversion,
    pattern: Pattern,
    identity: bool,
}

impl<R: Runtime> Converter<R> {
    fn convert_plane(
        &self,
        context: &FrameContext,
        input: &Plane,
        output: &Plane,
        channel: Channel,
        index: u32,
    ) {
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
        let conversion = match channel {
            Channel::U | Channel::V => &self.chroma_conversion,
            _ => &self.conversion,
        };

        convert(&self.client, &sizing, planes, conversion, &self.pattern, shift);
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

            self.convert_plane(context, input_plane, output_plane, *channel, index as u32);
        }

        Ok(vec![output])
    }
}
