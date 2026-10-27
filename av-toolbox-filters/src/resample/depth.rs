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
    Destination,
    Pattern,
    Source,
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

/// The float sample format a clip can be converted to.
#[derive(Debug, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FloatTarget {
    F32,
}

/// The sample format a clip is converted to.
#[derive(Debug, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(untagged)]
pub(crate) enum DepthTarget {
    Bits(u8),
    Float(FloatTarget),
}

/// Arguments for `resample/depth`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DepthArgs {
    /// The output depth, between 8 and 16, or `"f32"`.
    pub(crate) depth: DepthTarget,
    /// Whether any deeper integer depth the reader takes will do.
    #[serde(default)]
    pub(crate) at_least: bool,
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

fn missing_output(stage: &str) -> StageError {
    let message = format!("{stage} was built without a frame output format");

    StageError::msg(message)
}

fn bad_depth(stage: &str, depth: u8) -> StageError {
    let message = format!("{stage} needs depth between 8 and 16, got {depth}");

    StageError::msg(message)
}

fn float_floor(stage: &str) -> StageError {
    let message = format!("{stage} takes at_least only with an integer depth");

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

/// The sample formats `args` lets the output take, narrowest first.
///
/// With `at_least`, every integer depth from the deeper of the one asked for and the input's up to 16 will do.
fn offered_samples(args: &DepthArgs, input: &FrameFormat) -> Result<Vec<SampleFormat>, StageError> {
    let depth = match args.depth {
        DepthTarget::Float(FloatTarget::F32) if args.at_least => return Err(float_floor(STAGE)),
        DepthTarget::Float(FloatTarget::F32) => return Ok(vec![SampleFormat::F32]),
        DepthTarget::Bits(depth) => depth,
    };

    let target = target_sample(STAGE, depth)?;
    if !args.at_least {
        return Ok(vec![target]);
    }

    let first = match input.sample {
        SampleFormat::U8 | SampleFormat::F32 => depth,
        SampleFormat::U16 { depth: held } => depth.max(held.bits()),
    };
    let mut samples = vec![target_sample(STAGE, first)?];
    for deeper in first + 1..=16 {
        let sample = target_sample(STAGE, deeper)?;
        samples.push(sample);
    }

    Ok(samples)
}

/// The output format the solver settled on.
fn settled_output(stage: &str, formats: &ResolvedFormats) -> Result<FrameFormat, StageError> {
    let Some(output) = formats.output() else {
        return Err(missing_output(stage));
    };

    let Some(format) = output.as_frames() else {
        return Err(missing_output(stage));
    };

    Ok(*format)
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
        "Converts a clip to a different bit depth or to float, dithering when the depth is lowered.\n\n\
         A float clip is full range, running between 0 and 1. It returns to the range it is labelled \
         with when it becomes integers again."
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
        let samples = offered_samples(args, &input)?;
        let output = FrameFormat {
            sample: samples[0],
            ..input
        };
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of(samples),
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
        let output = settled_output(STAGE, formats)?;
        let converter = converter(context, STAGE, input, output.sample, args.dither);

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

    let plan = plan_conversion(input, target);
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

/// How each plane of a clip changes depth.
struct ConversionPlan {
    conversion: Conversion,
    chroma_conversion: Conversion,
    lowering: bool,
}

/// The integer range a clip labelled `range` uses. Anything not labelled full range is limited.
fn sample_range(range: ColourRange) -> SampleRange {
    match range {
        ColourRange::Full => SampleRange::Full,
        _ => SampleRange::Limited,
    }
}

/// A conversion handing float samples back unchanged.
fn float_identity() -> Conversion {
    Conversion {
        input: Source::F32,
        output: Destination::F32,
        multiplier: 1.0,
        offset: 0.0,
        divisor: 1.0,
        largest: 1.0,
    }
}

/// The conversions from `input` to `output`, and whether they lose precision.
///
/// The clip's range label decides how integers stretch, and chroma takes its own conversion since it
/// is centred on the middle value. Float samples are full range with chroma centred on zero, and always
/// lose precision when they become integers.
fn plan_conversion(input: FrameFormat, output: SampleFormat) -> ConversionPlan {
    let range = sample_range(input.colour.range);
    let input_float = input.sample == SampleFormat::F32;
    let output_float = output == SampleFormat::F32;

    if input_float && output_float {
        let conversion = float_identity();

        return ConversionPlan {
            conversion,
            chroma_conversion: conversion,
            lowering: false,
        };
    }

    if input_float {
        let output_bits = bits_of(output);
        let conversion = Conversion::from_float(output_bits, range, false);
        let chroma_conversion = Conversion::from_float(output_bits, range, true);

        return ConversionPlan {
            conversion,
            chroma_conversion,
            lowering: true,
        };
    }

    let input_bits = bits_of(input.sample);
    if output_float {
        let conversion = Conversion::to_float(input_bits, range, false);
        let chroma_conversion = Conversion::to_float(input_bits, range, true);

        return ConversionPlan {
            conversion,
            chroma_conversion,
            lowering: false,
        };
    }

    let output_bits = bits_of(output);
    let conversion = Conversion::new(input_bits, output_bits, range, false);
    let chroma_conversion = Conversion::new(input_bits, output_bits, range, true);

    ConversionPlan {
        conversion,
        chroma_conversion,
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
