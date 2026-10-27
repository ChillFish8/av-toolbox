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
use av_toolbox_cubek::resize::{ResizePlanes, Samples, Weights, apply, taps, upload};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ChromaLocation,
    ColourInfo,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::checks::accepts_integer_or_f32_frames;
use crate::resize::{KernelChoice, samples_for};
use crate::sizing::Sizing;

const STAGE: &str = "resample/chroma";

/// The subsampling names a clip's chroma resamples to.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
pub(crate) enum SubsamplingName {
    #[serde(rename = "4:4:4")]
    Yuv444,
    #[serde(rename = "4:2:2")]
    Yuv422,
    #[serde(rename = "4:2:0")]
    Yuv420,
}

impl From<SubsamplingName> for Subsampling {
    fn from(name: SubsamplingName) -> Self {
        match name {
            SubsamplingName::Yuv444 => Self::Yuv444,
            SubsamplingName::Yuv422 => Self::Yuv422,
            SubsamplingName::Yuv420 => Self::Yuv420,
        }
    }
}

/// Where a clip's chroma resamples to sit.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LocationName {
    Left,
    Center,
    TopLeft,
    Top,
    BottomLeft,
    Bottom,
}

impl From<LocationName> for ChromaLocation {
    fn from(name: LocationName) -> Self {
        match name {
            LocationName::Left => Self::Left,
            LocationName::Center => Self::Center,
            LocationName::TopLeft => Self::TopLeft,
            LocationName::Top => Self::Top,
            LocationName::BottomLeft => Self::BottomLeft,
            LocationName::Bottom => Self::Bottom,
        }
    }
}

/// Arguments for `resample/chroma`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ChromaArgs {
    /// The chroma subsampling to resample to.
    pub(crate) subsampling: SubsamplingName,
    /// The kernel to resample chroma with.
    #[serde(default)]
    pub(crate) kernel: KernelChoice,
    /// How much the cubic blurs. Only with `bicubic`.
    pub(crate) b: Option<f32>,
    /// How much the cubic sharpens. Only with `bicubic`.
    pub(crate) c: Option<f32>,
    /// How many input samples the kernel reaches either side. Only with `lanczos`.
    pub(crate) taps: Option<u32>,
    /// Where the resampled chroma sits, the input's own siting by default.
    pub(crate) location: Option<LocationName>,
}

/// The output chroma planes, and where each axis reads from in the input chroma planes.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(super) struct ChromaAxes {
    pub(super) width: u32,
    pub(super) height: u32,
    /// The scale and offset across, as [taps](av_toolbox_cubek::resize::taps) takes them.
    pub(super) across: (f32, f32),
    /// The scale and offset down.
    pub(super) down: (f32, f32),
}

/// Where the input sits on one axis, and where the output wants to sit.
struct AxisSiting {
    input_factor: u32,
    input_offset: f32,
    output_factor: u32,
    output_offset: f32,
}

/// The scale and offset that place output chroma sample `j` at luma position
/// `output_factor * j + output_offset`, in input chroma samples.
fn axis(siting: AxisSiting) -> (f32, f32) {
    let input_factor = siting.input_factor as f32;
    let scale = siting.output_factor as f32 / input_factor;
    let offset = (siting.output_offset - siting.input_offset) / input_factor;

    (scale, offset)
}

/// The chroma planes of `input`, subsampled by `source`, resampled to `target` and sited at `location`.
pub(super) fn chroma_axes(
    input: &FrameFormat,
    source: Subsampling,
    target: Subsampling,
    location: ChromaLocation,
) -> ChromaAxes {
    let (input_factor_x, input_factor_y) = source.factors();
    let (output_factor_x, output_factor_y) = target.factors();
    let (input_offset_x, input_offset_y) = input
        .colour
        .chroma_location
        .offsets(input_factor_x, input_factor_y);
    let (output_offset_x, output_offset_y) = location.offsets(output_factor_x, output_factor_y);
    let (width, height) = target.chroma_dimensions(input.width, input.height);

    let across = axis(AxisSiting {
        input_factor: input_factor_x,
        input_offset: input_offset_x,
        output_factor: output_factor_x,
        output_offset: output_offset_x,
    });
    let down = axis(AxisSiting {
        input_factor: input_factor_y,
        input_offset: input_offset_y,
        output_factor: output_factor_y,
        output_offset: output_offset_y,
    });

    ChromaAxes {
        width,
        height,
        across,
        down,
    }
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} resamples frames, not masks");

    StageError::msg(message)
}

fn not_yuv(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} resamples YUV frames, not {layout}");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_channel() -> StageError {
    let message = format!("{STAGE} was handed a frame missing a channel");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

fn not_divisible(dimension: &str, factor: u32, subsampling: Subsampling, value: u32) -> StageError {
    let message = format!("{STAGE} needs a {dimension} divisible by {factor} for {subsampling}, got {value}");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

fn location_of(args: &ChromaArgs, input: &FrameFormat) -> ChromaLocation {
    args.location
        .map(ChromaLocation::from)
        .unwrap_or(input.colour.chroma_location)
}

/// Checks the arguments against the input and gives the output format.
fn resampled(input: &FrameFormat, args: &ChromaArgs) -> Result<FrameFormat, StageError> {
    let FrameLayout::Yuv { .. } = input.layout else {
        return Err(not_yuv(input.layout));
    };

    let target = Subsampling::from(args.subsampling);
    args.kernel.kernel(STAGE, args.b, args.c, args.taps)?;

    let (factor_x, factor_y) = target.factors();
    if !input.width.is_multiple_of(factor_x) {
        return Err(not_divisible("width", factor_x, target, input.width));
    }

    if !input.height.is_multiple_of(factor_y) {
        return Err(not_divisible("height", factor_y, target, input.height));
    }

    let location = location_of(args, input);
    let colour = ColourInfo {
        chroma_location: location,
        ..input.colour
    };

    Ok(FrameFormat {
        layout: FrameLayout::Yuv { subsampling: target },
        colour,
        ..*input
    })
}

/// Changes a YUV clip's chroma subsampling and siting.
pub(crate) struct Chroma;

impl StageDefinition for Chroma {
    type Args = ChromaArgs;

    fn package() -> &'static str {
        "resample"
    }

    fn identifier() -> &'static str {
        "chroma"
    }

    fn description() -> &'static str {
        "Changes how many colour samples a YUV clip stores, such as 4:2:0 to 4:4:4.\n\n\
         It can also move where the colour samples sit relative to the brightness samples."
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

impl FilterDefinition for Chroma {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(samples) = accepts_integer_or_f32_frames(offered) else {
            return Err(not_frames());
        };

        Ok(samples)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        let output = resampled(&input, args)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let output = resampled(&input, &args)?;
        let kernel = args.kernel.kernel(STAGE, args.b, args.c, args.taps)?;

        let source = match input.layout {
            FrameLayout::Yuv { subsampling } => subsampling,
            _ => return Err(not_yuv(input.layout)),
        };
        let target = match output.layout {
            FrameLayout::Yuv { subsampling } => subsampling,
            _ => return Err(not_yuv(output.layout)),
        };

        let axes = chroma_axes(&input, source, target, output.colour.chroma_location);
        let identity_axis = (1.0, 0.0);
        let same_samples = source == target && axes.across == identity_axis && axes.down == identity_axis;

        let client = context.client().clone();
        let across_taps = taps(axes.width, axes.across.0, axes.across.1, &kernel);
        let down_taps = taps(axes.height, axes.down.0, axes.down.1, &kernel);
        let across = upload(&client, &across_taps);
        let down = upload(&client, &down_taps);
        let samples = samples_for(output.sample);

        let resampler = ChromaResampler {
            client,
            output,
            target,
            axes,
            across,
            down,
            samples,
            same_samples,
        };

        Ok(Box::new(resampler))
    }
}

/// Resamples a YUV clip's chroma planes to a new subsampling and siting, sharing the luma plane.
struct ChromaResampler<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
    target: Subsampling,
    axes: ChromaAxes,
    across: Weights,
    down: Weights,
    samples: Samples,
    /// Whether the chroma planes need no resampling, only relabelling when the siting changed.
    same_samples: bool,
}

impl<R: Runtime> ChromaResampler<R> {
    fn resize(&self, context: &FrameContext, input: &Plane, output: &Plane) {
        let (in_width, in_height) = input.dimensions();
        let (out_width, out_height) = output.dimensions();
        let scratch_bytes = out_width as usize * in_height as usize * size_of::<f32>();
        let scratch = context.scratch(scratch_bytes);

        let input_handle = input.buffer().handle();
        let scratch_handle = scratch.handle();
        let output_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let input_plane = unsafe { DevicePlane::new(input_handle, in_width, in_height) };

        // Safety: the scratch was sized for exactly `out_width * in_height` f32 samples.
        let scratch_plane = unsafe { DevicePlane::new(scratch_handle, out_width, in_height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let output_plane = unsafe { DevicePlane::new(output_handle, out_width, out_height) };

        let planes = ResizePlanes {
            input: input_plane,
            scratch: scratch_plane,
            output: output_plane,
        };
        let sizing = Sizing(context);

        apply(
            &self.client,
            &sizing,
            planes,
            &self.across,
            &self.down,
            self.samples,
        );
    }
}

impl<R: Runtime> StageInstance for ChromaResampler<R> {}

impl<R: Runtime> Filter for ChromaResampler<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        if input.format() == &self.output {
            return Ok(vec![input]);
        }

        let Some(luma) = input.plane(Channel::Y) else {
            return Err(missing_channel());
        };
        let Some(input_u) = input.plane(Channel::U) else {
            return Err(missing_channel());
        };
        let Some(input_v) = input.plane(Channel::V) else {
            return Err(missing_channel());
        };

        let luma = luma.clone();
        let input_u = input_u.clone();
        let input_v = input_v.clone();
        let meta = input.meta().clone();

        if self.same_samples {
            let planes = vec![luma, input_u, input_v];
            let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

            return Ok(vec![output]);
        }

        let chroma_format = FrameFormat {
            width: self.axes.width,
            height: self.axes.height,
            layout: FrameLayout::Chroma {
                subsampling: self.target,
            },
            ..self.output
        };
        let chroma_frame = context.alloc_frame(&chroma_format, meta.clone());

        let Some(output_u) = chroma_frame.plane(Channel::U) else {
            return Err(missing_channel());
        };
        let Some(output_v) = chroma_frame.plane(Channel::V) else {
            return Err(missing_channel());
        };
        let output_u = output_u.clone();
        let output_v = output_v.clone();

        self.resize(context, &input_u, &output_u);
        self.resize(context, &input_v, &output_v);

        let planes = vec![luma, output_u, output_v];
        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(vec![output])
    }
}
