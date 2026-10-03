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
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::warp::{BlurKind, WarpParams, WarpPlanes, warp_sharp};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
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
use crate::planes::ChannelName;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "warp/awarpsharp";

/// The largest `thresh`, on the 8-bit scale.
const MAX_THRESH: u32 = 255;

/// The most blur passes a warp runs.
const MAX_BLUR: u32 = 100;

const MIN_DEPTH: i32 = -128;

const MAX_DEPTH: i32 = 127;

/// Which blur smooths the edge mask.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum BlurType {
    #[default]
    R6,
    R2,
}

fn default_thresh() -> u32 {
    128
}

fn default_depth() -> i32 {
    16
}

/// Arguments for `warp/awarpsharp`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct AwarpsharpArgs {
    /// The most an edge can weigh in the mask, on the 8-bit scale.
    #[serde(default = "default_thresh")]
    pub(crate) thresh: u32,
    /// How many times the mask is blurred. 2 for `r6` and 3 for `r2` when not given.
    #[serde(default)]
    pub(crate) blur: Option<u32>,
    /// Which blur smooths the mask.
    #[serde(default, rename = "type")]
    pub(crate) blur_type: BlurType,
    /// How far samples move toward edges. A negative depth moves them away.
    #[serde(default = "default_depth")]
    pub(crate) depth: i32,
    /// The planes to warp, every plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// What `warp/awarpsharp` does to one frame format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WarpSettings {
    pub(crate) params: WarpParams,
    /// Whether each plane, in channel order, is warped.
    pub(crate) selected: Vec<bool>,
}

fn bad_thresh(thresh: u32) -> StageError {
    let message = format!("{STAGE} needs thresh between 0 and {MAX_THRESH}, got {thresh}");

    StageError::msg(message)
}

fn bad_blur(blur: u32) -> StageError {
    let message = format!("{STAGE} needs blur between 0 and {MAX_BLUR}, got {blur}");

    StageError::msg(message)
}

fn bad_depth(depth: i32) -> StageError {
    let message = format!("{STAGE} needs depth between {MIN_DEPTH} and {MAX_DEPTH}, got {depth}");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} warps frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_channel(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The warp settings for `input`, once the arguments are checked.
pub(crate) fn warp_settings(args: &AwarpsharpArgs, input: &FrameFormat) -> Result<WarpSettings, StageError> {
    if args.thresh > MAX_THRESH {
        return Err(bad_thresh(args.thresh));
    }

    let (blur, default_passes) = match args.blur_type {
        BlurType::R6 => (BlurKind::R6, 2),
        BlurType::R2 => (BlurKind::R2, 3),
    };

    let passes = args.blur.unwrap_or(default_passes);
    if passes > MAX_BLUR {
        return Err(bad_blur(passes));
    }

    if !(MIN_DEPTH..=MAX_DEPTH).contains(&args.depth) {
        return Err(bad_depth(args.depth));
    }

    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::Every)?;

    let bits = bits_of(input.sample);
    let element = if bits == 8 { Element::U8 } else { Element::U16 };
    let params = WarpParams {
        element,
        bits,
        thresh: args.thresh,
        blur,
        passes,
        depth: args.depth,
    };

    Ok(WarpSettings { params, selected })
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

/// Sharpens by pulling samples toward nearby edges, the way aWarpSharp2 does.
pub(crate) struct Awarpsharp;

impl StageDefinition for Awarpsharp {
    type Args = AwarpsharpArgs;

    fn package() -> &'static str {
        "warp"
    }

    fn identifier() -> &'static str {
        "awarpsharp"
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

impl FilterDefinition for Awarpsharp {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(not_frames());
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
        let input = checked_input(chosen)?;
        warp_settings(args, &input)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([input.sample]),
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
        let settings = warp_settings(&args, &input)?;
        let client = context.client().clone();
        let warper = Warper {
            client,
            output: input,
            settings,
        };

        Ok(Box::new(warper))
    }
}

/// Warps every selected plane of a frame, sharing the rest.
struct Warper<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
    settings: WarpSettings,
}

impl<R: Runtime> Warper<R> {
    /// Warps `source` into a fresh plane of the same size and hands that plane back.
    fn warp(&self, context: &FrameContext, source: &Plane, meta: &FrameMeta) -> Result<Plane, StageError> {
        let (width, height) = source.dimensions();
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.output
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let Some(output) = target.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        let samples = width as usize * height as usize;
        let mask_bytes = samples * size_of::<u32>();
        let mask_buffer = context.scratch(mask_bytes);
        let scratch_buffer = context.scratch(mask_bytes);

        let source_handle = source.buffer().handle();
        let output_handle = output.buffer().handle();
        let mask_handle = mask_buffer.handle();
        let scratch_handle = scratch_buffer.handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let input_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let output_plane = unsafe { DevicePlane::new(output_handle, width, height) };

        // Safety: the buffer was sized for `width * height` u32 values.
        let mask_plane = unsafe { DevicePlane::new(mask_handle, width, height) };

        // Safety: the buffer was sized for `width * height` u32 values.
        let scratch_plane = unsafe { DevicePlane::new(scratch_handle, width, height) };

        let planes = WarpPlanes {
            input: input_plane,
            mask: mask_plane,
            scratch: scratch_plane,
            output: output_plane,
        };
        let sizing = Sizing(context);

        warp_sharp(&self.client, &sizing, planes, &self.settings.params);

        Ok(output.clone())
    }
}

impl<R: Runtime> StageInstance for Warper<R> {}

impl<R: Runtime> Filter for Warper<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        let meta = input.meta().clone();
        let channels = self.output.channels();
        let mut planes = Vec::with_capacity(channels.len());

        for (plane_index, channel) in channels.iter().enumerate() {
            let Some(source) = input.plane(*channel) else {
                return Err(missing_channel(*channel));
            };

            if !self.settings.selected[plane_index] {
                planes.push(source.clone());
                continue;
            }

            let warped = self.warp(context, source, &meta)?;
            planes.push(warped);
        }

        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(vec![output])
    }
}
