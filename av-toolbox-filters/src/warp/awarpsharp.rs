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
use av_toolbox_cubek::values::{narrow, widen};
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
use crate::codes::CodeValues;
use crate::planes::ChannelName;
use crate::scratch::{self, Scratch};
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

    fn description() -> &'static str {
        "Sharpens edges by pulling nearby samples towards them, which also thins lines."
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
        let codes = float_codes(&input);
        let warper = Warper {
            client,
            output: input,
            settings,
            codes,
        };

        Ok(Box::new(warper))
    }
}

/// The code values an F32 `input` warps as, or `None` for integer input, which warps as it is.
fn float_codes(input: &FrameFormat) -> Option<CodeValues> {
    if input.sample != SampleFormat::F32 {
        return None;
    }

    let codes = CodeValues::new(input, SampleFormat::F32).whole_codes();

    Some(codes)
}

/// Warps every selected plane of a frame, sharing the rest.
struct Warper<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
    settings: WarpSettings,
    /// How F32 planes map onto the 16-bit codes the warp runs on.
    codes: Option<CodeValues>,
}

impl<R: Runtime> Warper<R> {
    /// Warps `source` into a fresh plane of the same size and hands that plane back.
    fn warp(
        &self,
        context: &FrameContext,
        source: &Plane,
        channel: Channel,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
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

        // Safety: every kernel given this plane reads it as the frame's sample format.
        let source_plane = unsafe { scratch::device_plane(source) };

        // Safety: every kernel given this plane writes it as the frame's sample format.
        let output_plane = unsafe { scratch::device_plane(output) };

        match &self.codes {
            Some(codes) => self.warp_float(context, codes, source_plane, output_plane, channel),
            None => self.warp_samples(context, source_plane, output_plane),
        }

        Ok(output.clone())
    }

    /// Warps the integer samples of `source` into `output`.
    fn warp_samples(&self, context: &FrameContext, source: DevicePlane<'_>, output: DevicePlane<'_>) {
        let sizing = Sizing(context);
        let width = source.width();
        let height = source.height();

        let mask = Scratch::new::<u32>(context, width, height);
        let warp_scratch = Scratch::new::<u32>(context, width, height);
        let mask_plane = mask.plane();
        let warp_scratch_plane = warp_scratch.plane();
        let planes = WarpPlanes {
            input: source,
            mask: mask_plane,
            scratch: warp_scratch_plane,
            output,
        };
        warp_sharp(&self.client, &sizing, planes, &self.settings.params);
    }

    /// Warps an F32 `source` as whole 16-bit codes into `output`.
    fn warp_float(
        &self,
        context: &FrameContext,
        codes: &CodeValues,
        source: DevicePlane<'_>,
        output: DevicePlane<'_>,
        channel: Channel,
    ) {
        let sizing = Sizing(context);
        let width = source.width();
        let height = source.height();
        let params = &self.settings.params;

        let values = Scratch::new::<f32>(context, width, height);
        let values_plane = values.plane();
        codes.widen(&self.client, &sizing, source, values_plane, channel);

        let narrowed = scratch::samples(context, width, height, params.element);
        let narrowed_plane = narrowed.plane();
        narrow(
            &self.client,
            &sizing,
            values_plane,
            narrowed_plane,
            params.element,
            codes.peak(),
        );

        let warped = scratch::samples(context, width, height, params.element);
        let warped_plane = warped.plane();
        self.warp_samples(context, narrowed_plane, warped_plane);

        widen(&self.client, &sizing, warped_plane, values_plane, params.element);
        codes.store(&self.client, &sizing, values_plane, output, channel);
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

            let warped = self.warp(context, source, *channel, &meta)?;
            planes.push(warped);
        }

        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(vec![output])
    }
}
