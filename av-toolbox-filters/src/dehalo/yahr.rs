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
use av_toolbox_cubek::dehalo::yahr::{YahrParams, YahrPlanes, yahr_combine};
use av_toolbox_cubek::dehalo::{DifferencePlanes, difference};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::neighbourhood::{MinBlurPlanes, NeighbourPlanes, Smooth, min_blur, smooth};
use av_toolbox_cubek::rgvs::{RepairMode, RepairPlanes, repair};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
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

use super::padded_warp::padded_warp;
use crate::bits::bits_of;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "dehalo/yahr";

const MAX_BLUR: u32 = 100;

const MIN_DEPTH: i32 = -128;

const MAX_DEPTH: i32 = 127;

fn default_blur() -> u32 {
    2
}

fn default_depth() -> i32 {
    32
}

/// Arguments for `dehalo/yahr`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct YahrArgs {
    /// How many times the edge mask is blurred before it warps.
    #[serde(default = "default_blur")]
    pub(crate) blur: u32,
    /// How far samples move toward edges. A negative depth moves them away.
    #[serde(default = "default_depth")]
    pub(crate) depth: i32,
    /// The planes to clean, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn bad_blur(blur: u32) -> StageError {
    let message = format!("{STAGE} needs blur between 0 and {MAX_BLUR}, got {blur}");

    StageError::msg(message)
}

fn bad_depth(depth: i32) -> StageError {
    let message = format!("{STAGE} needs depth between {MIN_DEPTH} and {MAX_DEPTH}, got {depth}");

    StageError::msg(message)
}

/// The YAHR parameters for `sample`, once the arguments are checked.
pub(crate) fn yahr_params(args: &YahrArgs, sample: SampleFormat) -> Result<YahrParams, StageError> {
    if args.blur > MAX_BLUR {
        return Err(bad_blur(args.blur));
    }

    if !(MIN_DEPTH..=MAX_DEPTH).contains(&args.depth) {
        return Err(bad_depth(args.depth));
    }

    let bits = bits_of(sample);
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    Ok(YahrParams {
        blur: args.blur,
        depth: args.depth,
        element,
        bits,
    })
}

/// Runs MinBlur radius 2 then a binomial smooth on `input`, into a fresh scratch plane.
fn min_blur_then_smooth<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
) -> Scratch {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();

    let picked = Scratch::new::<f32>(context, width, height);
    let blur_chain = Scratch::new::<f32>(context, width, height);
    let median = Scratch::new::<f32>(context, width, height);
    let blur_scratch = Scratch::new::<f32>(context, width, height);
    let picked_plane = picked.plane();
    let blur_chain_plane = blur_chain.plane();
    let median_plane = median.plane();
    let blur_scratch_plane = blur_scratch.plane();
    let min_blur_planes = MinBlurPlanes {
        input,
        blurred: blur_chain_plane,
        median: median_plane,
        scratch: blur_scratch_plane,
        output: picked_plane,
    };
    min_blur(client, &sizing, min_blur_planes, 2);

    let blurred = Scratch::new::<f32>(context, width, height);
    let blurred_plane = blurred.plane();
    let smooth_planes = NeighbourPlanes {
        input: picked_plane,
        output: blurred_plane,
    };
    smooth(client, &sizing, smooth_planes, Smooth::Binomial);

    blurred
}

/// Runs havsfunc's YAHR on one f32 plane, limiting its detail to the detail of its warped self.
fn yahr<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &YahrParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();

    let blurred = min_blur_then_smooth(client, context, input);
    let blurred_plane = blurred.plane();
    let detail = Scratch::new::<f32>(context, width, height);
    let detail_plane = detail.plane();
    let detail_planes = DifferencePlanes {
        first: input,
        second: blurred_plane,
        output: detail_plane,
    };
    difference(client, &sizing, detail_planes);

    let warped = Scratch::new::<f32>(context, width, height);
    let warped_plane = warped.plane();
    let warp_params = params.warp_params();
    padded_warp(client, context, input, warped_plane, &warp_params);

    let warped_blurred = min_blur_then_smooth(client, context, warped_plane);
    let warped_blurred_plane = warped_blurred.plane();
    let warped_detail = Scratch::new::<f32>(context, width, height);
    let warped_detail_plane = warped_detail.plane();
    let warped_detail_planes = DifferencePlanes {
        first: warped_plane,
        second: warped_blurred_plane,
        output: warped_detail_plane,
    };
    difference(client, &sizing, warped_detail_planes);

    let limited = Scratch::new::<f32>(context, width, height);
    let limited_plane = limited.plane();
    let limit_planes = RepairPlanes {
        clip: detail_plane,
        reference: warped_detail_plane,
        output: limited_plane,
    };
    repair(client, &sizing, limit_planes, RepairMode::Thirteen);

    let combine_planes = YahrPlanes {
        clip: input,
        detail: detail_plane,
        limited: limited_plane,
        output,
    };
    yahr_combine(client, &sizing, combine_planes);
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} cleans frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

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

/// Removes halos by limiting a clip's detail to the detail of its warp-sharpened self, the way
/// havsfunc's `YAHR` does.
pub(crate) struct Yahr;

impl StageDefinition for Yahr {
    type Args = YahrArgs;

    fn package() -> &'static str {
        "dehalo"
    }

    fn identifier() -> &'static str {
        "yahr"
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

impl FilterDefinition for Yahr {
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
        yahr_params(args, input.sample)?;
        selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

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
        let params = yahr_params(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let client = context.client().clone();
        let element = params.element;
        let peak = params.warp_params().peak() as f32;

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let cleaner = Cleaner {
            client,
            planewise,
            params,
        };

        Ok(Box::new(cleaner))
    }
}

/// Runs `dehalo/yahr` on every selected plane of a frame.
struct Cleaner<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: YahrParams,
}

impl<R: Runtime> PlaneFilter for Cleaner<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        yahr(&self.client, context, input, output, &self.params);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Cleaner<R> {}

impl<R: Runtime> Filter for Cleaner<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        self.planewise.run(&self.client, context, input, self)
    }
}
