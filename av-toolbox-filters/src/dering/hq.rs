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
use av_toolbox_cubek::dehalo::dering::{
    ContraPlanes,
    DeringParams,
    LimitPlanes,
    RingPlanes,
    contra_limit,
    limit_filter,
    ring_mask,
};
use av_toolbox_cubek::dehalo::edge_clean::threshold_mask;
use av_toolbox_cubek::dehalo::fine::RampPlanes;
use av_toolbox_cubek::dehalo::{DifferencePlanes, MergePlanes, difference, merge_masked};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::hysteresis::{HysteresisPlanes, hysteresis};
use av_toolbox_cubek::labels::Settled;
use av_toolbox_cubek::neighbourhood::{
    Extreme,
    MinBlurPlanes,
    NeighbourPlanes,
    Shape,
    Smooth,
    Spread,
    extreme,
    median,
    min_blur,
    prewitt,
    smooth,
    spread,
};
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
use cubecl::server::ServerError;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::bits_of;
use crate::checks::{at_least, between_f32, between_u32};
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "dering/hq";

fn default_mrad() -> u32 {
    1
}

fn default_msmooth() -> u32 {
    1
}

fn default_mthr() -> f32 {
    60.0
}

fn default_minp() -> u32 {
    1
}

fn default_sharp() -> u32 {
    1
}

fn default_drrep() -> u32 {
    24
}

fn default_thr() -> f32 {
    12.0
}

fn default_elast() -> f32 {
    2.0
}

/// Arguments for `dering/hq`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct HqArgs {
    /// How many square maximum passes grow the ring mask's outer edge.
    #[serde(default = "default_mrad")]
    pub(crate) mrad: u32,
    /// How many inflate passes soften the ring mask's outer edge, after `mrad`.
    #[serde(default = "default_msmooth")]
    pub(crate) msmooth: u32,
    /// Whether the ring mask keeps the edge itself, rather than cutting it out with `minp`.
    #[serde(default)]
    pub(crate) incedge: bool,
    /// Edges weaker than this, on the 8-bit scale, are not part of the ring mask.
    #[serde(default = "default_mthr")]
    pub(crate) mthr: f32,
    /// Which combination of inflate and square minimum passes shrinks the ring mask's inner edge.
    #[serde(default = "default_minp")]
    pub(crate) minp: u32,
    /// The MinBlur radius the smoothing pass runs, picked from the frame size when not given.
    #[serde(default)]
    pub(crate) nrmode: Option<u32>,
    /// How many contrasharpening passes run. 0 skips contrasharpening.
    #[serde(default = "default_sharp")]
    pub(crate) sharp: u32,
    /// The Repair mode the smoothed plane is repaired against the clip with. 0 skips it.
    #[serde(default = "default_drrep")]
    pub(crate) drrep: u32,
    /// How far a darkening sample may move before it is limited back toward the plane.
    #[serde(default = "default_thr")]
    pub(crate) thr: f32,
    /// How far past `thr`/`darkthr` a sample blends rather than falling straight back.
    #[serde(default = "default_elast")]
    pub(crate) elast: f32,
    /// How far a brightening sample may move before it is limited back toward the plane. `thr / 4`
    /// when not given.
    #[serde(default)]
    pub(crate) darkthr: Option<f32>,
    /// The planes to dering, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn bad_drrep(drrep: u32) -> StageError {
    let message = format!("{STAGE} does not support drrep {drrep}");

    StageError::msg(message)
}

fn checked_drrep(drrep: u32) -> Result<Option<RepairMode>, StageError> {
    if drrep == 0 {
        return Ok(None);
    }

    let Some(mode) = RepairMode::from_number(drrep) else {
        return Err(bad_drrep(drrep));
    };

    Ok(Some(mode))
}

/// `nrmode`'s default for a frame of `width` by `height`.
fn default_nrmode(width: u32, height: u32) -> u32 {
    if width > 1024 || height > 576 { 2 } else { 1 }
}

/// The HQDeringmod parameters for `input`, once the arguments are checked.
///
/// It takes the whole frame format because `nrmode`'s default, when not given, depends on the
/// frame size rather than the plane it ends up running on: 2 when the frame is wider than 1024 or
/// taller than 576, otherwise 1.
pub(crate) fn dering_params(args: &HqArgs, input: &FrameFormat) -> Result<DeringParams, StageError> {
    between_u32(STAGE, "mrad", args.mrad, 0, 16)?;
    between_u32(STAGE, "msmooth", args.msmooth, 0, 16)?;
    between_f32(STAGE, "mthr", args.mthr, 0.0, 255.0)?;
    between_u32(STAGE, "minp", args.minp, 0, 4)?;

    let nrmode = match args.nrmode {
        Some(value) => value,
        None => default_nrmode(input.width, input.height),
    };
    between_u32(STAGE, "nrmode", nrmode, 1, 3)?;

    between_u32(STAGE, "sharp", args.sharp, 0, 3)?;

    let drrep = checked_drrep(args.drrep)?;

    at_least(STAGE, "thr", args.thr, 0.0)?;
    at_least(STAGE, "elast", args.elast, 1.0)?;

    let darkthr = args.darkthr.unwrap_or(args.thr / 4.0);
    at_least(STAGE, "darkthr", darkthr, 0.0)?;

    let bits = bits_of(input.sample);

    Ok(DeringParams {
        mrad: args.mrad,
        msmooth: args.msmooth,
        incedge: args.incedge,
        mthr: args.mthr,
        minp: args.minp,
        nrmode,
        sharp: args.sharp,
        drrep,
        thr: args.thr,
        darkthr,
        elast: args.elast,
        bits,
    })
}

/// One pass of a chain run through [ping_pong].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Step {
    Maximum,
    Minimum,
    Inflate,
    Box,
}

/// Runs `steps` from `input` into `output`, through two ping-pong scratch planes.
fn ping_pong<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    steps: &[Step],
) {
    debug_assert!(!steps.is_empty());

    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let first = Scratch::new::<f32>(context, width, height);
    let second = Scratch::new::<f32>(context, width, height);
    let first_plane = first.plane();
    let second_plane = second.plane();

    let mut source = input;
    for (index, step) in steps.iter().enumerate() {
        let last = index + 1 == steps.len();
        let target = if last {
            output
        } else if index % 2 == 0 {
            first_plane
        } else {
            second_plane
        };

        let planes = NeighbourPlanes {
            input: source,
            output: target,
        };
        match step {
            Step::Maximum => extreme(client, &sizing, planes, Extreme::Maximum, Shape::Square),
            Step::Minimum => extreme(client, &sizing, planes, Extreme::Minimum, Shape::Square),
            Step::Inflate => spread(client, &sizing, planes, Spread::Inflate),
            Step::Box => smooth(client, &sizing, planes, Smooth::Box),
        }

        source = target;
    }
}

/// The ring mask's outer edge, `mrad` square maximum passes then `msmooth` inflate passes.
fn outer_steps(mrad: u32, msmooth: u32) -> Vec<Step> {
    let mut steps = vec![Step::Maximum; mrad as usize];
    steps.extend(vec![Step::Inflate; msmooth as usize]);

    steps
}

/// The ring mask's inner edge for `minp`.
fn inner_steps(minp: u32) -> Vec<Step> {
    match minp {
        0 => Vec::new(),
        1 => vec![Step::Inflate, Step::Minimum],
        2 => vec![Step::Minimum],
        3 => vec![Step::Inflate, Step::Minimum, Step::Minimum],
        4 => vec![Step::Minimum, Step::Minimum],
        _ => unreachable!("minp is bounded to 0..=4 by dering_params"),
    }
}

/// Contrasharpens `min_blurred` against `input`, `sharp` passes deep, into `output`.
fn contra_sharpen<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    min_blurred: DevicePlane<'_>,
    output: DevicePlane<'_>,
    sharp: u32,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();

    let pre = Scratch::new::<f32>(context, width, height);
    let pre_plane = pre.plane();
    let pre_planes = NeighbourPlanes {
        input: min_blurred,
        output: pre_plane,
    };
    median(client, &sizing, pre_planes, 1);

    let binomial = Scratch::new::<f32>(context, width, height);
    let binomial_plane = binomial.plane();
    let binomial_planes = NeighbourPlanes {
        input: pre_plane,
        output: binomial_plane,
    };
    smooth(client, &sizing, binomial_planes, Smooth::Binomial);

    let box_passes = sharp.saturating_sub(1);
    let method_scratch = if box_passes > 0 {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };
    let method_plane = match &method_scratch {
        Some(scratch) => {
            let target = scratch.plane();
            let steps = vec![Step::Box; box_passes as usize];
            ping_pong(client, context, binomial_plane, target, &steps);

            target
        },
        None => binomial_plane,
    };

    let sharpened = Scratch::new::<f32>(context, width, height);
    let sharpened_plane = sharpened.plane();
    let sharpened_diff_planes = DifferencePlanes {
        first: pre_plane,
        second: method_plane,
        output: sharpened_plane,
    };
    difference(client, &sizing, sharpened_diff_planes);

    let removed = Scratch::new::<f32>(context, width, height);
    let removed_plane = removed.plane();
    let removed_diff_planes = DifferencePlanes {
        first: input,
        second: min_blurred,
        output: removed_plane,
    };
    difference(client, &sizing, removed_diff_planes);

    let repaired = Scratch::new::<f32>(context, width, height);
    let repaired_plane = repaired.plane();
    let repair_planes = RepairPlanes {
        clip: sharpened_plane,
        reference: removed_plane,
        output: repaired_plane,
    };
    repair(client, &sizing, repair_planes, RepairMode::One);

    let contra_planes = ContraPlanes {
        smoothed: min_blurred,
        sharpened: sharpened_plane,
        repaired: repaired_plane,
        output,
    };
    contra_limit(client, &sizing, contra_planes);
}

fn unreadable_labels(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read back its ring mask labels");

    StageError::with_source(message, error)
}

/// The seed-and-keep hysteresis pass that joins the ring mask's edges, `pm_median` seeding
/// `pm`'s 8-connected components.
fn ring_edges<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    pm_median: DevicePlane<'_>,
    pm: DevicePlane<'_>,
    output: DevicePlane<'_>,
    peak: f32,
) -> Result<(), StageError> {
    let sizing = Sizing(context);
    let width = pm.width();
    let height = pm.height();

    let marks = Scratch::new::<u8>(context, width, height);
    let seeds = Scratch::new::<u8>(context, width, height);
    let kept = Scratch::new::<u8>(context, width, height);
    let labels = Scratch::new::<u32>(context, width, height);
    let counts = Scratch::new::<u32>(context, width, height);
    let changed = Scratch::new::<u32>(context, 1, 1);

    let hysteresis_planes = HysteresisPlanes {
        seeds_from: pm_median,
        region: pm,
        marks: marks.plane(),
        seeds: seeds.plane(),
        labels: labels.plane(),
        changed: changed.plane(),
        counts: counts.plane(),
        kept: kept.plane(),
        output,
    };
    let settled = hysteresis(client, &sizing, hysteresis_planes, peak).map_err(unreadable_labels)?;

    if let Settled::Capped { passes } = settled {
        tracing::warn!("{STAGE} stopped joining ring edges after {passes} passes, so some may be split");
    }

    Ok(())
}

/// Runs havsfunc's HQDeringmod on one f32 plane.
fn hq_dering<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &DeringParams,
) -> Result<(), StageError> {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let peak = params.peak();

    let blurred = Scratch::new::<f32>(context, width, height);
    let min_median = Scratch::new::<f32>(context, width, height);
    let min_blur_scratch = Scratch::new::<f32>(context, width, height);
    let min_blurred = Scratch::new::<f32>(context, width, height);
    let min_blurred_plane = min_blurred.plane();
    let min_blur_planes = MinBlurPlanes {
        input,
        blurred: blurred.plane(),
        median: min_median.plane(),
        scratch: min_blur_scratch.plane(),
        output: min_blurred_plane,
    };
    min_blur(client, &sizing, min_blur_planes, params.nrmode);

    let smoothed_scratch = if params.sharp > 0 {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };
    let smoothed_plane = match &smoothed_scratch {
        Some(scratch) => {
            let target = scratch.plane();
            contra_sharpen(client, context, input, min_blurred_plane, target, params.sharp);

            target
        },
        None => min_blurred_plane,
    };

    let repaired_scratch = params
        .drrep
        .map(|mode| (Scratch::new::<f32>(context, width, height), mode));
    let repaired_plane = match &repaired_scratch {
        Some((scratch, mode)) => {
            let target = scratch.plane();
            let repair_planes = RepairPlanes {
                clip: input,
                reference: smoothed_plane,
                output: target,
            };
            repair(client, &sizing, repair_planes, *mode);

            target
        },
        None => smoothed_plane,
    };

    let limited = Scratch::new::<f32>(context, width, height);
    let limited_plane = limited.plane();
    let limit_planes = LimitPlanes {
        plane: input,
        filtered: repaired_plane,
        output: limited_plane,
    };
    let thr = params.scale(params.thr);
    let darkthr = params.scale(params.darkthr);
    limit_filter(client, &sizing, limit_planes, thr, darkthr, params.elast);

    let edges = Scratch::new::<f32>(context, width, height);
    let edges_plane = edges.plane();
    let edge_planes = NeighbourPlanes {
        input,
        output: edges_plane,
    };
    prewitt(client, &sizing, edge_planes, peak);

    let pm = Scratch::new::<f32>(context, width, height);
    let pm_plane = pm.plane();
    let floor = params.scale(params.mthr);
    let pm_planes = RampPlanes {
        input: edges_plane,
        output: pm_plane,
    };
    threshold_mask(client, &sizing, pm_planes, floor, f32::MAX, peak, false);

    let pm_median = Scratch::new::<f32>(context, width, height);
    let pm_median_plane = pm_median.plane();
    let pm_median_planes = NeighbourPlanes {
        input: pm_plane,
        output: pm_median_plane,
    };
    median(client, &sizing, pm_median_planes, 1);

    let fmask = Scratch::new::<f32>(context, width, height);
    let fmask_plane = fmask.plane();
    ring_edges(client, context, pm_median_plane, pm_plane, fmask_plane, peak)?;

    let outer_scratch = if params.mrad > 0 || params.msmooth > 0 {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };
    let outer_plane = match &outer_scratch {
        Some(scratch) => {
            let target = scratch.plane();
            let steps = outer_steps(params.mrad, params.msmooth);
            ping_pong(client, context, fmask_plane, target, &steps);

            target
        },
        None => fmask_plane,
    };

    let inner_scratch = if params.minp > 0 {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };
    let inner_plane = match &inner_scratch {
        Some(scratch) => {
            let target = scratch.plane();
            let steps = inner_steps(params.minp);
            ping_pong(client, context, fmask_plane, target, &steps);

            target
        },
        None => fmask_plane,
    };

    let ring_mask_scratch = Scratch::new::<f32>(context, width, height);
    let ring_mask_plane = ring_mask_scratch.plane();
    let ring_planes = RingPlanes {
        outer: outer_plane,
        inner: inner_plane,
        output: ring_mask_plane,
    };
    ring_mask(client, &sizing, ring_planes, params.incedge, peak);

    let merge_planes = MergePlanes {
        clip: input,
        filtered: limited_plane,
        mask: ring_mask_plane,
        output,
    };
    merge_masked(client, &sizing, merge_planes, peak);

    Ok(())
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} derings frames, not masks");

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

/// Smooths ringing inside a mask of the area around edges, the way havsfunc's `HQDeringmod` does.
pub(crate) struct Hq;

impl StageDefinition for Hq {
    type Args = HqArgs;

    fn package() -> &'static str {
        "dering"
    }

    fn identifier() -> &'static str {
        "hq"
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

impl FilterDefinition for Hq {
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
        dering_params(args, &input)?;
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
        let params = dering_params(&args, &input)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let client = context.client().clone();
        let element = if params.bits == 8 {
            Element::U8
        } else {
            Element::U16
        };
        let peak = params.peak();

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let deringer = Deringer {
            client,
            planewise,
            params,
        };

        Ok(Box::new(deringer))
    }
}

/// Runs `dering/hq` on every selected plane of a frame.
struct Deringer<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: DeringParams,
}

impl<R: Runtime> PlaneFilter for Deringer<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        hq_dering(&self.client, context, input, output, &self.params)
    }
}

impl<R: Runtime> StageInstance for Deringer<R> {}

impl<R: Runtime> Filter for Deringer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        let output = self.planewise.run(&self.client, context, input, self)?;

        Ok(vec![output])
    }
}
