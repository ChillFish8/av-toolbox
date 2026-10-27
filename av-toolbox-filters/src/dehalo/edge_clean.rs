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
use av_toolbox_cubek::dehalo::edge_clean::{EdgeCleanParams, levels, offset_difference, threshold_mask};
use av_toolbox_cubek::dehalo::fine::RampPlanes;
use av_toolbox_cubek::dehalo::{DifferencePlanes, MergePlanes, merge_masked};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::neighbourhood::{NeighbourPlanes, Smooth, prewitt, smooth};
use av_toolbox_cubek::rgvs::{
    GrainBorder,
    RemoveGrainMode,
    RemoveGrainPlanes,
    RepairMode,
    RepairPlanes,
    remove_grain,
    repair,
};
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

use crate::bits::bits_of;
use crate::checks::{accepts_integer_or_f32_frames, between_u32};
use crate::codes::CodeValues;
use crate::padded_warp::padded_warp;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "dehalo/edge_clean";

const MAX_STRENGTH: u32 = 250;

fn default_strength() -> u32 {
    10
}

fn default_rep() -> bool {
    true
}

fn default_rmode() -> u32 {
    17
}

/// Arguments for `dehalo/edge_clean`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct EdgeCleanArgs {
    /// How far samples move in the warp.
    #[serde(default = "default_strength")]
    pub(crate) strength: u32,
    /// Whether the warp is repaired against the clip before merging.
    #[serde(default = "default_rep")]
    pub(crate) rep: bool,
    /// The RemoveGrain Repair mode `rep` runs, when set.
    #[serde(default = "default_rmode")]
    pub(crate) rmode: u32,
    /// Whether the particle-cleaning pass runs.
    #[serde(default)]
    pub(crate) smode: bool,
    /// Whether the merged result is repaired against hot pixels.
    #[serde(default)]
    pub(crate) hot: bool,
    /// The planes to clean, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn bad_rmode(rmode: u32) -> StageError {
    let message = format!("{STAGE} does not support rmode {rmode}");

    StageError::msg(message)
}

/// The EdgeCleaner parameters for `sample`, once the arguments are checked.
pub(crate) fn edge_clean_params(
    args: &EdgeCleanArgs,
    sample: SampleFormat,
) -> Result<EdgeCleanParams, StageError> {
    between_u32(STAGE, "strength", args.strength, 0, MAX_STRENGTH)?;

    let Some(rmode) = RepairMode::from_number(args.rmode) else {
        return Err(bad_rmode(args.rmode));
    };

    let bits = bits_of(sample);
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    Ok(EdgeCleanParams {
        strength: args.strength,
        rep: args.rep,
        rmode,
        smode: args.smode,
        hot: args.hot,
        element,
        bits,
    })
}

/// Runs havsfunc's EdgeCleaner particle-cleaning pass into `output`, merging `input` back in
/// wherever small bright particles are found in `result`.
fn smode_clean<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    result: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &EdgeCleanParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let peak = params.peak();

    let smoothed = Scratch::new::<f32>(context, width, height);
    let smoothed_plane = smoothed.plane();
    let smooth_planes = RemoveGrainPlanes {
        input,
        output: smoothed_plane,
    };
    remove_grain(
        client,
        &sizing,
        smooth_planes,
        RemoveGrainMode::Seventeen,
        GrainBorder::Copy,
    );

    let diff = Scratch::new::<f32>(context, width, height);
    let diff_plane = diff.plane();
    let neutral = params.scale(128.0);
    let diff_planes = DifferencePlanes {
        first: input,
        second: smoothed_plane,
        output: diff_plane,
    };
    offset_difference(client, &sizing, diff_planes, neutral);

    let brightened = Scratch::new::<f32>(context, width, height);
    let brightened_plane = brightened.plane();
    let levels_low = params.scale(40.0);
    let levels_span = params.scale(128.0);
    let exponent = 1.0 / 0.35;
    let levels_planes = RampPlanes {
        input: diff_plane,
        output: brightened_plane,
    };
    levels(
        client,
        &sizing,
        levels_planes,
        levels_low,
        levels_span,
        exponent,
        peak,
    );

    let cleaned = Scratch::new::<f32>(context, width, height);
    let cleaned_plane = cleaned.plane();
    let clean_planes = RemoveGrainPlanes {
        input: brightened_plane,
        output: cleaned_plane,
    };
    remove_grain(
        client,
        &sizing,
        clean_planes,
        RemoveGrainMode::Seven,
        GrainBorder::Copy,
    );

    let particle_edges = Scratch::new::<f32>(context, width, height);
    let particle_edges_plane = particle_edges.plane();
    let particle_edge_planes = NeighbourPlanes {
        input: cleaned_plane,
        output: particle_edges_plane,
    };
    prewitt(client, &sizing, particle_edge_planes, peak);

    let particles = Scratch::new::<f32>(context, width, height);
    let particles_plane = particles.plane();
    let threshold_low = params.scale(4.0);
    let particle_high = params.scale(16.0);
    let particle_planes = RampPlanes {
        input: particle_edges_plane,
        output: particles_plane,
    };
    threshold_mask(
        client,
        &sizing,
        particle_planes,
        threshold_low,
        particle_high,
        peak,
        false,
    );

    let merge_planes = MergePlanes {
        clip: result,
        filtered: input,
        mask: particles_plane,
        output,
    };
    merge_masked(client, &sizing, merge_planes, peak);
}

/// Runs havsfunc's EdgeCleaner on one f32 plane, warp-sharpening then repairing away from strong
/// edges.
fn edge_clean<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &EdgeCleanParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let peak = params.peak();

    let warped = Scratch::new::<f32>(context, width, height);
    let warped_plane = warped.plane();
    let warp_params = params.warp_params();
    padded_warp(client, context, input, warped_plane, &warp_params);

    let repaired_scratch = if params.rep {
        let repaired = Scratch::new::<f32>(context, width, height);
        let repaired_plane = repaired.plane();
        let main_repair_planes = RepairPlanes {
            clip: warped_plane,
            reference: input,
            output: repaired_plane,
        };
        repair(client, &sizing, main_repair_planes, params.rmode);

        Some(repaired)
    } else {
        None
    };

    let main_plane = match &repaired_scratch {
        Some(scratch) => scratch.plane(),
        None => warped_plane,
    };

    let edges = Scratch::new::<f32>(context, width, height);
    let edges_plane = edges.plane();
    let edge_planes = NeighbourPlanes {
        input,
        output: edges_plane,
    };
    prewitt(client, &sizing, edge_planes, peak);

    let thresholded = Scratch::new::<f32>(context, width, height);
    let thresholded_plane = thresholded.plane();
    let threshold_low = params.scale(4.0);
    let threshold_high = params.scale(32.0);
    let threshold_planes = RampPlanes {
        input: edges_plane,
        output: thresholded_plane,
    };
    threshold_mask(
        client,
        &sizing,
        threshold_planes,
        threshold_low,
        threshold_high,
        peak,
        true,
    );

    let mask = Scratch::new::<f32>(context, width, height);
    let mask_plane = mask.plane();
    let mask_planes = NeighbourPlanes {
        input: thresholded_plane,
        output: mask_plane,
    };
    smooth(client, &sizing, mask_planes, Smooth::Box);

    let merged_scratch = if params.hot || params.smode {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };

    let merge_target = match &merged_scratch {
        Some(scratch) => scratch.plane(),
        None => output,
    };
    let merge_planes = MergePlanes {
        clip: input,
        filtered: main_plane,
        mask: mask_plane,
        output: merge_target,
    };
    merge_masked(client, &sizing, merge_planes, peak);

    let hot_scratch = if params.hot && params.smode {
        Some(Scratch::new::<f32>(context, width, height))
    } else {
        None
    };

    let after_hot = if params.hot {
        let hot_target = match &hot_scratch {
            Some(scratch) => scratch.plane(),
            None => output,
        };
        let hot_repair_planes = RepairPlanes {
            clip: merge_target,
            reference: input,
            output: hot_target,
        };
        repair(client, &sizing, hot_repair_planes, RepairMode::Two);

        hot_target
    } else {
        merge_target
    };

    if params.smode {
        smode_clean(client, context, input, after_hot, output, params);
    }
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

/// Warp-sharpens and repairs away from strong edges, the way havsfunc's `EdgeCleaner` does.
pub(crate) struct EdgeClean;

impl StageDefinition for EdgeClean {
    type Args = EdgeCleanArgs;

    fn package() -> &'static str {
        "dehalo"
    }

    fn identifier() -> &'static str {
        "edge_clean"
    }

    fn description() -> &'static str {
        "Cleans halos and specks from around edges, as havsfunc's `EdgeCleaner` does.\n\n\
         It warp-sharpens the clip, which thins edges by pulling nearby pixels towards them."
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

impl FilterDefinition for EdgeClean {
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
        edge_clean_params(args, input.sample)?;
        selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([SampleFormat::F32]),
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
        let params = edge_clean_params(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let client = context.client().clone();

        let codes = CodeValues::new(&input, SampleFormat::F32);
        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            codes,
        };
        let cleaner = Cleaner {
            client,
            planewise,
            params,
        };

        Ok(Box::new(cleaner))
    }
}

/// Runs `dehalo/edge_clean` on every selected plane of a frame.
struct Cleaner<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: EdgeCleanParams,
}

impl<R: Runtime> PlaneFilter for Cleaner<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        edge_clean(&self.client, context, input, output, &self.params);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Cleaner<R> {}

impl<R: Runtime> Filter for Cleaner<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        let output = self.planewise.run(&self.client, context, input, self)?;

        Ok(vec![output])
    }
}
