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
use av_toolbox_cubek::dehalo::alpha::AlphaParams;
use av_toolbox_cubek::dehalo::fine::{
    ContrastPlanes,
    FineParams,
    OutsidePlanes,
    RampPlanes,
    contrasharpen,
    outside_mask,
    ramp,
    scale_values,
};
use av_toolbox_cubek::dehalo::{MergePlanes, merge_masked};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::neighbourhood::{
    Expansion,
    Extreme,
    NeighbourPlanes,
    Shape,
    Smooth,
    expansion_passes,
    extreme,
    median,
    prewitt,
    smooth,
};
use av_toolbox_cubek::rgvs::{RepairMode, RepairPlanes, repair};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
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

use super::alpha::{
    AlphaTables,
    at_least,
    checked_alpha_sizes,
    checked_plane_sizes,
    default_radius,
    default_strength,
    dehalo_alpha,
};
use crate::bits::bits_of;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "dehalo/fine";

fn default_thmi() -> f32 {
    80.0
}

fn default_thma() -> f32 {
    128.0
}

fn default_thlimi() -> f32 {
    50.0
}

fn default_thlima() -> f32 {
    100.0
}

fn default_excl() -> bool {
    true
}

/// Arguments for `dehalo/fine`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct FineArgs {
    /// How far a halo reaches across, in samples.
    #[serde(default = "default_radius")]
    pub(crate) rx: f32,
    /// How far a halo reaches down, in samples. `rx` when not given.
    #[serde(default)]
    pub(crate) ry: Option<f32>,
    /// Edges weaker than this, on the 8-bit scale, are not strong edges.
    #[serde(default = "default_thmi")]
    pub(crate) thmi: f32,
    /// Edges stronger than this are fully strong edges.
    #[serde(default = "default_thma")]
    pub(crate) thma: f32,
    /// Edges weaker than this are not light edges.
    #[serde(default = "default_thlimi")]
    pub(crate) thlimi: f32,
    /// Edges stronger than this are fully light edges.
    #[serde(default = "default_thlima")]
    pub(crate) thlima: f32,
    /// How strongly dark halos are removed.
    #[serde(default = "default_strength")]
    pub(crate) darkstr: f32,
    /// How strongly bright halos are removed.
    #[serde(default = "default_strength")]
    pub(crate) brightstr: f32,
    /// How far the result is contrasharpened back toward the clip. 0 skips it.
    #[serde(default)]
    pub(crate) contra: f32,
    /// Whether light edges are kept out of the mask as well as strong ones.
    #[serde(default = "default_excl")]
    pub(crate) excl: bool,
    /// How much of the strong edges themselves are dehaloed.
    #[serde(default)]
    pub(crate) edgeproc: f32,
    /// The planes to dehalo, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn crossed_thresholds(low_name: &str, low: f32, high_name: &str, high: f32) -> Result<(), StageError> {
    if low < high {
        return Ok(());
    }

    let message = format!("{STAGE} needs {low_name} below {high_name}, got {low} and {high}");

    Err(StageError::msg(message))
}

fn threshold(name: &str, value: f32) -> Result<(), StageError> {
    if (0.0..=255.0).contains(&value) {
        return Ok(());
    }

    let message = format!("{STAGE} needs {name} between 0 and 255, got {value}");

    Err(StageError::msg(message))
}

/// The fine parameters for `sample`, once the arguments are checked.
pub(crate) fn fine_params(args: &FineArgs, sample: SampleFormat) -> Result<FineParams, StageError> {
    at_least(STAGE, "rx", args.rx, 1.0)?;
    let ry = args.ry.unwrap_or(args.rx);
    at_least(STAGE, "ry", ry, 1.0)?;
    at_least(STAGE, "darkstr", args.darkstr, 0.0)?;
    at_least(STAGE, "brightstr", args.brightstr, 0.0)?;
    at_least(STAGE, "contra", args.contra, 0.0)?;
    at_least(STAGE, "edgeproc", args.edgeproc, 0.0)?;

    threshold("thmi", args.thmi)?;
    threshold("thma", args.thma)?;
    threshold("thlimi", args.thlimi)?;
    threshold("thlima", args.thlima)?;
    crossed_thresholds("thmi", args.thmi, "thma", args.thma)?;
    crossed_thresholds("thlimi", args.thlimi, "thlima", args.thlima)?;

    let bits = bits_of(sample);

    let alpha = AlphaParams {
        rx: args.rx,
        ry,
        darkstr: args.darkstr,
        brightstr: args.brightstr,
        lowsens: 50.0,
        highsens: 50.0,
        ss: 1.5,
        bits,
    };

    Ok(FineParams {
        alpha,
        thmi: args.thmi,
        thma: args.thma,
        thlimi: args.thlimi,
        thlima: args.thlima,
        contra: args.contra,
        excl: args.excl,
        edgeproc: args.edgeproc,
    })
}

/// Runs a chain of maximum or minimum passes from `input` into `output`, through two scratch planes.
fn expand_into<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    passes: &[Shape],
    which: Extreme,
) {
    debug_assert!(!passes.is_empty());

    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let first = Scratch::new::<f32>(context, width, height);
    let second = Scratch::new::<f32>(context, width, height);
    let first_plane = first.plane();
    let second_plane = second.plane();

    let mut source = input;
    for (index, shape) in passes.iter().enumerate() {
        let last = index + 1 == passes.len();
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
        extreme(client, &sizing, planes, which, *shape);
        source = target;
    }
}

/// Runs `dehalo/fine` on one f32 plane.
fn fine_dehalo<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    tables: &AlphaTables,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &FineParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let peak = params.alpha.peak();
    let (across, down) = params.radii();

    let dehaloed = Scratch::new::<f32>(context, width, height);
    let dehaloed_plane = dehaloed.plane();
    dehalo_alpha(client, context, tables, input, dehaloed_plane, &params.alpha);

    let mut contrasharpened_scratch = None;
    if params.contra > 0.0 {
        let blurred = Scratch::new::<f32>(context, width, height);
        let blurred_plane = blurred.plane();
        let blur_planes = NeighbourPlanes {
            input: dehaloed_plane,
            output: blurred_plane,
        };
        smooth(client, &sizing, blur_planes, Smooth::Binomial);

        let median_plane_scratch = Scratch::new::<f32>(context, width, height);
        let median_plane = median_plane_scratch.plane();
        let median_planes = NeighbourPlanes {
            input: blurred_plane,
            output: median_plane,
        };
        median(client, &sizing, median_planes, 2);

        let first = Scratch::new::<f32>(context, width, height);
        let first_plane = first.plane();
        let first_repair_planes = RepairPlanes {
            clip: blurred_plane,
            reference: median_plane,
            output: first_plane,
        };
        repair(client, &sizing, first_repair_planes, RepairMode::One);

        let repaired = Scratch::new::<f32>(context, width, height);
        let repaired_plane = repaired.plane();
        let second_repair_planes = RepairPlanes {
            clip: blurred_plane,
            reference: first_plane,
            output: repaired_plane,
        };
        repair(client, &sizing, second_repair_planes, RepairMode::One);

        let contrasharpened = Scratch::new::<f32>(context, width, height);
        let contrasharpened_plane = contrasharpened.plane();
        let contrast_planes = ContrastPlanes {
            dehaloed: dehaloed_plane,
            blurred: blurred_plane,
            repaired: repaired_plane,
            clip: input,
            output: contrasharpened_plane,
        };
        contrasharpen(client, &sizing, contrast_planes, params.contra);

        contrasharpened_scratch = Some(contrasharpened);
    }

    let sharpened = match &contrasharpened_scratch {
        Some(scratch) => scratch.plane(),
        None => dehaloed_plane,
    };

    let edges = Scratch::new::<f32>(context, width, height);
    let edges_plane = edges.plane();
    let edge_planes = NeighbourPlanes {
        input,
        output: edges_plane,
    };
    prewitt(client, &sizing, edge_planes, peak);

    let strong = Scratch::new::<f32>(context, width, height);
    let strong_plane = strong.plane();
    let strong_low = params.alpha.scale(params.thmi);
    let strong_range = params.thma - params.thmi;
    let strong_ramp_planes = RampPlanes {
        input: edges_plane,
        output: strong_plane,
    };
    ramp(
        client,
        &sizing,
        strong_ramp_planes,
        strong_low,
        strong_range,
        peak,
    );

    let light = Scratch::new::<f32>(context, width, height);
    let light_plane = light.plane();
    let light_low = params.alpha.scale(params.thlimi);
    let light_range = params.thlima - params.thlimi;
    let light_ramp_planes = RampPlanes {
        input: edges_plane,
        output: light_plane,
    };
    ramp(client, &sizing, light_ramp_planes, light_low, light_range, peak);

    let rectangle = expansion_passes(across, down, Expansion::Rectangle);
    let large = Scratch::new::<f32>(context, width, height);
    let large_plane = large.plane();
    expand_into(
        client,
        context,
        strong_plane,
        large_plane,
        &rectangle,
        Extreme::Maximum,
    );

    let ellipse = expansion_passes(across, down, Expansion::Ellipse);
    let grown = Scratch::new::<f32>(context, width, height);
    let grown_plane = grown.plane();
    expand_into(
        client,
        context,
        light_plane,
        grown_plane,
        &ellipse,
        Extreme::Maximum,
    );

    let brightened = Scratch::new::<f32>(context, width, height);
    let brightened_plane = brightened.plane();
    let brighten_planes = RampPlanes {
        input: grown_plane,
        output: brightened_plane,
    };
    scale_values(client, &sizing, brighten_planes, 4.0, peak);

    let thinned = Scratch::new::<f32>(context, width, height);
    let thinned_plane = thinned.plane();
    expand_into(
        client,
        context,
        brightened_plane,
        thinned_plane,
        &ellipse,
        Extreme::Minimum,
    );

    let once = Scratch::new::<f32>(context, width, height);
    let once_plane = once.plane();
    let once_planes = NeighbourPlanes {
        input: thinned_plane,
        output: once_plane,
    };
    smooth(client, &sizing, once_planes, Smooth::Box);

    let shrink = Scratch::new::<f32>(context, width, height);
    let shrink_plane = shrink.plane();
    let shrink_planes = NeighbourPlanes {
        input: once_plane,
        output: shrink_plane,
    };
    smooth(client, &sizing, shrink_planes, Smooth::Box);

    let band = Scratch::new::<f32>(context, width, height);
    let band_plane = band.plane();
    let outside_planes = OutsidePlanes {
        large: large_plane,
        strong: strong_plane,
        shrink: shrink_plane,
        output: band_plane,
    };
    outside_mask(
        client,
        &sizing,
        outside_planes,
        params.excl,
        params.edgeproc,
        peak,
    );

    let blurred_band = Scratch::new::<f32>(context, width, height);
    let blurred_band_plane = blurred_band.plane();
    let blurred_band_planes = NeighbourPlanes {
        input: band_plane,
        output: blurred_band_plane,
    };
    smooth(client, &sizing, blurred_band_planes, Smooth::Box);

    let outside = Scratch::new::<f32>(context, width, height);
    let outside_plane = outside.plane();
    let outside_scale_planes = RampPlanes {
        input: blurred_band_plane,
        output: outside_plane,
    };
    scale_values(client, &sizing, outside_scale_planes, 2.0, peak);

    let merge_planes = MergePlanes {
        clip: input,
        filtered: sharpened,
        mask: outside_plane,
        output,
    };
    merge_masked(client, &sizing, merge_planes, peak);
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} dehaloes frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_dimensions(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

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

/// Removes halos while running `dehalo/alpha` only in a band around strong edges, the way
/// havsfunc's `FineDehalo` does.
pub(crate) struct Fine;

impl StageDefinition for Fine {
    type Args = FineArgs;

    fn package() -> &'static str {
        "dehalo"
    }

    fn identifier() -> &'static str {
        "fine"
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

impl FilterDefinition for Fine {
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
        let params = fine_params(args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
        checked_plane_sizes(STAGE, &input, &selected, &params.alpha)?;

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
        let params = fine_params(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let client = context.client().clone();
        let bits = params.alpha.bits;
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = params.alpha.peak();

        let mut tables = Vec::with_capacity(input.channels().len());
        for (plane_index, channel) in input.channels().iter().enumerate() {
            if !selected[plane_index] {
                tables.push(None);
                continue;
            }

            let Some((width, height)) = input.channel_dimensions(*channel) else {
                return Err(missing_dimensions(*channel));
            };
            let sizes = checked_alpha_sizes(STAGE, *channel, width, height, &params.alpha)?;
            let plane_tables = AlphaTables::new(&client, width, height, &sizes);
            tables.push(Some(plane_tables));
        }

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let dehaloer = FineDehaloer {
            client,
            planewise,
            params,
            tables,
        };

        Ok(Box::new(dehaloer))
    }
}

/// Runs `dehalo/fine` on every selected plane of a frame.
struct FineDehaloer<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: FineParams,
    /// One set of tables per plane, for the selected planes only.
    tables: Vec<Option<AlphaTables>>,
}

fn missing_tables(plane_index: usize) -> StageError {
    let message = format!("{STAGE} has no resize tables for plane {plane_index}");

    StageError::msg(message)
}

impl<R: Runtime> PlaneFilter for FineDehaloer<R> {
    fn filter(
        &self,
        context: &FrameContext,
        plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let Some(tables) = &self.tables[plane_index] else {
            return Err(missing_tables(plane_index));
        };

        fine_dehalo(&self.client, context, tables, input, output, &self.params);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for FineDehaloer<R> {}

impl<R: Runtime> Filter for FineDehaloer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        self.planewise.run(&self.client, context, input, self)
    }
}
