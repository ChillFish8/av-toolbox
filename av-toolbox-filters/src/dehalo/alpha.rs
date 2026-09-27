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
use av_toolbox_cubek::dehalo::alpha::{
    AlphaParams,
    ClampPlanes,
    HaloPlanes,
    StrengthPlanes,
    alpha_sizes,
    apply_strength,
    blend_halo_mask,
    clamp_between,
};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::neighbourhood::{Extreme, NeighbourPlanes, Shape, extreme};
use av_toolbox_cubek::resize::{Kernel, ResizePlanes, Weights, apply_float, centred_taps, upload};
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

use crate::bits::bits_of;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "dehalo/alpha";

fn default_radius() -> f32 {
    2.0
}

fn default_strength() -> f32 {
    1.0
}

fn default_sensitivity() -> f32 {
    50.0
}

fn default_supersampling() -> f32 {
    1.5
}

/// Arguments for `dehalo/alpha`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct AlphaArgs {
    /// How far a halo reaches across, in samples.
    #[serde(default = "default_radius")]
    pub(crate) rx: f32,
    /// How far a halo reaches down, in samples.
    #[serde(default = "default_radius")]
    pub(crate) ry: f32,
    /// How strongly dark halos are removed.
    #[serde(default = "default_strength")]
    pub(crate) darkstr: f32,
    /// How strongly bright halos are removed.
    #[serde(default = "default_strength")]
    pub(crate) brightstr: f32,
    /// Raising this protects more low-contrast detail.
    #[serde(default = "default_sensitivity")]
    pub(crate) lowsens: f32,
    /// Raising this removes more high-contrast halos.
    #[serde(default = "default_sensitivity")]
    pub(crate) highsens: f32,
    /// How far the final clamp supersamples. 1 clamps without resizing.
    #[serde(default = "default_supersampling")]
    pub(crate) ss: f32,
    /// The planes to dehalo, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn below_minimum(name: &str, value: f32, minimum: f32) -> StageError {
    let message = format!("{STAGE} needs a finite {name} of at least {minimum}, got {value}");

    StageError::msg(message)
}

fn out_of_range(name: &str, value: f32) -> StageError {
    let message = format!("{STAGE} needs {name} between 0 and 100, got {value}");

    StageError::msg(message)
}

fn at_least(name: &str, value: f32, minimum: f32) -> Result<(), StageError> {
    if value.is_finite() && value >= minimum {
        return Ok(());
    }

    Err(below_minimum(name, value, minimum))
}

fn percentage(name: &str, value: f32) -> Result<(), StageError> {
    if (0.0..=100.0).contains(&value) {
        return Ok(());
    }

    Err(out_of_range(name, value))
}

/// The alpha parameters for `sample`, once the arguments are checked.
pub(crate) fn alpha_params(args: &AlphaArgs, sample: SampleFormat) -> Result<AlphaParams, StageError> {
    at_least("rx", args.rx, 1.0)?;
    at_least("ry", args.ry, 1.0)?;
    at_least("darkstr", args.darkstr, 0.0)?;
    at_least("brightstr", args.brightstr, 0.0)?;
    percentage("lowsens", args.lowsens)?;
    percentage("highsens", args.highsens)?;
    at_least("ss", args.ss, 1.0)?;

    let bits = bits_of(sample);

    Ok(AlphaParams {
        rx: args.rx,
        ry: args.ry,
        darkstr: args.darkstr,
        brightstr: args.brightstr,
        lowsens: args.lowsens,
        highsens: args.highsens,
        ss: args.ss,
        bits,
    })
}

const MITCHELL: Kernel = Kernel::Bicubic {
    b: 1.0 / 3.0,
    c: 1.0 / 3.0,
};

const SOFT_CUBIC: Kernel = Kernel::Bicubic { b: 1.0, c: 0.0 };

const LANCZOS: Kernel = Kernel::Lanczos { taps: 3 };

/// One resize's tables, across then down.
struct Resize {
    across: Weights,
    down: Weights,
}

impl Resize {
    fn new<R: Runtime>(client: &ComputeClient<R>, from: (u32, u32), to: (u32, u32), kernel: &Kernel) -> Self {
        let across_taps = centred_taps(from.0, to.0, kernel);
        let down_taps = centred_taps(from.1, to.1, kernel);
        let across = upload(client, &across_taps);
        let down = upload(client, &down_taps);

        Self { across, down }
    }
}

/// The supersampled clamp's tables.
struct Supersampled {
    size: (u32, u32),
    clip_up: Resize,
    mask_up: Resize,
    back_down: Resize,
}

/// Every table `dehalo/alpha` needs for one plane size.
pub(super) struct AlphaTables {
    small: (u32, u32),
    down: Resize,
    up: Resize,
    supersampled: Option<Supersampled>,
}

impl AlphaTables {
    pub(super) fn new<R: Runtime>(
        client: &ComputeClient<R>,
        width: u32,
        height: u32,
        params: &AlphaParams,
    ) -> Self {
        let full = (width, height);
        let sizes = alpha_sizes(width, height, params);
        let down = Resize::new(client, full, sizes.small, &MITCHELL);
        let up = Resize::new(client, sizes.small, full, &SOFT_CUBIC);

        let supersampled = sizes.supersampled.map(|large| Supersampled {
            size: large,
            clip_up: Resize::new(client, full, large, &LANCZOS),
            mask_up: Resize::new(client, full, large, &MITCHELL),
            back_down: Resize::new(client, large, full, &LANCZOS),
        });

        Self {
            small: sizes.small,
            down,
            up,
            supersampled,
        }
    }
}

/// Resizes `input` into `output` through `tables`, with a fresh scratch plane between the passes.
fn resize_into<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    tables: &Resize,
) {
    let sizing = Sizing(context);
    let rows = Scratch::new::<f32>(context, output.width(), input.height());
    let rows_plane = rows.plane();
    let planes = ResizePlanes {
        input,
        scratch: rows_plane,
        output,
    };

    apply_float(client, &sizing, planes, &tables.across, &tables.down);
}

/// Writes one plane's 3x3 maximum or minimum into `output`.
fn extreme_into<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    which: Extreme,
) {
    let sizing = Sizing(context);
    let planes = NeighbourPlanes { input, output };

    extreme(client, &sizing, planes, which, Shape::Square);
}

/// Runs `dehalo/alpha` on one f32 plane.
pub(super) fn dehalo_alpha<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    tables: &AlphaTables,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &AlphaParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let (small_width, small_height) = tables.small;

    let small = Scratch::new::<f32>(context, small_width, small_height);
    let halos = Scratch::new::<f32>(context, width, height);
    let small_plane = small.plane();
    let halos_plane = halos.plane();
    resize_into(client, context, input, small_plane, &tables.down);
    resize_into(client, context, small_plane, halos_plane, &tables.up);

    let clip_max = Scratch::new::<f32>(context, width, height);
    let clip_min = Scratch::new::<f32>(context, width, height);
    let halo_max = Scratch::new::<f32>(context, width, height);
    let halo_min = Scratch::new::<f32>(context, width, height);
    let clip_max_plane = clip_max.plane();
    let clip_min_plane = clip_min.plane();
    let halo_max_plane = halo_max.plane();
    let halo_min_plane = halo_min.plane();
    extreme_into(client, context, input, clip_max_plane, Extreme::Maximum);
    extreme_into(client, context, input, clip_min_plane, Extreme::Minimum);
    extreme_into(client, context, halos_plane, halo_max_plane, Extreme::Maximum);
    extreme_into(client, context, halos_plane, halo_min_plane, Extreme::Minimum);

    let lets = Scratch::new::<f32>(context, width, height);
    let lets_plane = lets.plane();
    let halo_planes = HaloPlanes {
        clip: input,
        halos: halos_plane,
        clip_max: clip_max_plane,
        clip_min: clip_min_plane,
        halo_max: halo_max_plane,
        halo_min: halo_min_plane,
        output: lets_plane,
    };
    blend_halo_mask(client, &sizing, halo_planes, params);

    let remove = Scratch::new::<f32>(context, width, height);
    let remove_plane = remove.plane();
    match &tables.supersampled {
        None => {
            let planes = RepairPlanes {
                clip: input,
                reference: lets_plane,
                output: remove_plane,
            };
            repair(client, &sizing, planes, RepairMode::One);
        },
        Some(large) => supersampled_clamp(client, context, large, input, lets_plane, remove_plane),
    }

    let strength_planes = StrengthPlanes {
        clip: input,
        remove: remove_plane,
        output,
    };
    apply_strength(client, &sizing, strength_planes, params);
}

/// Clamps the upscaled clip between the upscaled local extremes of `lets`, then scales it back.
fn supersampled_clamp<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    tables: &Supersampled,
    input: DevicePlane<'_>,
    lets: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let (large_width, large_height) = tables.size;

    let lets_max = Scratch::new::<f32>(context, width, height);
    let lets_min = Scratch::new::<f32>(context, width, height);
    let lets_max_plane = lets_max.plane();
    let lets_min_plane = lets_min.plane();
    extreme_into(client, context, lets, lets_max_plane, Extreme::Maximum);
    extreme_into(client, context, lets, lets_min_plane, Extreme::Minimum);

    let clip_up = Scratch::new::<f32>(context, large_width, large_height);
    let highest = Scratch::new::<f32>(context, large_width, large_height);
    let lowest = Scratch::new::<f32>(context, large_width, large_height);
    let clip_up_plane = clip_up.plane();
    let highest_plane = highest.plane();
    let lowest_plane = lowest.plane();
    resize_into(client, context, input, clip_up_plane, &tables.clip_up);
    resize_into(client, context, lets_max_plane, highest_plane, &tables.mask_up);
    resize_into(client, context, lets_min_plane, lowest_plane, &tables.mask_up);

    let clamped = Scratch::new::<f32>(context, large_width, large_height);
    let clamped_plane = clamped.plane();
    let clamp_planes = ClampPlanes {
        clip: clip_up_plane,
        highest: highest_plane,
        lowest: lowest_plane,
        output: clamped_plane,
    };
    clamp_between(client, &sizing, clamp_planes);

    resize_into(client, context, clamped_plane, output, &tables.back_down);
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

/// Removes halos by pulling pixels toward a halo-free estimate made by resizing, the way
/// havsfunc's `DeHalo_alpha` does.
pub(crate) struct Alpha;

impl StageDefinition for Alpha {
    type Args = AlphaArgs;

    fn package() -> &'static str {
        "dehalo"
    }

    fn identifier() -> &'static str {
        "alpha"
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

impl FilterDefinition for Alpha {
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
        alpha_params(args, input.sample)?;
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
        let params = alpha_params(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;

        let client = context.client().clone();
        let bits = params.bits;
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = params.peak();

        let mut tables = Vec::with_capacity(input.channels().len());
        for (plane_index, channel) in input.channels().iter().enumerate() {
            if !selected[plane_index] {
                tables.push(None);
                continue;
            }

            let Some((width, height)) = input.channel_dimensions(*channel) else {
                return Err(missing_dimensions(*channel));
            };
            let plane_tables = AlphaTables::new(&client, width, height, &params);
            tables.push(Some(plane_tables));
        }

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let dehaloer = Dehaloer {
            client,
            planewise,
            params,
            tables,
        };

        Ok(Box::new(dehaloer))
    }
}

/// Runs `dehalo/alpha` on every selected plane of a frame.
struct Dehaloer<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: AlphaParams,
    /// One set of tables per plane, for the selected planes only.
    tables: Vec<Option<AlphaTables>>,
}

fn missing_tables(plane_index: usize) -> StageError {
    let message = format!("{STAGE} has no resize tables for plane {plane_index}");

    StageError::msg(message)
}

impl<R: Runtime> PlaneFilter for Dehaloer<R> {
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

        dehalo_alpha(&self.client, context, tables, input, output, &self.params);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Dehaloer<R> {}

impl<R: Runtime> Filter for Dehaloer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        self.planewise.run(&self.client, context, input, self)
    }
}
