use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::aa::{
    Eedi3Arguments,
    Eedi3Check,
    Eedi3DeviceClips,
    Eedi3Scratch,
    Eedi3Settings,
    eedi3_field,
    eedi3_fields,
    eedi3_planned_check,
    eedi3_scratch,
};
use av_toolbox_cubek::depth::Element;
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
    PlaneBuffer,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

use crate::aa::based::Eedi3Tuning;
use crate::aa::direction::{Direction, FieldPass, checked_dimensions, missing_plane, run_direction};
use crate::bits::bits_of;
use crate::checks::between_u32;
use crate::masked::{
    INPUT,
    MASK,
    MASKED_INPUTS,
    MaskedFilter,
    accepts_masked_input,
    checked_input,
    checked_mask,
    missing_input,
};
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "aa/eedi3";

const MAX_NRAD: u32 = 3;

const MIN_MDIS: u32 = 1;

const MAX_MDIS: u32 = 40;

const MAX_VCHECK: u32 = 3;

/// The smallest non-zero `alpha` or `beta`, and the smallest `vthresh`, which keep every weight a normal
/// f32.
const MIN_POSITIVE: f32 = 1e-30;

/// The largest `gamma`, which keeps every path cost finite.
const MAX_GAMMA: f32 = 1e30;

/// The fewest samples a plane needs in either dimension.
const MIN_PLANE_SIZE: u32 = 4;

fn default_alpha() -> f64 {
    0.2
}

pub(crate) fn default_beta() -> f64 {
    0.25
}

fn default_gamma() -> f64 {
    20.0
}

pub(crate) fn default_nrad() -> u32 {
    2
}

pub(crate) fn default_mdis() -> u32 {
    20
}

pub(crate) fn default_vcheck() -> u32 {
    2
}

fn default_vthresh() -> [f64; 3] {
    [32.0, 64.0, 4.0]
}

/// Arguments for `aa/eedi3`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Eedi3Args {
    /// The weight of the neighbourhood cost, 0 or between 1e-30 and 1.
    #[serde(default = "default_alpha")]
    pub(crate) alpha: f64,
    /// The weight of the vertical difference cost, 0 or between 1e-30 and 1, on the 8-bit scale.
    ///
    /// `alpha + beta` may not pass 1.
    #[serde(default = "default_beta")]
    pub(crate) beta: f64,
    /// The cost of changing direction, between 0 and 1e30, on the 8-bit scale.
    #[serde(default = "default_gamma")]
    pub(crate) gamma: f64,
    /// The neighbourhood radius, between 0 and 3.
    #[serde(default = "default_nrad")]
    pub(crate) nrad: u32,
    /// The furthest direction searched, between 1 and 40.
    #[serde(default = "default_mdis")]
    pub(crate) mdis: u32,
    /// How strongly an unreliable direction falls back, between 0 and 3.
    #[serde(default = "default_vcheck")]
    pub(crate) vcheck: u32,
    /// The three thresholds of the reliability check, each at least 1e-30.
    ///
    /// The first two are on the 8-bit scale.
    #[serde(default = "default_vthresh", deserialize_with = "three_thresholds")]
    pub(crate) vthresh: [f64; 3],
    /// The planes to anti-alias, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// The axes to anti-alias along, both when not given.
    #[serde(default)]
    pub(crate) direction: Direction,
}

fn wrong_threshold_count(stage: &str, count: usize) -> String {
    format!("{stage} needs vthresh of 3 numbers, got {count}")
}

/// Reads `vthresh`, refusing a list of any length but 3 by naming its length.
fn three_thresholds<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f64; 3], D::Error> {
    thresholds_for(STAGE, deserializer)
}

/// Reads `vthresh` for `stage`, refusing a list of any length but 3 by naming its length.
pub(crate) fn thresholds_for<'de, D: Deserializer<'de>>(
    stage: &str,
    deserializer: D,
) -> Result<[f64; 3], D::Error> {
    let values = Vec::<f64>::deserialize(deserializer)?;
    let count = values.len();

    match <[f64; 3]>::try_from(values) {
        Ok(thresholds) => Ok(thresholds),
        Err(_) => {
            let message = wrong_threshold_count(stage, count);
            Err(serde::de::Error::custom(message))
        },
    }
}

fn not_a_weight(stage: &str, name: &str, value: f32) -> StageError {
    let message = format!("{stage} needs {name} of 0 or between 1e-30 and 1, got {value:?}");

    StageError::msg(message)
}

fn weights_past_one(stage: &str, sum: f32) -> StageError {
    let message = format!("{stage} needs alpha and beta to add up to at most 1, got {sum:?}");

    StageError::msg(message)
}

fn gamma_out_of_range(stage: &str, value: f32) -> StageError {
    let message = format!("{stage} needs gamma between 0 and 1e30, got {value:?}");

    StageError::msg(message)
}

fn threshold_too_small(stage: &str, index: usize, value: f32) -> StageError {
    let message = format!("{stage} needs a finite vthresh[{index}] of at least 1e-30, got {value:?}");

    StageError::msg(message)
}

fn too_small(channel: Channel, width: u32, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs plane {channel} to be at least {MIN_PLANE_SIZE}x{MIN_PLANE_SIZE}, got {width}x{height}"
    );

    StageError::msg(message)
}

fn too_large(channel: Channel, direction: Direction, mdis: u32, width: u32, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs plane {channel} small enough to index with mdis {mdis} and direction {direction}, \
         got {width}x{height}"
    );

    StageError::msg(message)
}

/// Rounds the weight `name` to f32, refusing it unless it is 0 or between 1e-30 and 1.
fn checked_weight(stage: &str, name: &str, value: f64) -> Result<f32, StageError> {
    let rounded = value as f32;
    if rounded == 0.0 || (MIN_POSITIVE..=1.0).contains(&rounded) {
        return Ok(rounded);
    }

    Err(not_a_weight(stage, name, rounded))
}

/// Rounds the threshold `vthresh[index]` to f32, refusing it unless it is finite and at least 1e-30.
fn checked_threshold(stage: &str, index: usize, value: f64) -> Result<f32, StageError> {
    let rounded = value as f32;
    if rounded.is_finite() && rounded >= MIN_POSITIVE {
        return Ok(rounded);
    }

    Err(threshold_too_small(stage, index, rounded))
}

/// The arguments the device pass takes, once `args` are rounded to f32 and checked.
pub(crate) fn checked_arguments(args: &Eedi3Args) -> Result<Eedi3Arguments, StageError> {
    let tuning = Eedi3Tuning {
        alpha: args.alpha,
        beta: args.beta,
        gamma: args.gamma,
        nrad: args.nrad,
        mdis: args.mdis,
        vcheck: args.vcheck,
        vthresh: args.vthresh,
    };

    checked_tuning(STAGE, &tuning)
}

/// The arguments the device pass takes, once `tuning` is rounded to f32 and checked for `stage`.
///
/// `alpha + beta` is checked in f32.
pub(crate) fn checked_tuning(stage: &str, tuning: &Eedi3Tuning) -> Result<Eedi3Arguments, StageError> {
    let alpha = checked_weight(stage, "alpha", tuning.alpha)?;
    let beta = checked_weight(stage, "beta", tuning.beta)?;
    let sum = alpha + beta;
    if sum > 1.0 {
        return Err(weights_past_one(stage, sum));
    }

    let gamma = tuning.gamma as f32;
    if !(0.0..=MAX_GAMMA).contains(&gamma) {
        return Err(gamma_out_of_range(stage, gamma));
    }

    between_u32(stage, "nrad", tuning.nrad, 0, MAX_NRAD)?;
    between_u32(stage, "mdis", tuning.mdis, MIN_MDIS, MAX_MDIS)?;
    between_u32(stage, "vcheck", tuning.vcheck, 0, MAX_VCHECK)?;

    let mut vthresh = [0.0; 3];
    for (index, value) in tuning.vthresh.iter().enumerate() {
        vthresh[index] = checked_threshold(stage, index, *value)?;
    }

    Ok(Eedi3Arguments {
        alpha,
        beta,
        gamma,
        nrad: tuning.nrad,
        mdis: tuning.mdis,
        vcheck: tuning.vcheck,
        vthresh,
    })
}

/// Whether a field pass over a `width` by `height` plane indexes every sample and backpointer word in a
/// `u32`.
pub(crate) fn indexable(width: u32, height: u32, mdis: u32) -> bool {
    let samples = width as usize * height as usize;
    let sizes = eedi3_scratch(width, height, mdis);
    let words = sizes.backpointers / size_of::<u32>();

    samples <= u32::MAX as usize && words <= u32::MAX as usize
}

/// Refuses a `width` by `height` plane `channel` that the field passes along `direction` cannot run on.
///
/// A plane needs at least 4 samples in each dimension. A horizontal pass runs on the transposed plane.
fn checked_size(
    channel: Channel,
    direction: Direction,
    mdis: u32,
    width: u32,
    height: u32,
) -> Result<(), StageError> {
    if width < MIN_PLANE_SIZE || height < MIN_PLANE_SIZE {
        return Err(too_small(channel, width, height));
    }

    let runs_vertically = matches!(direction, Direction::Vertical | Direction::Both);
    if runs_vertically && !indexable(width, height, mdis) {
        return Err(too_large(channel, direction, mdis, width, height));
    }

    let runs_horizontally = matches!(direction, Direction::Horizontal | Direction::Both);
    if runs_horizontally && !indexable(height, width, mdis) {
        return Err(too_large(channel, direction, mdis, width, height));
    }

    Ok(())
}

/// Whether each plane of `input` is anti-aliased, once `args` are checked against it.
fn checked_planes(args: &Eedi3Args, input: &FrameFormat) -> Result<Vec<bool>, StageError> {
    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::First)?;
    checked_dimensions(STAGE, input, &selected, args.direction)?;

    let channels = input.channels();
    let chosen = channels
        .iter()
        .zip(&selected)
        .filter(|(_, is_selected)| **is_selected);
    for (channel, _) in chosen {
        let Some((width, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(STAGE, *channel));
        };

        checked_size(*channel, args.direction, args.mdis, width, height)?;
    }

    Ok(selected)
}

/// Anti-aliases by re-interpolating each field along the cheapest path of edge directions across each row,
/// the way EEDI3 does.
pub(crate) struct Eedi3;

impl StageDefinition for Eedi3 {
    type Args = Eedi3Args;

    fn package() -> &'static str {
        "aa"
    }

    fn identifier() -> &'static str {
        "eedi3"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Eedi3 {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_masked_input(STAGE, index, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        let mask = chosen.get(MASK).copied().flatten();
        checked_arguments(args)?;
        checked_planes(args, &input)?;
        checked_mask(STAGE, &input, mask)?;

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
        let input = checked_input(STAGE, inputs)?;
        let mask = inputs.get(MASK).copied().flatten();
        let arguments = checked_arguments(&args)?;
        let selected = checked_planes(&args, &input)?;
        checked_mask(STAGE, &input, mask)?;

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let settings = Eedi3Settings::new(&arguments, bits);

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let antialiaser = Antialiaser {
            client: context.client().clone(),
            planewise,
            direction: args.direction,
            settings,
        };

        Ok(Box::new(antialiaser))
    }
}

/// One EEDI3 field pass, reading `clips` beside its input.
pub(crate) struct Eedi3FieldPass<'a, R: Runtime> {
    pub(crate) client: &'a ComputeClient<R>,
    pub(crate) settings: Eedi3Settings,
    pub(crate) clips: Eedi3DeviceClips<'a>,
}

impl<R: Runtime> FieldPass for Eedi3FieldPass<'_, R> {
    fn run(&self, context: &FrameContext, input: DevicePlane<'_>, output: DevicePlane<'_>, offset: u32) {
        let mdis = self.settings.mdis();
        let buffers = FieldScratch::new(context, input.width(), input.height(), mdis);
        let scratch = buffers.scratch();

        let sizing = Sizing(context);
        eedi3_field(
            self.client,
            &sizing,
            input,
            output,
            scratch,
            offset,
            &self.settings,
            self.clips,
        );
    }

    fn run_both(
        &self,
        context: &FrameContext,
        input: DevicePlane<'_>,
        even_output: DevicePlane<'_>,
        odd_output: DevicePlane<'_>,
    ) {
        // Only the persistent check gains from both passes at once, so the others take one scratch in turn.
        let planned = eedi3_planned_check(self.client, &self.settings, input.width());
        if planned != Eedi3Check::Persistent {
            self.run(context, input, even_output, 0);
            self.run(context, input, odd_output, 1);
            return;
        }

        let mdis = self.settings.mdis();
        let even_buffers = FieldScratch::new(context, input.width(), input.height(), mdis);
        let odd_buffers = FieldScratch::new(context, input.width(), input.height(), mdis);
        let outputs = [even_output, odd_output];
        let scratches = [even_buffers.scratch(), odd_buffers.scratch()];

        let sizing = Sizing(context);
        eedi3_fields(
            self.client,
            &sizing,
            input,
            outputs,
            scratches,
            &self.settings,
            self.clips,
        );
    }
}

/// The scratch buffers of one EEDI3 field pass, returned to the pool when they drop.
struct FieldScratch {
    rebuilt: PlaneBuffer,
    directions: PlaneBuffer,
    backpointers: PlaneBuffer,
    lenders: PlaneBuffer,
}

impl FieldScratch {
    fn new(context: &FrameContext, width: u32, height: u32, mdis: u32) -> Self {
        let sizes = eedi3_scratch(width, height, mdis);

        Self {
            rebuilt: context.scratch(sizes.rebuilt),
            directions: context.scratch(sizes.directions),
            backpointers: context.scratch(sizes.backpointers),
            lenders: context.scratch(sizes.lenders),
        }
    }

    fn scratch(&self) -> Eedi3Scratch<'_> {
        Eedi3Scratch {
            rebuilt: self.rebuilt.handle(),
            directions: self.directions.handle(),
            backpointers: self.backpointers.handle(),
            lenders: self.lenders.handle(),
        }
    }
}

/// Runs `aa/eedi3` on every selected plane of a frame.
struct Antialiaser<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    direction: Direction,
    settings: Eedi3Settings,
}

impl<R: Runtime> PlaneFilter for Antialiaser<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let pass = Eedi3FieldPass {
            client: &self.client,
            settings: self.settings,
            clips: Eedi3DeviceClips::default(),
        };

        run_direction(&self.client, context, &pass, &pass, input, output, self.direction);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Antialiaser<R> {}

impl<R: Runtime> Filter for Antialiaser<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let mask = inputs.take_mask(MASK);
        let masked = MaskedFilter {
            client: &self.client,
            stage: STAGE,
            format: &self.planewise.format,
            mask: mask.as_ref(),
            peak: self.planewise.peak,
            inner: &*self,
        };

        self.planewise.run(&self.client, context, input, &masked)
    }
}
