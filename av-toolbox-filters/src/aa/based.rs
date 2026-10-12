#[cfg(feature = "test-support")]
use std::fmt::Display;

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
use av_toolbox_cubek::aa::{
    Eedi3Arguments,
    Eedi3DeviceClips,
    Eedi3Settings,
    binarize_prewitt,
    double_mclip,
    double_mclip_turned,
    prewitt_threshold,
    pscale_blend,
    transpose,
};
use av_toolbox_cubek::bilateral::{BilateralPlanes, BilateralSettings, bilateral};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::lineart::{MergePlanes, clamped_merge};
use av_toolbox_cubek::neighbourhood::{
    Extreme,
    NeighbourPlanes,
    Shape,
    Smooth,
    extreme,
    median_of_three,
    smooth,
};
use av_toolbox_cubek::resize::{Kernel, ResizePlanes, Weights, apply_float, centred_taps, upload};
use av_toolbox_cubek::{DevicePlane, assert_plane};
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
    FrameLayout,
    Mask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_models::Store;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer};

use crate::aa::direction::{Direction, run_direction};
use crate::aa::eedi3::{
    Eedi3FieldPass,
    checked_tuning,
    default_beta,
    default_mdis,
    default_nrad,
    default_vcheck,
    indexable,
    thresholds_for,
};
use crate::artcnn::{ModelName, Supersampler, locate};
use crate::bits::bits_of;
use crate::checks::{between_f32, between_u32};
use crate::cnn::{PrecisionName, checked_precision};
use crate::masked::{
    INPUT,
    MASK,
    MASK_PEAK,
    MASKED_INPUTS,
    accepts_masked_input,
    checked_input,
    checked_mask,
    mask_weights,
    missing_input,
};
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::sizing::Sizing;

const STAGE: &str = "aa/based";

const MAX_MASK_THR: u32 = 255;

/// The fewest samples the luma plane needs in either dimension.
const MIN_LUMA_SIZE: u32 = 2;

/// The postfilter's bilateral spatial sigma.
const SIGMA_S: f32 = 2.0;

/// The postfilter's bilateral range sigma, on the `0..1` scale.
const SIGMA_R: f32 = 1.0 / 255.0;

fn default_mask_thr() -> u32 {
    60
}

fn enabled() -> bool {
    true
}

/// Arguments for `aa/based`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct BasedArgs {
    /// The built mask's threshold, between 0 and 255, on the 8-bit scale.
    #[serde(default = "default_mask_thr")]
    pub(crate) mask_thr: u32,
    /// Whether to merge the anti-aliased luma into the input through a mask.
    #[serde(default = "enabled")]
    pub(crate) edge_mask: bool,
    /// How much of the supersample's own change to keep, between 0 and 1. At 1 the anti-aliased result
    /// is used as it is.
    #[serde(default)]
    pub(crate) pscale: f32,
    /// ArtCNN's network: `r8f64`, or the larger and slower `r16f96`.
    #[serde(default)]
    pub(crate) model: ModelName,
    /// ArtCNN's feature map precision: `f16`, or the slower and more exact `f32`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
    /// Whether to take the median of the result, the input and a bilateral of the result, which removes
    /// halos.
    #[serde(default = "enabled")]
    pub(crate) postfilter: bool,
    /// The axes EEDI3 anti-aliases along, both when not given.
    #[serde(default)]
    pub(crate) direction: Direction,
    /// EEDI3's settings.
    #[serde(default)]
    pub(crate) eedi3: Eedi3Tuning,
}

/// EEDI3's settings before they are checked, as `aa/based`'s `eedi3` object gives them.
#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct Eedi3Tuning {
    /// The weight of the neighbourhood cost, 0 or between 1e-30 and 1.
    pub(crate) alpha: f64,
    /// The weight of the vertical difference cost, 0 or between 1e-30 and 1, on the 8-bit scale.
    ///
    /// `alpha + beta` may not pass 1.
    pub(crate) beta: f64,
    /// The cost of changing direction, between 0 and 1e30, on the 8-bit scale.
    pub(crate) gamma: f64,
    /// The neighbourhood radius, between 0 and 3.
    pub(crate) nrad: u32,
    /// The furthest direction searched, between 1 and 40.
    pub(crate) mdis: u32,
    /// How strongly an unreliable direction falls back, between 0 and 3.
    pub(crate) vcheck: u32,
    /// The three thresholds of the reliability check, each at least 1e-30.
    ///
    /// The first two are on the 8-bit scale.
    #[serde(deserialize_with = "three_thresholds")]
    pub(crate) vthresh: [f64; 3],
}

impl Default for Eedi3Tuning {
    fn default() -> Self {
        Self {
            alpha: 0.125,
            beta: default_beta(),
            gamma: 40.0,
            nrad: default_nrad(),
            mdis: default_mdis(),
            vcheck: default_vcheck(),
            vthresh: [12.0, 24.0, 4.0],
        }
    }
}

fn three_thresholds<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[f64; 3], D::Error> {
    thresholds_for(STAGE, deserializer)
}

fn unusable_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} anti-aliases grey or yuv frames, not {layout} frames");

    StageError::msg(message)
}

fn unusable_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} anti-aliases U8 or U16 frames, not {sample} frames");

    StageError::msg(message)
}

fn mask_without_merge() -> StageError {
    let message = format!("{STAGE} needs edge_mask = true to merge through a bound mask, got false");

    StageError::msg(message)
}

fn too_small(width: u32, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs a luma plane of at least {MIN_LUMA_SIZE}x{MIN_LUMA_SIZE}, got {width}x{height}"
    );

    StageError::msg(message)
}

fn too_large(direction: Direction, mdis: u32, width: u32, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs a luma plane small enough for EEDI3 to index at twice its size with mdis {mdis} and \
         direction {direction}, got {width}x{height}"
    );

    StageError::msg(message)
}

#[cfg(feature = "test-support")]
fn unreadable_args(error: impl Display) -> StageError {
    let message = format!("{STAGE} could not read its arguments: {error}");

    StageError::msg(message)
}

#[cfg(feature = "test-support")]
fn edges_mismatch(edge_mask: bool) -> StageError {
    let message = format!("{STAGE} was handed mask planes that do not match edge_mask = {edge_mask}");

    StageError::msg(message)
}

fn missing_frame_mask() -> StageError {
    let message = format!("{STAGE} was handed a frame without its mask");

    StageError::msg(message)
}

/// The frame's mask, refusing a frame that lacks one when a mask is `bound`.
pub(crate) fn checked_frame_mask(bound: bool, mask: Option<Mask>) -> Result<Option<Mask>, StageError> {
    if bound && mask.is_none() {
        return Err(missing_frame_mask());
    }

    Ok(mask)
}

fn based_layout(layout: FrameLayout) -> bool {
    matches!(layout, FrameLayout::Grey | FrameLayout::Yuv { .. })
}

/// Refuses a layout other than grey or YUV, and samples other than U8 or U16.
fn checked_format(format: &FrameFormat) -> Result<(), StageError> {
    if !based_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !matches!(format.sample, SampleFormat::U8 | SampleFormat::U16 { .. }) {
        return Err(unusable_sample(format.sample));
    }

    Ok(())
}

/// The arguments EEDI3 takes, once every argument of `args` is checked.
fn checked_settings(args: &BasedArgs) -> Result<Eedi3Arguments, StageError> {
    between_u32(STAGE, "mask_thr", args.mask_thr, 0, MAX_MASK_THR)?;
    between_f32(STAGE, "pscale", args.pscale, 0.0, 1.0)?;

    checked_tuning(STAGE, &args.eedi3)
}

/// Refuses a `width` by `height` luma plane that EEDI3 cannot run on once doubled.
///
/// A horizontal pass runs on the transposed doubled plane.
fn checked_luma(width: u32, height: u32, direction: Direction, mdis: u32) -> Result<(), StageError> {
    if width < MIN_LUMA_SIZE || height < MIN_LUMA_SIZE {
        return Err(too_small(width, height));
    }

    let doubled_width = 2 * u64::from(width);
    let doubled_height = 2 * u64::from(height);
    let fits = doubled_width <= u64::from(u32::MAX) && doubled_height <= u64::from(u32::MAX);
    if !fits {
        return Err(too_large(direction, mdis, width, height));
    }

    let doubled_width = doubled_width as u32;
    let doubled_height = doubled_height as u32;
    if direction.runs_vertically() && !indexable(doubled_width, doubled_height, mdis) {
        return Err(too_large(direction, mdis, width, height));
    }

    if direction.runs_horizontally() && !indexable(doubled_height, doubled_width, mdis) {
        return Err(too_large(direction, mdis, width, height));
    }

    Ok(())
}

/// The input format and EEDI3's checked arguments, once `args` and the bound inputs are checked.
#[derive(Debug)]
pub(crate) struct Plan {
    pub(crate) input: FrameFormat,
    pub(crate) arguments: Eedi3Arguments,
    /// Whether a mask is bound.
    pub(crate) masked: bool,
}

/// The stage's plan, once `args` and the bound inputs are checked and the model is found in `store`.
pub(crate) fn planned(
    capabilities: &Capabilities,
    args: &BasedArgs,
    chosen: &[Option<StreamFormat>],
    store: &Store,
) -> Result<Plan, StageError> {
    let input = checked_input(STAGE, chosen)?;
    checked_format(&input)?;

    let mask = chosen.get(MASK).copied().flatten();
    checked_mask(STAGE, &input, mask)?;
    if mask.is_some() && !args.edge_mask {
        return Err(mask_without_merge());
    }

    let arguments = checked_settings(args)?;
    checked_luma(input.width, input.height, args.direction, arguments.mdis)?;
    checked_precision(STAGE, capabilities, args.precision)?;
    locate(STAGE, store, args.model)?;

    Ok(Plan {
        input,
        arguments,
        masked: mask.is_some(),
    })
}

/// Anti-aliases luma the way vs-jetpack's `based_aa` does, running EEDI3 on an ArtCNN supersample and
/// merging the result back through an edge mask.
///
/// It runs as a single instance, since it holds ArtCNN's tiles and several doubled planes.
pub(crate) struct Based;

impl StageDefinition for Based {
    type Args = BasedArgs;

    fn package() -> &'static str {
        "aa"
    }

    fn identifier() -> &'static str {
        "based"
    }

    fn description() -> &'static str {
        "Anti-aliases luma by doubling it with ArtCNN, running EEDI3, then halving it.\n\n\
         Other planes pass through unchanged. It is slower than `aa/eedi3` alone."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Stream
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Based {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let frames = offered.as_frames();
        if let (INPUT, Some(format)) = (index, frames)
            && !based_layout(format.layout)
        {
            return Err(unusable_layout(format.layout));
        }

        accepts_masked_input(STAGE, index, offered)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let store = Store::from_environment();
        let plan = planned(capabilities, args, chosen, &store)?;
        let input = plan.input;

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
        let capabilities = context.capabilities();
        let store = Store::from_environment();
        let plan = planned(capabilities, &args, inputs, &store)?;
        let input = plan.input;

        let bits = bits_of(input.sample);
        let chain = BasedChain::new(context, &args, &plan.arguments, bits, input.width, input.height)?;

        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let channels = input.channels();
        let selected = channels.iter().map(|channel| *channel == Channel::Y).collect();
        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak: chain.peak,
        };
        let antialiaser = Antialiaser {
            chain,
            planewise,
            edge_mask: args.edge_mask,
            masked: plan.masked,
        };

        Ok(Box::new(antialiaser))
    }
}

/// The f32 planes one run of `aa/based` reads and writes, in code values.
///
/// `luma` is read and the rest are written. `luma` and `merged` are the luma's size, and
/// `supersampled` and `antialiased` are twice it. `merged` holds the result before its one rounding.
#[derive(Debug, Clone, Copy)]
pub struct BasedPlanes<'a> {
    pub luma: DevicePlane<'a>,
    pub supersampled: DevicePlane<'a>,
    pub antialiased: DevicePlane<'a>,
    pub merged: DevicePlane<'a>,
    /// The mask planes, given exactly when `edge_mask` is set.
    pub edges: Option<EdgePlanes<'a>>,
}

/// The mask a run merges through, and the clip EEDI3 searches within.
///
/// A `bound` mask holds 8-bit values in `mask` before the run. Otherwise the run builds the mask there
/// from luma. `mask` is the luma's size and `mclip` twice it.
#[derive(Debug, Clone, Copy)]
pub struct EdgePlanes<'a> {
    pub mask: DevicePlane<'a>,
    pub mclip: DevicePlane<'a>,
    pub bound: bool,
}

/// The box tables that halve a doubled plane on each axis.
struct Halving {
    across: Weights,
    down: Weights,
}

impl Halving {
    fn new<R: Runtime>(client: &ComputeClient<R>, width: u32, height: u32) -> Self {
        let across_taps = centred_taps(2 * width, width, &Kernel::Box);
        let down_taps = centred_taps(2 * height, height, &Kernel::Box);
        let across = upload(client, &across_taps);
        let down = upload(client, &down_taps);

        Self { across, down }
    }
}

/// The transposed supersample and mask clip the horizontal EEDI3 pass reads.
struct TurnedClips {
    supersampled: Scratch,
    mclip: Option<Scratch>,
}

impl TurnedClips {
    /// Transposes `supersampled`, keeping `mclip`, the clip already transposed.
    fn new<R: Runtime>(
        client: &ComputeClient<R>,
        context: &FrameContext,
        supersampled: DevicePlane<'_>,
        mclip: Option<Scratch>,
    ) -> Self {
        let sizing = Sizing(context);
        let width = supersampled.width();
        let height = supersampled.height();

        let turned_supersampled = Scratch::new::<f32>(context, height, width);
        let turned_supersampled_plane = turned_supersampled.plane();
        transpose(client, &sizing, supersampled, turned_supersampled_plane);

        Self {
            supersampled: turned_supersampled,
            mclip,
        }
    }

    fn clips(&self) -> Eedi3DeviceClips<'_> {
        let sclip = self.supersampled.plane();
        let mclip = self.mclip.as_ref().map(Scratch::plane);

        Eedi3DeviceClips {
            sclip: Some(sclip),
            mclip,
        }
    }
}

/// Everything `aa/based` settles at build, which runs its chain on a luma plane of one size.
pub(crate) struct BasedChain<R: Runtime> {
    client: ComputeClient<R>,
    supersampler: Supersampler,
    settings: Eedi3Settings,
    direction: Direction,
    /// `(mask_thr * 2^(bits - 8))^2`, the squared gradient a built mask sample is set at.
    threshold: u32,
    /// `1 - pscale`, or `None` when `pscale` is 1 and the blend is skipped.
    keep: Option<f32>,
    postfilter: bool,
    halving: Halving,
    peak: f32,
}

impl<R: Runtime> BasedChain<R> {
    /// Loads ArtCNN and settles every setting for a `width` by `height` luma plane of `bits` per sample.
    fn new(
        context: &BuildContext<R>,
        args: &BasedArgs,
        arguments: &Eedi3Arguments,
        bits: u32,
        width: u32,
        height: u32,
    ) -> Result<Self, StageError> {
        let supersampler = Supersampler::load(STAGE, context, args.model, args.precision, width, height)?;

        let client = context.client().clone();
        let settings = Eedi3Settings::new(arguments, bits);
        let threshold = prewitt_threshold(args.mask_thr, bits);
        let keep = (args.pscale != 1.0).then_some(1.0 - args.pscale);
        let halving = Halving::new(&client, width, height);
        let peak = ((1u32 << bits) - 1) as f32;

        Ok(Self {
            client,
            supersampler,
            settings,
            direction: args.direction,
            threshold,
            keep,
            postfilter: args.postfilter,
            halving,
            peak,
        })
    }

    /// Runs every step of `aa/based` on `planes.luma`, writing each step's result into `planes`.
    pub(crate) fn run(&self, context: &FrameContext, planes: BasedPlanes<'_>) -> Result<(), StageError> {
        let width = planes.luma.width();
        let height = planes.luma.height();
        assert_plane(&planes.luma, width, height, "luma");
        assert_plane(&planes.supersampled, 2 * width, 2 * height, "supersampled");
        assert_plane(&planes.antialiased, 2 * width, 2 * height, "antialiased");
        assert_plane(&planes.merged, width, height, "merged");
        if let Some(edges) = planes.edges {
            assert_plane(&edges.mask, width, height, "mask");
            assert_plane(&edges.mclip, 2 * width, 2 * height, "mclip");
        }

        // The horizontal pass reads the clip transposed, written beside the clip itself.
        let reads_turned_mclip = self.direction.runs_horizontally() && planes.edges.is_some();
        let turned_mclip = reads_turned_mclip.then(|| Scratch::new::<f32>(context, 2 * height, 2 * width));
        let turned_mclip_plane = turned_mclip.as_ref().map(Scratch::plane);

        let mask_peak = match planes.edges {
            Some(edges) => self.prepare_mask(context, planes.luma, edges, turned_mclip_plane),
            None => self.peak,
        };

        self.supersampler
            .run(&self.client, context, planes.luma, self.peak, planes.supersampled)?;

        let mclip = planes.edges.map(|edges| edges.mclip);
        self.antialias(
            context,
            planes.supersampled,
            mclip,
            turned_mclip,
            planes.antialiased,
        );

        if self.postfilter {
            let blended = Scratch::new::<f32>(context, width, height);
            let blended_plane = blended.plane();

            self.downscale(context, &planes, blended_plane);
            self.postfilter(context, planes.luma, blended_plane, planes.merged);
        } else {
            self.downscale(context, &planes, planes.merged);
        }

        if let Some(edges) = planes.edges {
            let sizing = Sizing(context);
            let merge = MergePlanes {
                input: planes.luma,
                filtered: planes.merged,
                weight: edges.mask,
            };
            clamped_merge(&self.client, &sizing, merge, mask_peak, self.peak);
        }

        Ok(())
    }

    /// Builds the mask when it is not bound, then doubles it into the clip, giving the mask's peak.
    ///
    /// When `turned` is given, the clip is also written into it transposed.
    fn prepare_mask(
        &self,
        context: &FrameContext,
        luma: DevicePlane<'_>,
        edges: EdgePlanes<'_>,
        turned: Option<DevicePlane<'_>>,
    ) -> f32 {
        let sizing = Sizing(context);

        let mask_peak = if edges.bound {
            MASK_PEAK
        } else {
            self.build_mask(context, luma, edges.mask);
            self.peak
        };

        match turned {
            Some(turned) => {
                double_mclip_turned(&self.client, &sizing, edges.mask, edges.mclip, turned, mask_peak);
            },
            None => double_mclip(&self.client, &sizing, edges.mask, edges.mclip, mask_peak),
        }

        mask_peak
    }

    /// Writes the Prewitt binarize of `luma`, widened by a 3x3 maximum and softened by a 3x3 box, into
    /// `mask`.
    fn build_mask(&self, context: &FrameContext, luma: DevicePlane<'_>, mask: DevicePlane<'_>) {
        let sizing = Sizing(context);
        let width = luma.width();
        let height = luma.height();
        let binarized = Scratch::new::<f32>(context, width, height);
        let widened = Scratch::new::<f32>(context, width, height);
        let binarized_plane = binarized.plane();
        let widened_plane = widened.plane();

        binarize_prewitt(
            &self.client,
            &sizing,
            luma,
            binarized_plane,
            self.threshold,
            self.peak,
        );

        let maximum = NeighbourPlanes {
            input: binarized_plane,
            output: widened_plane,
        };
        extreme(&self.client, &sizing, maximum, Extreme::Maximum, Shape::Square);

        let softened = NeighbourPlanes {
            input: widened_plane,
            output: mask,
        };
        smooth(&self.client, &sizing, softened, Smooth::Box);
    }

    /// Runs EEDI3 on `supersampled` into `output`, with `supersampled` as its `sclip`.
    ///
    /// The horizontal pass reads the transposed supersample, made once before either pass, and
    /// `turned_mclip`, the clip already transposed.
    fn antialias(
        &self,
        context: &FrameContext,
        supersampled: DevicePlane<'_>,
        mclip: Option<DevicePlane<'_>>,
        turned_mclip: Option<Scratch>,
        output: DevicePlane<'_>,
    ) {
        let vertical = Eedi3FieldPass {
            client: &self.client,
            settings: self.settings,
            clips: Eedi3DeviceClips {
                sclip: Some(supersampled),
                mclip,
            },
        };

        let turned = self
            .direction
            .runs_horizontally()
            .then(|| TurnedClips::new(&self.client, context, supersampled, turned_mclip));
        let horizontal_clips = match &turned {
            Some(turned) => turned.clips(),
            None => Eedi3DeviceClips::default(),
        };
        let horizontal = Eedi3FieldPass {
            client: &self.client,
            settings: self.settings,
            clips: horizontal_clips,
        };

        run_direction(
            &self.client,
            context,
            &vertical,
            &horizontal,
            supersampled,
            output,
            self.direction,
        );
    }

    /// Halves the anti-aliased plane into `output`, putting back `1 - pscale` of what supersampling
    /// changed.
    fn downscale(&self, context: &FrameContext, planes: &BasedPlanes<'_>, output: DevicePlane<'_>) {
        let Some(keep) = self.keep else {
            self.halve(context, planes.antialiased, output);
            return;
        };

        let sizing = Sizing(context);
        let width = output.width();
        let height = output.height();
        let halved = Scratch::new::<f32>(context, width, height);
        let down = Scratch::new::<f32>(context, width, height);
        let halved_plane = halved.plane();
        let down_plane = down.plane();

        self.halve(context, planes.antialiased, halved_plane);
        self.halve(context, planes.supersampled, down_plane);
        pscale_blend(
            &self.client,
            &sizing,
            halved_plane,
            planes.luma,
            down_plane,
            output,
            keep,
        );
    }

    /// Averages each 2x2 block of `input` into `output`.
    fn halve(&self, context: &FrameContext, input: DevicePlane<'_>, output: DevicePlane<'_>) {
        let sizing = Sizing(context);
        let rows = Scratch::new::<f32>(context, output.width(), input.height());
        let rows_plane = rows.plane();
        let planes = ResizePlanes {
            input,
            scratch: rows_plane,
            output,
        };

        apply_float(
            &self.client,
            &sizing,
            planes,
            &self.halving.across,
            &self.halving.down,
        );
    }

    /// Writes the median of `blended`, `luma` and the bilateral of `blended` against `luma` into
    /// `output`.
    fn postfilter(
        &self,
        context: &FrameContext,
        luma: DevicePlane<'_>,
        blended: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) {
        let sizing = Sizing(context);
        let smoothed = Scratch::new::<f32>(context, luma.width(), luma.height());
        let smoothed_plane = smoothed.plane();

        let planes = BilateralPlanes {
            input: blended,
            reference: luma,
            output: smoothed_plane,
        };
        let settings = BilateralSettings {
            sigma_s: SIGMA_S,
            sigma_r: SIGMA_R,
            peak: self.peak,
        };
        bilateral(&self.client, &sizing, planes, settings);

        median_of_three(&self.client, &sizing, blended, luma, smoothed_plane, output);
    }
}

/// The doubled planes and mask planes one frame's run works in.
struct FramePlanes {
    supersampled: Scratch,
    antialiased: Scratch,
    mask: Option<Scratch>,
    mclip: Option<Scratch>,
}

/// Runs the chain on the luma plane handed to it, merging through the frame's mask when one is bound.
struct BasedFilter<'a, R: Runtime> {
    chain: &'a BasedChain<R>,
    format: &'a FrameFormat,
    mask: Option<&'a Mask>,
    edge_mask: bool,
}

impl<R: Runtime> BasedFilter<'_, R> {
    /// Takes the planes the chain writes for a `width` by `height` luma plane, widening a bound mask.
    fn frame_planes(
        &self,
        context: &FrameContext,
        width: u32,
        height: u32,
    ) -> Result<FramePlanes, StageError> {
        let supersampled = Scratch::new::<f32>(context, 2 * width, 2 * height);
        let antialiased = Scratch::new::<f32>(context, 2 * width, 2 * height);
        if !self.edge_mask {
            return Ok(FramePlanes {
                supersampled,
                antialiased,
                mask: None,
                mclip: None,
            });
        }

        let mask = match self.mask {
            Some(bound) => mask_weights(
                &self.chain.client,
                context,
                STAGE,
                self.format,
                bound,
                Channel::Y,
                (width, height),
            )?,
            None => Scratch::new::<f32>(context, width, height),
        };
        let mclip = Scratch::new::<f32>(context, 2 * width, 2 * height);

        Ok(FramePlanes {
            supersampled,
            antialiased,
            mask: Some(mask),
            mclip: Some(mclip),
        })
    }
}

impl<R: Runtime> PlaneFilter for BasedFilter<'_, R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let frame_planes = self.frame_planes(context, input.width(), input.height())?;
        let edges = match (&frame_planes.mask, &frame_planes.mclip) {
            (Some(mask), Some(mclip)) => Some(EdgePlanes {
                mask: mask.plane(),
                mclip: mclip.plane(),
                bound: self.mask.is_some(),
            }),
            _ => None,
        };

        let planes = BasedPlanes {
            luma: input,
            supersampled: frame_planes.supersampled.plane(),
            antialiased: frame_planes.antialiased.plane(),
            merged: output,
            edges,
        };

        self.chain.run(context, planes)
    }
}

/// Runs `aa/based` on every frame's luma, handing the other planes through.
struct Antialiaser<R: Runtime> {
    chain: BasedChain<R>,
    planewise: Planewise,
    edge_mask: bool,
    /// Whether a mask is bound, so every frame must arrive with one.
    masked: bool,
}

impl<R: Runtime> StageInstance for Antialiaser<R> {}

impl<R: Runtime> Filter for Antialiaser<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let taken = inputs.take_mask(MASK);
        let mask = checked_frame_mask(self.masked, taken)?;
        let filter = BasedFilter {
            chain: &self.chain,
            format: &self.planewise.format,
            mask: mask.as_ref(),
            edge_mask: self.edge_mask,
        };

        let output = self.planewise.run(&self.chain.client, context, input, &filter)?;

        Ok(vec![output])
    }
}

/// Runs `aa/based` with `args` on one luma plane of `bits` per sample, writing every step's result into
/// `planes`.
///
/// `planes.edges` must be given exactly when `edge_mask` is set. ArtCNN is loaded and warmed up before
/// the run.
#[cfg(feature = "test-support")]
pub fn based_stages<'de, R: Runtime, D: Deserializer<'de>>(
    context: &BuildContext<R>,
    args: D,
    bits: u32,
    planes: BasedPlanes<'_>,
) -> Result<(), StageError> {
    let parsed = BasedArgs::deserialize(args);
    let args = match parsed {
        Ok(args) => args,
        Err(error) => return Err(unreadable_args(error)),
    };

    between_u32(STAGE, "bits", bits, 8, 16)?;

    let arguments = checked_settings(&args)?;
    if args.edge_mask != planes.edges.is_some() {
        return Err(edges_mismatch(args.edge_mask));
    }

    let width = planes.luma.width();
    let height = planes.luma.height();
    checked_luma(width, height, args.direction, arguments.mdis)?;

    let chain = BasedChain::new(context, &args, &arguments, bits, width, height)?;
    let frame_context = context.frame_context();

    chain.run(&frame_context, planes)
}
