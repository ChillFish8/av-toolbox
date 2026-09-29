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
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::edges::{EdgeLimits, Operator, binarize, edges};
use av_toolbox_cubek::lineart::{MergePlanes, ThinSettings, merge};
use av_toolbox_cubek::neighbourhood::{ChainPlanes, Extreme, NeighbourPlanes, Shape, Spread, Step, chain};
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
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

use super::mask::{
    INPUT,
    LINEART_INPUTS,
    MASK,
    MaskedFilter,
    accepts_lineart_input,
    checked_input,
    checked_mask,
    missing_input,
};
use crate::bits::bits_of;
use crate::checks::{between_f32, between_i32, between_u32};
use crate::padded_warp::padded_warp;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "lineart/thin";

const MAX_BLUR: u32 = 100;

const MAX_THRESH: u32 = 255;

const MAX_RADIUS: u32 = 4;

const MIN_DEPTH: i32 = -128;

const MAX_DEPTH: i32 = 127;

fn default_depth() -> i32 {
    16
}

fn default_blur() -> u32 {
    2
}

fn default_thresh() -> u32 {
    128
}

fn default_mthr() -> f32 {
    0.08
}

fn default_mrad() -> u32 {
    1
}

/// Arguments for `lineart/thin`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ThinArgs {
    /// How far the warp moves samples toward lines, between -128 and 127.
    #[serde(default = "default_depth")]
    pub(crate) depth: i32,
    /// How many times the warp's edge mask is blurred, between 0 and 100.
    #[serde(default = "default_blur")]
    pub(crate) blur: u32,
    /// The most an edge can weigh in the warp's edge mask, between 0 and 255.
    #[serde(default = "default_thresh")]
    pub(crate) thresh: u32,
    /// The share of the sample range a Prewitt edge must reach to count as a line, between 0 and 1.
    #[serde(default = "default_mthr")]
    pub(crate) mthr: f32,
    /// How many samples the line mask is grown by, between 0 and 4.
    #[serde(default = "default_mrad")]
    pub(crate) mrad: u32,
    /// The planes to thin, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// The passes that grow and soften a binarized line mask, `radius` square maxima then an inflate.
pub(crate) fn mask_steps(radius: u32) -> Vec<Step> {
    let maximum = Step::Extreme(Extreme::Maximum, Shape::Square);
    let mut steps = vec![maximum; radius as usize];
    let inflate = Step::Spread(Spread::Inflate);
    steps.push(inflate);

    steps
}

/// The thin settings for `sample`, once the arguments are checked.
pub(crate) fn thin_settings(args: &ThinArgs, sample: SampleFormat) -> Result<ThinSettings, StageError> {
    between_u32(STAGE, "blur", args.blur, 0, MAX_BLUR)?;
    between_u32(STAGE, "thresh", args.thresh, 0, MAX_THRESH)?;
    between_f32(STAGE, "mthr", args.mthr, 0.0, 1.0)?;
    between_u32(STAGE, "mrad", args.mrad, 0, MAX_RADIUS)?;
    between_i32(STAGE, "depth", args.depth, MIN_DEPTH, MAX_DEPTH)?;

    let bits = bits_of(sample);
    let peak = ((1u32 << bits) - 1) as f32;
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    let warp = WarpParams {
        element,
        bits,
        thresh: args.thresh,
        blur: BlurKind::R6,
        passes: args.blur,
        depth: args.depth,
    };

    let threshold = (f64::from(args.mthr) * f64::from(peak)).round() as f32;

    Ok(ThinSettings {
        warp,
        threshold,
        radius: args.mrad,
        peak,
    })
}

/// Warp-sharpens lines to thin them, inside a mask of the lines.
pub(crate) struct Thin;

impl StageDefinition for Thin {
    type Args = ThinArgs;

    fn package() -> &'static str {
        "lineart"
    }

    fn identifier() -> &'static str {
        "thin"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        LINEART_INPUTS
    }
}

impl FilterDefinition for Thin {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_lineart_input(STAGE, index, offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        let mask = chosen.get(MASK).copied().flatten();
        thin_settings(args, input.sample)?;
        selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
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
        let settings = thin_settings(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
        checked_mask(STAGE, &input, mask)?;

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element: settings.warp.element,
            peak: settings.peak,
        };
        let steps = mask_steps(settings.radius);
        let thin = Thinner {
            client: context.client().clone(),
            planewise,
            settings,
            steps,
        };

        Ok(Box::new(thin))
    }
}

/// Runs `lineart/thin` on every selected plane of a frame.
struct Thinner<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    settings: ThinSettings,
    steps: Vec<Step>,
}

/// Warps one plane. With `built_in_mask` set, the warped plane is merged over the input by a mask of
/// the plane's lines.
struct ThinPass<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    settings: &'a ThinSettings,
    steps: &'a [Step],
    built_in_mask: bool,
}

impl<R: Runtime> PlaneFilter for ThinPass<'_, R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        padded_warp(self.client, context, input, output, &self.settings.warp);

        if !self.built_in_mask {
            return Ok(());
        }

        let width = input.width();
        let height = input.height();
        let sizing = Sizing(context);
        let peak = self.settings.peak;

        let magnitude = Scratch::new::<f32>(context, width, height);
        let binary = Scratch::new::<f32>(context, width, height);
        let line_mask = Scratch::new::<f32>(context, width, height);
        let magnitude_plane = magnitude.plane();
        let binary_plane = binary.plane();
        let mask_plane = line_mask.plane();

        let edge_planes = NeighbourPlanes {
            input,
            output: magnitude_plane,
        };
        let limits = EdgeLimits::open(peak);
        edges(self.client, &sizing, edge_planes, Operator::Prewitt, limits);

        let binary_planes = NeighbourPlanes {
            input: magnitude_plane,
            output: binary_plane,
        };
        binarize(self.client, &sizing, binary_planes, self.settings.threshold, peak);

        let chain_planes = ChainPlanes {
            input: binary_plane,
            scratch: magnitude_plane,
            output: mask_plane,
        };
        chain(self.client, &sizing, chain_planes, self.steps);

        let merge_planes = MergePlanes {
            input,
            filtered: output,
            weight: mask_plane,
        };
        merge(self.client, &sizing, merge_planes, peak);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Thinner<R> {}

impl<R: Runtime> Filter for Thinner<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let mask = inputs.take_mask(MASK);
        let pass = ThinPass {
            client: &self.client,
            settings: &self.settings,
            steps: &self.steps,
            built_in_mask: mask.is_none(),
        };
        let masked = MaskedFilter {
            client: &self.client,
            stage: STAGE,
            format: &self.planewise.format,
            mask: mask.as_ref(),
            peak: self.planewise.peak,
            inner: &pass,
        };

        self.planewise.run(&self.client, context, input, &masked)
    }
}
