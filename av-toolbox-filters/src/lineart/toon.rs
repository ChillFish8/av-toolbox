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
use av_toolbox_cubek::lineart::{
    ToonDifferencePlanes,
    ToonFinishPlanes,
    ToonSettings,
    toon_difference,
    toon_finish,
};
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

use crate::bits::bits_of;
use crate::checks::{at_least, between_f32, between_i32, between_u32};
use crate::codes::CodeValues;
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
use crate::padded_warp::padded_warp;
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "lineart/toon";

const MAX_EIGHT_BIT: f32 = 255.0;

const MAX_BLUR: u32 = 100;

const MIN_DEPTH: i32 = -128;

const MAX_DEPTH: i32 = 127;

fn default_strength() -> f32 {
    1.0
}

fn default_l_thr() -> f32 {
    2.0
}

fn default_u_thr() -> f32 {
    12.0
}

fn default_blur() -> u32 {
    2
}

fn default_depth() -> i32 {
    32
}

/// Arguments for `lineart/toon`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToonArgs {
    /// How strongly lines are deepened.
    #[serde(default = "default_strength")]
    pub(crate) strength: f32,
    /// The darkness at or below which a line is left alone, on the 8-bit scale. It sits below `u_thr`.
    #[serde(default = "default_l_thr")]
    pub(crate) l_thr: f32,
    /// The darkness at or above which a line is deepened in full, on the 8-bit scale.
    #[serde(default = "default_u_thr")]
    pub(crate) u_thr: f32,
    /// How many times the warp's edge mask is blurred, between 0 and 100.
    #[serde(default = "default_blur")]
    pub(crate) blur: u32,
    /// How far the warp thins lines, between -128 and 127.
    #[serde(default = "default_depth")]
    pub(crate) depth: i32,
    /// The planes to darken, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

fn thresholds_out_of_order(l_thr: f32, u_thr: f32) -> StageError {
    let message = format!("{STAGE} needs l_thr below u_thr, got l_thr {l_thr} and u_thr {u_thr}");

    StageError::msg(message)
}

/// The toon settings for `sample`, once the arguments are checked.
pub(crate) fn toon_settings(args: &ToonArgs, sample: SampleFormat) -> Result<ToonSettings, StageError> {
    at_least(STAGE, "strength", args.strength, 0.0)?;
    between_f32(STAGE, "l_thr", args.l_thr, 0.0, MAX_EIGHT_BIT)?;
    between_f32(STAGE, "u_thr", args.u_thr, 0.0, MAX_EIGHT_BIT)?;
    between_u32(STAGE, "blur", args.blur, 0, MAX_BLUR)?;
    between_i32(STAGE, "depth", args.depth, MIN_DEPTH, MAX_DEPTH)?;

    if args.l_thr >= args.u_thr {
        return Err(thresholds_out_of_order(args.l_thr, args.u_thr));
    }

    let bits = bits_of(sample);
    let scale = (1u32 << (bits - 8)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;
    let element = if bits == 8 { Element::U8 } else { Element::U16 };

    let warp = WarpParams {
        element,
        bits,
        thresh: 128,
        blur: BlurKind::R6,
        passes: args.blur,
        depth: args.depth,
    };

    Ok(ToonSettings {
        strength: args.strength,
        l: args.l_thr * scale,
        u: args.u_thr * scale,
        n: (1u32 << (bits - 1)) as f32,
        peak,
        warp,
    })
}

/// Darkens lines by a warp-thinned map of how dark they are, as havsfunc's Toon does.
pub(crate) struct Toon;

impl StageDefinition for Toon {
    type Args = ToonArgs;

    fn package() -> &'static str {
        "lineart"
    }

    fn identifier() -> &'static str {
        "toon"
    }

    fn description() -> &'static str {
        "Deepens dark lines in proportion to how dark they already are, for line art."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Toon {
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
        toon_settings(args, input.sample)?;
        selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
        checked_mask(STAGE, &input, mask)?;

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
        let input = checked_input(STAGE, inputs)?;
        let mask = inputs.get(MASK).copied().flatten();
        let settings = toon_settings(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
        checked_mask(STAGE, &input, mask)?;

        let codes = CodeValues::new(&input, SampleFormat::F32);
        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            codes,
        };
        let toon = Toner {
            client: context.client().clone(),
            planewise,
            settings,
        };

        Ok(Box::new(toon))
    }
}

/// Runs `lineart/toon` on every selected plane of a frame.
struct Toner<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    settings: ToonSettings,
}

impl<R: Runtime> PlaneFilter for Toner<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let width = input.width();
        let height = input.height();
        let sizing = Sizing(context);

        let grown = Scratch::new::<f32>(context, width, height);
        let closed = Scratch::new::<f32>(context, width, height);
        let diff = Scratch::new::<f32>(context, width, height);
        let shifted = Scratch::new::<f32>(context, width, height);
        let difference_planes = ToonDifferencePlanes {
            input,
            grown: grown.plane(),
            closed: closed.plane(),
            diff: diff.plane(),
            shifted: shifted.plane(),
        };
        toon_difference(&self.client, &sizing, difference_planes, &self.settings);

        let warped_plane = closed.plane();
        let shifted_plane = shifted.plane();
        padded_warp(
            &self.client,
            context,
            shifted_plane,
            warped_plane,
            &self.settings.warp,
        );

        let finish_planes = ToonFinishPlanes {
            input,
            output,
            diff: diff.plane(),
            warped: warped_plane,
            widest: grown.plane(),
        };
        toon_finish(&self.client, &sizing, finish_planes, &self.settings);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Toner<R> {}

impl<R: Runtime> Filter for Toner<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let mask = inputs.take_mask(MASK);
        let masked = MaskedFilter {
            client: &self.client,
            stage: STAGE,
            format: &self.planewise.format,
            mask: mask.as_ref(),
            codes: self.planewise.codes,
            inner: &*self,
        };

        let output = self.planewise.run(&self.client, context, input, &masked)?;

        Ok(vec![output])
    }
}
