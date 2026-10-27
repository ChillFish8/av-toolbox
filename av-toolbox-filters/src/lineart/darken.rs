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
use av_toolbox_cubek::lineart::{DarkenPlanes, DarkenSettings, darken};
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
use crate::checks::between_f32;
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
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "lineart/darken";

const MAX_STRENGTH: f32 = 256.0;

const MAX_PROTECTION: f32 = 50.0;

const MAX_EIGHT_BIT: f32 = 255.0;

const MAX_THINNING: f32 = 256.0;

fn default_strength() -> f32 {
    48.0
}

fn default_protection() -> f32 {
    5.0
}

fn default_luma_cap() -> f32 {
    191.0
}

fn default_threshold() -> f32 {
    4.0
}

/// Arguments for `lineart/darken`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DarkenArgs {
    /// How strongly lines are deepened, between 0 and 256.
    #[serde(default = "default_strength")]
    pub(crate) strength: f32,
    /// How much a line's surroundings are kept from brightening it, between 0 and 50.
    #[serde(default = "default_protection")]
    pub(crate) protection: f32,
    /// The brightest a line's surroundings count as, on the 8-bit scale.
    #[serde(default = "default_luma_cap")]
    pub(crate) luma_cap: f32,
    /// How far below its surroundings a sample must sit to be a line, on the 8-bit scale.
    #[serde(default = "default_threshold")]
    pub(crate) threshold: f32,
    /// How hard lines are thinned, between 0 and 256. Thinning is off at 0.
    #[serde(default)]
    pub(crate) thinning: f32,
    /// The planes to darken, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// The darken settings for `sample`, once the arguments are checked.
pub(crate) fn darken_settings(args: &DarkenArgs, sample: SampleFormat) -> Result<DarkenSettings, StageError> {
    between_f32(STAGE, "strength", args.strength, 0.0, MAX_STRENGTH)?;
    between_f32(STAGE, "protection", args.protection, 0.0, MAX_PROTECTION)?;
    between_f32(STAGE, "luma_cap", args.luma_cap, 0.0, MAX_EIGHT_BIT)?;
    between_f32(STAGE, "threshold", args.threshold, 0.0, MAX_EIGHT_BIT)?;
    between_f32(STAGE, "thinning", args.thinning, 0.0, MAX_THINNING)?;

    let bits = bits_of(sample);
    let scale = (1u32 << (bits - 8)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;

    Ok(DarkenSettings {
        s: args.strength / 128.0,
        limit: peak / (args.protection + 1.0),
        cap: args.luma_cap * scale,
        thr: args.threshold * scale,
        t: args.thinning / 16.0,
        n: (1u32 << (bits - 1)) as f32,
        peak,
    })
}

/// Darkens thin lines, as vs-jetpack's fast_line_darken does.
pub(crate) struct Darken;

impl StageDefinition for Darken {
    type Args = DarkenArgs;

    fn package() -> &'static str {
        "lineart"
    }

    fn identifier() -> &'static str {
        "darken"
    }

    fn description() -> &'static str {
        "Darkens thin dark lines, mainly for anime and other line art."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Darken {
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
        darken_settings(args, input.sample)?;
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
        let settings = darken_settings(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::First)?;
        checked_mask(STAGE, &input, mask)?;

        let codes = CodeValues::new(&input, SampleFormat::F32);
        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            codes,
        };
        let darkener = Darkener {
            client: context.client().clone(),
            planewise,
            settings,
        };

        Ok(Box::new(darkener))
    }
}

/// Runs `lineart/darken` on every selected plane of a frame.
struct Darkener<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    settings: DarkenSettings,
}

impl<R: Runtime> PlaneFilter for Darkener<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let width = input.width();
        let height = input.height();

        let maxed = Scratch::new::<f32>(context, width, height);
        let limited = Scratch::new::<f32>(context, width, height);
        let closed = Scratch::new::<f32>(context, width, height);
        let diff = Scratch::new::<f32>(context, width, height);
        let planes = DarkenPlanes {
            input,
            output,
            maxed: maxed.plane(),
            limited: limited.plane(),
            closed: closed.plane(),
            diff: diff.plane(),
            lines: closed.plane(),
            blurred: limited.plane(),
        };
        let sizing = Sizing(context);
        darken(&self.client, &sizing, planes, &self.settings);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Darkener<R> {}

impl<R: Runtime> Filter for Darkener<R> {
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
