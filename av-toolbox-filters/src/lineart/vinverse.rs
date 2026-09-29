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
use av_toolbox_cubek::lineart::{VinverseParams, VinversePlanes, vinverse};
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
use crate::checks::{at_least, between_f32};
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "lineart/vinverse";

const MAX_SCL: f32 = 1.0;

const MAX_AMNT: f32 = 255.0;

fn default_contra_str() -> f32 {
    2.7
}

fn default_scl() -> f32 {
    0.25
}

/// Arguments for `lineart/vinverse`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct VinverseArgs {
    /// How strongly the remaining combing is countered.
    #[serde(default = "default_contra_str")]
    pub(crate) contra_str: f32,
    /// How much of a correction that fights the input is kept, between 0 and 1.
    #[serde(default = "default_scl")]
    pub(crate) scl: f32,
    /// The most a sample may move from the input, on the 8-bit scale. Unlimited when not given.
    #[serde(default)]
    pub(crate) amnt: Option<f32>,
    /// The planes to clean, every plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// The vinverse parameters for `sample`, once the arguments are checked.
pub(crate) fn vinverse_params(
    args: &VinverseArgs,
    sample: SampleFormat,
) -> Result<VinverseParams, StageError> {
    at_least(STAGE, "contra_str", args.contra_str, 0.0)?;
    between_f32(STAGE, "scl", args.scl, 0.0, MAX_SCL)?;

    let scale = (1u32 << (bits_of(sample) - 8)) as f32;
    let amnt = match args.amnt {
        Some(amnt) => {
            between_f32(STAGE, "amnt", amnt, 0.0, MAX_AMNT)?;

            Some(amnt * scale)
        },
        None => None,
    };

    Ok(VinverseParams {
        contra_str: args.contra_str,
        scl: args.scl,
        amnt,
    })
}

/// Removes residual combing, the way vs-jetpack's vinverse does.
pub(crate) struct Vinverse;

impl StageDefinition for Vinverse {
    type Args = VinverseArgs;

    fn package() -> &'static str {
        "lineart"
    }

    fn identifier() -> &'static str {
        "vinverse"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        LINEART_INPUTS
    }
}

impl FilterDefinition for Vinverse {
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
        vinverse_params(args, input.sample)?;
        selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::Every)?;
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
        let params = vinverse_params(&args, input.sample)?;
        let selected = selected_planes(STAGE, args.planes.as_deref(), input.layout, PlaneDefault::Every)?;
        checked_mask(STAGE, &input, mask)?;

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;

        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            element,
            peak,
        };
        let cleaner = Cleaner {
            client: context.client().clone(),
            planewise,
            params,
        };

        Ok(Box::new(cleaner))
    }
}

/// Runs `lineart/vinverse` on every selected plane of a frame.
struct Cleaner<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    params: VinverseParams,
}

impl<R: Runtime> PlaneFilter for Cleaner<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let width = input.width();
        let height = input.height();

        let blurred = Scratch::new::<f32>(context, width, height);
        let diff = Scratch::new::<f32>(context, width, height);
        let diff_blurred = Scratch::new::<f32>(context, width, height);
        let sbr = Scratch::new::<f32>(context, width, height);
        let planes = VinversePlanes {
            input,
            output,
            blurred: blurred.plane(),
            diff: diff.plane(),
            diff_blurred: diff_blurred.plane(),
            sbr: sbr.plane(),
        };
        let sizing = Sizing(context);
        vinverse(&self.client, &sizing, planes, &self.params);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Cleaner<R> {}

impl<R: Runtime> Filter for Cleaner<R> {
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
