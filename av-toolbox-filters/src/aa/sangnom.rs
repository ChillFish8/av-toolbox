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
use av_toolbox_cubek::aa::{SangnomPlanes, sangnom_aaf, sangnom_field, sangnom_max, sangnom_scratch};
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
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::aa::direction::{Direction, FieldPass, checked_dimensions, run_direction};
use crate::bits::bits_of;
use crate::checks::between_u32;
use crate::codes::CodeValues;
use crate::masked::{
    INPUT,
    MASK,
    MASKED_INPUTS,
    MaskedFilter,
    accepts_masked_input,
    checked_input,
    checked_mask,
    masked_output,
    missing_input,
};
use crate::planes::ChannelName;
use crate::planewise::{PlaneFilter, Planewise};
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

const STAGE: &str = "aa/sangnom";

const MAX_AA: u32 = 128;

fn default_aa() -> u32 {
    48
}

/// Arguments for `aa/sangnom`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct SangnomArgs {
    /// How readily edges are followed rather than rows averaged, between 0 and 128.
    #[serde(default = "default_aa")]
    pub(crate) aa: u32,
    /// The planes to anti-alias, the first plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// The axes to anti-alias along, both when not given.
    #[serde(default)]
    pub(crate) direction: Direction,
}

/// SangNom's settings without `planes` or `direction`, as another stage nests them.
#[derive(Debug, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct SangnomTuning {
    /// How readily edges are followed rather than rows averaged, between 0 and 128.
    pub(crate) aa: u32,
}

impl Default for SangnomTuning {
    fn default() -> Self {
        Self { aa: default_aa() }
    }
}

/// `tuning.aa` once it is checked for `stage`.
pub(crate) fn checked_sangnom_tuning(stage: &str, tuning: &SangnomTuning) -> Result<u32, StageError> {
    between_u32(stage, "aa", tuning.aa, 0, MAX_AA)?;

    Ok(tuning.aa)
}

/// Whether each plane of `input` is anti-aliased, once `args` are checked against it.
fn checked_planes(args: &SangnomArgs, input: &FrameFormat) -> Result<Vec<bool>, StageError> {
    let tuning = SangnomTuning { aa: args.aa };
    checked_sangnom_tuning(STAGE, &tuning)?;

    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::First)?;
    checked_dimensions(STAGE, input, &selected, args.direction)?;

    Ok(selected)
}

/// Anti-aliases by rebuilding each field along its edges, the way SangNom2 does.
pub(crate) struct Sangnom;

impl StageDefinition for Sangnom {
    type Args = SangnomArgs;

    fn package() -> &'static str {
        "aa"
    }

    fn identifier() -> &'static str {
        "sangnom"
    }

    fn description() -> &'static str {
        "Smooths jagged edges by rebuilding rows along their edges, as SangNom2 does."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        MASKED_INPUTS
    }
}

impl FilterDefinition for Sangnom {
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
        checked_planes(args, &input)?;
        checked_mask(STAGE, &input, mask)?;

        let format = StreamFormat::Frames(input);
        let output = masked_output(input.sample, mask);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output]),
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
        let selected = checked_planes(&args, &input)?;
        checked_mask(STAGE, &input, mask)?;

        let bits = bits_of(input.sample);
        let max = sangnom_max(bits);
        let aaf = sangnom_aaf(args.aa, bits);

        let output = masked_output(input.sample, mask);
        let codes = CodeValues::new(&input, output).whole_codes();
        let planewise = Planewise {
            stage: STAGE,
            format: input,
            selected,
            codes,
        };
        let antialiaser = Antialiaser {
            client: context.client().clone(),
            planewise,
            direction: args.direction,
            aaf,
            max,
        };

        Ok(Box::new(antialiaser))
    }
}

/// One SangNom field pass at one plane's bit depth.
pub(crate) struct SangnomPass<'a, R: Runtime> {
    pub(crate) client: &'a ComputeClient<R>,
    pub(crate) aaf: f32,
    pub(crate) max: f32,
}

impl<R: Runtime> FieldPass for SangnomPass<'_, R> {
    fn run(&self, context: &FrameContext, input: DevicePlane<'_>, output: DevicePlane<'_>, offset: u32) {
        let bytes = sangnom_scratch(input.width(), input.height());
        let smoothed = context.scratch(bytes);
        let planes = SangnomPlanes {
            input,
            output,
            smoothed: smoothed.handle(),
        };
        let sizing = Sizing(context);
        sangnom_field(self.client, &sizing, planes, offset, self.aaf, self.max);
    }
}

/// Runs `aa/sangnom` on every selected plane of a frame.
struct Antialiaser<R: Runtime> {
    client: ComputeClient<R>,
    planewise: Planewise,
    direction: Direction,
    aaf: f32,
    max: f32,
}

impl<R: Runtime> PlaneFilter for Antialiaser<R> {
    fn filter(
        &self,
        context: &FrameContext,
        _plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let pass = SangnomPass {
            client: &self.client,
            aaf: self.aaf,
            max: self.max,
        };

        run_direction(&self.client, context, &pass, &pass, input, output, self.direction);

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Antialiaser<R> {}

impl<R: Runtime> Filter for Antialiaser<R> {
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
