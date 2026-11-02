use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
    Tag,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    FrameContext,
    HostFrame,
    HostSide,
    MetaValue,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use avtbx_vship::VshipError;
use schemars::JsonSchema;
use serde::Deserialize;

use super::common::{Fallbacks, VshipPlanes, accepts_clip, checked_pair, color_space, pick_device};
use super::summary::Summary;

const STAGE: &str = "vship/ssimulacra2";

const REFERENCE: usize = 0;
const DISTORTED: usize = 1;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Ssimulacra2Args {
    /// The Vship device to score on, matching the pipeline's device by default.
    #[serde(default)]
    pub(crate) gpu: Option<usize>,
}

/// Scores each distorted frame against its reference with SSIMULACRA2, through Vship.
///
/// Each distorted frame passes through carrying its `score`, where 100 means identical. At the end
/// of the stream it writes a `summary` record of the scores.
pub(crate) struct Ssimulacra2;

impl StageDefinition for Ssimulacra2 {
    type Args = Ssimulacra2Args;

    fn package() -> &'static str {
        "vship"
    }

    fn identifier() -> &'static str {
        "ssimulacra2"
    }

    fn description() -> &'static str {
        "Scores how close each frame looks to a reference, where 100 means identical."
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Evaluation]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("reference")
                .accepting(StreamType::Frames)
                .chunk_authority(),
            InputParam::required("distorted").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Ssimulacra2 {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let other_index = match index {
            REFERENCE => DISTORTED,
            _ => REFERENCE,
        };
        let other = inputs.get(other_index).copied().unwrap_or(Offered::Unknown);

        accepts_clip(offered, other)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let (_, distorted) = checked_pair(chosen)?;
        let format = StreamFormat::Frames(distorted);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([distorted.sample]),
        })
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        let (reference, distorted) = checked_pair(formats.inputs())?;

        let mut fallbacks = Fallbacks::new(STAGE);
        let reference_space = color_space(&reference, &mut fallbacks)?;
        let distorted_space = color_space(&distorted, &mut fallbacks)?;

        let gpu_id = pick_device(STAGE, args.gpu, context.device())?;
        let metric = match avtbx_vship::Ssimulacra2::new(reference_space, distorted_space, gpu_id) {
            Ok(metric) => metric,
            Err(error) => return Err(start_error(gpu_id, error)),
        };

        Ok(Box::new(Ssimulacra2Instance {
            metric,
            summary: Summary::default(),
        }))
    }
}

fn start_error(gpu_id: i32, error: VshipError) -> StageError {
    StageError::with_source(format!("{STAGE} could not start on Vship device {gpu_id}"), error)
}

fn compute_error(error: VshipError) -> StageError {
    StageError::with_source(format!("{STAGE} could not score a frame"), error)
}

fn missing_frame_error(input: &str) -> StageError {
    StageError::msg(format!("{STAGE} was not given a {input} frame"))
}

struct Ssimulacra2Instance {
    metric: avtbx_vship::Ssimulacra2,
    summary: Summary,
}

impl StageInstance for Ssimulacra2Instance {}

impl Filter<HostSide> for Ssimulacra2Instance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostFrame>, StageError> {
        let Some(reference) = inputs.take_frame(REFERENCE) else {
            return Err(missing_frame_error("reference"));
        };
        let Some(mut distorted) = inputs.take_frame(DISTORTED) else {
            return Err(missing_frame_error("distorted"));
        };

        let reference_vship = VshipPlanes::of(&reference)?;
        let reference_planes = reference_vship.planes();
        let distorted_vship = VshipPlanes::of(&distorted)?;
        let distorted_planes = distorted_vship.planes();
        let score = self
            .metric
            .compute(&reference_planes, &distorted_planes)
            .map_err(compute_error)?;

        self.summary.push(score);
        distorted.meta_mut().attach("score", MetaValue::Float(score));

        Ok(vec![distorted])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<HostFrame>, StageError> {
        let record = context.record("summary");
        self.summary.write(record, "");

        Ok(Vec::new())
    }
}
