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
    FrameMeta,
    HostFrame,
    HostSide,
    MetaValue,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use avtbx_vship::{ButteraugliScore, VshipError};
use schemars::JsonSchema;
use serde::Deserialize;

use super::common::{Fallbacks, accepts_clip, checked_pair, color_space, pick_device, planes};
use super::output::{MapOutput, Output, declared_map, map_format, map_frame};
use super::summary::Summary;

const STAGE: &str = "vship/butteraugli";

const REFERENCE: usize = 0;
const DISTORTED: usize = 1;

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ButteraugliArgs {
    /// The norm reported as `norm_q`.
    #[serde(default = "default_qnorm")]
    pub(crate) qnorm: u32,
    /// The display brightness in nits.
    #[serde(default = "default_intensity")]
    pub(crate) intensity: f32,
    /// Outputs the distorted clip, or a map of the Butteraugli distance at each pixel.
    #[serde(default)]
    pub(crate) output: Output,
    /// The Vship device to score on, matching the pipeline's device by default.
    #[serde(default)]
    pub(crate) gpu: Option<usize>,
}

fn default_qnorm() -> u32 {
    2
}

fn default_intensity() -> f32 {
    203.0
}

/// Scores each distorted frame against its reference with Butteraugli, through Vship.
///
/// Each output frame carries `norm_q`, `norm3` and `norm_inf`, where 0 means identical. The output
/// is the distorted frame, or with `output = "map"` the per-pixel map of Butteraugli distance at the
/// clip's size. At the end of the stream it writes a `summary` record of `norm_q` and the largest
/// `norm_inf`.
pub(crate) struct Butteraugli;

impl StageDefinition for Butteraugli {
    type Args = ButteraugliArgs;

    fn package() -> &'static str {
        "vship"
    }

    fn identifier() -> &'static str {
        "butteraugli"
    }

    fn description() -> &'static str {
        "Scores how different each frame looks from a reference, where 0 means identical."
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

impl FilterDefinition for Butteraugli {
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
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let (reference, distorted) = checked_pair(chosen)?;
        checked_qnorm(args.qnorm)?;
        checked_intensity(args.intensity)?;

        let output = match args.output {
            Output::Clip => distorted,
            Output::Map => map_format(&distorted, reference.width, reference.height),
        };
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        let (reference, distorted) = checked_pair(formats.inputs())?;

        let mut fallbacks = Fallbacks::new(STAGE);
        let reference_space = color_space(&reference, &mut fallbacks)?;
        let distorted_space = color_space(&distorted, &mut fallbacks)?;

        let qnorm = checked_qnorm(args.qnorm)?;
        let intensity = checked_intensity(args.intensity)?;

        let gpu_id = pick_device(STAGE, args.gpu, context.device())?;
        let created =
            avtbx_vship::Butteraugli::new(reference_space, distorted_space, qnorm, intensity, gpu_id);
        let metric = match created {
            Ok(metric) => metric,
            Err(error) => return Err(start_error(gpu_id, error)),
        };

        let map = match args.output {
            Output::Clip => None,
            Output::Map => Some(declared_map(STAGE, formats, metric.distortion_map())?),
        };

        Ok(Box::new(ButteraugliInstance {
            metric,
            map,
            summary: Summary::default(),
            max_norm_inf: None,
        }))
    }
}

/// The `qnorm` Vship takes, refusing 0 and anything above `i32::MAX`.
pub(crate) fn checked_qnorm(qnorm: u32) -> Result<i32, StageError> {
    let converted = i32::try_from(qnorm);
    match converted {
        Ok(value) if value > 0 => Ok(value),
        _ => Err(qnorm_error(qnorm)),
    }
}

/// The display brightness, refusing one that is not a positive number of nits.
pub(crate) fn checked_intensity(intensity: f32) -> Result<f32, StageError> {
    let usable = intensity.is_finite() && intensity > 0.0;
    if !usable {
        return Err(intensity_error(intensity));
    }

    Ok(intensity)
}

fn qnorm_error(qnorm: u32) -> StageError {
    StageError::msg(format!(
        "{STAGE} takes a qnorm between 1 and {}, not {qnorm}",
        i32::MAX
    ))
}

fn intensity_error(intensity: f32) -> StageError {
    StageError::msg(format!(
        "{STAGE} takes a positive intensity in nits, not {intensity}"
    ))
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

struct ButteraugliInstance {
    metric: avtbx_vship::Butteraugli,
    map: Option<MapOutput>,
    summary: Summary,
    max_norm_inf: Option<f64>,
}

impl ButteraugliInstance {
    fn note(&mut self, score: &ButteraugliScore) {
        self.summary.push(score.norm_q);

        let largest = match self.max_norm_inf {
            Some(previous) => previous.max(score.norm_inf),
            None => score.norm_inf,
        };
        self.max_norm_inf = Some(largest);
    }
}

impl StageInstance for ButteraugliInstance {}

impl Filter<HostSide> for ButteraugliInstance {
    fn process(
        &mut self,
        context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostFrame>, StageError> {
        let Some(reference) = inputs.take_frame(REFERENCE) else {
            return Err(missing_frame_error("reference"));
        };
        let Some(mut distorted) = inputs.take_frame(DISTORTED) else {
            return Err(missing_frame_error("distorted"));
        };

        let reference_planes = planes(&reference)?;
        let distorted_planes = planes(&distorted)?;
        let computed = match &mut self.map {
            Some(output) => {
                self.metric
                    .compute_with_map(&reference_planes, &distorted_planes, &mut output.map)
            },
            None => self.metric.compute(&reference_planes, &distorted_planes),
        };
        let score = computed.map_err(compute_error)?;

        self.note(&score);

        let Some(output) = &self.map else {
            attach_score(distorted.meta_mut(), &score);
            return Ok(vec![distorted]);
        };

        let mut meta = distorted.meta().clone();
        attach_score(&mut meta, &score);
        let frame = map_frame(context, &output.format, meta, &output.map, 1.0)?;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<HostFrame>, StageError> {
        let record = context.record("summary");
        self.summary.write(record, "");

        if let Some(max_norm_inf) = self.max_norm_inf {
            record.attach("max_norm_inf", MetaValue::Float(max_norm_inf));
        }

        Ok(Vec::new())
    }
}

fn attach_score(meta: &mut FrameMeta, score: &ButteraugliScore) {
    meta.attach("norm_q", MetaValue::Float(score.norm_q));
    meta.attach("norm3", MetaValue::Float(score.norm3));
    meta.attach("norm_inf", MetaValue::Float(score.norm_inf));
}
