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
    FrameFormat,
    HostFrame,
    HostSide,
    MetaValue,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use avtbx_vship::{CvvdpOptions, DisplayModel, VshipError};
use schemars::JsonSchema;
use serde::Deserialize;

use super::common::{Fallbacks, VshipPlanes, accepts_clip, checked_pair, color_space, pick_device};
use super::output::{MapOutput, Output, declared_map, map_format, map_frame};
use super::summary::Summary;

const STAGE: &str = "vship/cvvdp";

/// The highest frame rate Vship takes, as it sizes its frame history from it.
const MAX_FPS: f64 = 1000.0;

const REFERENCE: usize = 0;
const DISTORTED: usize = 1;

/// Vship's map holds `1 - JOD / 10`, so this turns it into the JOD lost.
const JOD_LOST_PER_MAP_UNIT: f32 = 10.0;

/// The built-in displays by name, as Vship knows them.
const DISPLAYS: [(&str, DisplayModel); 6] = [
    ("standard_4k", DisplayModel::Standard4k),
    ("standard_fhd", DisplayModel::StandardFhd),
    ("standard_hdr_pq", DisplayModel::StandardHdrPq),
    ("standard_hdr_hlg", DisplayModel::StandardHdrHlg),
    ("standard_hdr_dark", DisplayModel::StandardHdrDark),
    ("standard_hdr_linear_zoom", DisplayModel::StandardHdrLinearZoom),
];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CvvdpArgs {
    /// The display to model viewing on, one of the built-in names or the one `display_config` defines.
    #[serde(default = "default_display")]
    pub(crate) display: String,
    /// Inline JSON in Vship's display config format, adding or overriding `display`.
    #[serde(default)]
    pub(crate) display_config: Option<String>,
    /// Scales frames to the display's resolution before scoring.
    #[serde(default)]
    pub(crate) resize_to_display: bool,
    /// Scores the clip as a video. When false, each frame scores alone.
    #[serde(default = "default_temporal")]
    pub(crate) temporal: bool,
    /// Outputs the distorted clip, or a map of the JOD lost at each pixel.
    #[serde(default)]
    pub(crate) output: Output,
    /// The Vship device to score on, matching the pipeline's device by default.
    #[serde(default)]
    pub(crate) gpu: Option<usize>,
}

fn default_display() -> String {
    "standard_fhd".to_string()
}

fn default_temporal() -> bool {
    true
}

/// Scores a distorted clip against its reference with CVVDP, through Vship.
///
/// Each output frame carries `score` in JOD, where 10 means no visible difference. The score covers
/// every frame so far, or with `temporal = false` only that frame. The output is the distorted
/// frame, or with `output = "map"` the frame's map of the JOD lost at each pixel. At the end of the
/// stream it writes a `summary` record of the final score.
pub(crate) struct Cvvdp;

impl StageDefinition for Cvvdp {
    type Args = CvvdpArgs;

    fn package() -> &'static str {
        "vship"
    }

    fn identifier() -> &'static str {
        "cvvdp"
    }

    fn description() -> &'static str {
        "Scores how visible the differences from a reference are on a chosen display.\n\n\
         Scores run up to 10, which means no visible difference. By default it takes motion across frames \
         into account."
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

impl FilterDefinition for Cvvdp {
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
        let options = options(args, &reference)?;

        let output = match args.output {
            Output::Clip => distorted,
            Output::Map => {
                let (width, height) = map_size(args, &options.display, &reference)?;
                map_format(&distorted, width, height)
            },
        };
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        let (reference, distorted) = checked_pair(formats.inputs())?;
        let options = options(&args, &reference)?;

        let mut fallbacks = Fallbacks::new(STAGE);
        let reference_space = color_space(&reference, &mut fallbacks)?;
        let distorted_space = color_space(&distorted, &mut fallbacks)?;

        let gpu_id = pick_device(STAGE, args.gpu, context.device())?;
        let created = avtbx_vship::Cvvdp::new(reference_space, distorted_space, &options, gpu_id);
        let metric = match created {
            Ok(metric) => metric,
            Err(error) => return Err(start_error(gpu_id, error)),
        };

        let map = match args.output {
            Output::Clip => None,
            Output::Map => Some(declared_map(STAGE, formats, metric.distortion_map())?),
        };

        Ok(Box::new(CvvdpInstance {
            metric,
            map,
            temporal: args.temporal,
            summary: Summary::default(),
            last_score: None,
        }))
    }
}

/// How Vship should model viewing for `args` on a clip of `reference`.
fn options(args: &CvvdpArgs, reference: &FrameFormat) -> Result<CvvdpOptions, StageError> {
    let display = display_model(args)?;
    let fps = frame_rate(reference)?;

    Ok(CvvdpOptions {
        fps,
        resize_to_display: args.resize_to_display,
        display,
    })
}

/// The display named by `args`, custom when `display_config` is set.
pub(crate) fn display_model(args: &CvvdpArgs) -> Result<DisplayModel, StageError> {
    if let Some(config) = &args.display_config {
        return Ok(DisplayModel::Custom {
            name: args.display.clone(),
            json: config.clone(),
        });
    }

    let found = DISPLAYS.iter().find(|(name, _)| *name == args.display);
    match found {
        Some((_, model)) => Ok(model.clone()),
        None => Err(unknown_display_error(&args.display)),
    }
}

/// The reference clip's frame rate, which CVVDP models.
pub(crate) fn frame_rate(reference: &FrameFormat) -> Result<f32, StageError> {
    let Some(rate) = reference.frame_rate else {
        return Err(missing_frame_rate_error());
    };

    let numerator = f64::from(rate.numerator());
    let denominator = f64::from(rate.denominator());

    fps_to_f32(numerator / denominator)
}

/// Narrows a frame rate to the `f32` Vship takes, accepting one above 0 and at most [MAX_FPS].
fn fps_to_f32(fps: f64) -> Result<f32, StageError> {
    let usable = fps > 0.0 && fps <= MAX_FPS;
    if !usable {
        return Err(fps_error(fps));
    }

    Ok(fps as f32)
}

/// The width and height Vship writes the distortion map at.
///
/// It is the clip's size, or with `resize_to_display` the clip scaled to fit the display while
/// keeping its aspect ratio, in the same integer steps Vship uses.
pub(crate) fn map_size(
    args: &CvvdpArgs,
    display: &DisplayModel,
    reference: &FrameFormat,
) -> Result<(u32, u32), StageError> {
    if !args.resize_to_display {
        return Ok((reference.width, reference.height));
    }

    if reference.width == 0 || reference.height == 0 {
        return Err(empty_clip_error());
    }

    let (display_width, display_height) = display_resolution(display)?;
    let width = u64::from(reference.width);
    let height = u64::from(reference.height);

    let wider_than_display = width * display_height >= height * display_width;
    let (map_width, map_height) = if wider_than_display {
        (display_width, height * display_width / width)
    } else {
        (width * display_height / height, display_height)
    };

    let converted_width = u32::try_from(map_width);
    let converted_height = u32::try_from(map_height);
    let (Ok(map_width), Ok(map_height)) = (converted_width, converted_height) else {
        return Err(map_too_large_error(map_width, map_height));
    };

    Ok((map_width, map_height))
}

/// The resolution of `display`, read from its config when it sets one.
fn display_resolution(display: &DisplayModel) -> Result<(u64, u64), StageError> {
    let DisplayModel::Custom { name, json } = display else {
        return Ok(built_in_resolution(display));
    };

    let parsed: serde_json::Value = match serde_json::from_str(json) {
        Ok(parsed) => parsed,
        Err(error) => return Err(display_config_error(error)),
    };

    let resolution = parsed
        .get(name.as_str())
        .and_then(|entry| entry.get("resolution"));
    if let Some(resolution) = resolution {
        return configured_resolution(name, resolution);
    }

    let built_in = DISPLAYS.iter().find(|(built_in_name, _)| built_in_name == name);
    match built_in {
        Some((_, model)) => Ok(built_in_resolution(model)),
        None => Err(missing_resolution_error(name)),
    }
}

/// A display's resolution as Vship's built-in models set it.
///
/// A custom display starts from Vship's default of 1920 by 1080.
fn built_in_resolution(display: &DisplayModel) -> (u64, u64) {
    match display {
        DisplayModel::StandardFhd | DisplayModel::Custom { .. } => (1920, 1080),
        DisplayModel::Standard4k
        | DisplayModel::StandardHdrPq
        | DisplayModel::StandardHdrHlg
        | DisplayModel::StandardHdrDark
        | DisplayModel::StandardHdrLinearZoom => (3840, 2160),
    }
}

/// A config's `[width, height]`, truncated to whole pixels as Vship reads it.
fn configured_resolution(name: &str, resolution: &serde_json::Value) -> Result<(u64, u64), StageError> {
    let pair = resolution.as_array().map(Vec::as_slice);
    let Some([width, height]) = pair else {
        return Err(bad_resolution_error(name));
    };

    let width = width.as_f64().filter(is_pixel_count);
    let height = height.as_f64().filter(is_pixel_count);
    let (Some(width), Some(height)) = (width, height) else {
        return Err(bad_resolution_error(name));
    };

    Ok((width as u64, height as u64))
}

fn is_pixel_count(value: &f64) -> bool {
    (1.0..=f64::from(u32::MAX)).contains(value)
}

fn unknown_display_error(name: &str) -> StageError {
    let names: Vec<&str> = DISPLAYS.iter().map(|(name, _)| *name).collect();

    StageError::msg(format!(
        "{STAGE} has no display named {name}, expected one of {}, or pass display_config",
        names.join(", ")
    ))
}

fn missing_frame_rate_error() -> StageError {
    StageError::msg("cvvdp needs the clip's frame rate, and this clip has none")
}

fn fps_error(fps: f64) -> StageError {
    StageError::msg(format!(
        "{STAGE} needs the clip's frame rate above 0 and at most {MAX_FPS}, not {fps}"
    ))
}

fn empty_clip_error() -> StageError {
    StageError::msg(format!("{STAGE} cannot resize an empty clip"))
}

fn display_config_error(error: serde_json::Error) -> StageError {
    StageError::with_source(format!("{STAGE} could not read display_config"), error)
}

fn missing_resolution_error(name: &str) -> StageError {
    StageError::msg(format!(
        "{STAGE} needs the display's resolution to resize, but display_config gives no resolution for {name}"
    ))
}

fn bad_resolution_error(name: &str) -> StageError {
    StageError::msg(format!(
        "{STAGE} needs the resolution for {name} as [width, height] of whole pixel counts"
    ))
}

fn map_too_large_error(width: u64, height: u64) -> StageError {
    StageError::msg(format!(
        "{STAGE} would write a {width} by {height} distortion map, which is too large for a frame"
    ))
}

fn start_error(gpu_id: i32, error: VshipError) -> StageError {
    StageError::with_source(format!("{STAGE} could not start on Vship device {gpu_id}"), error)
}

fn compute_error(error: VshipError) -> StageError {
    StageError::with_source(format!("{STAGE} could not score a frame"), error)
}

fn reset_error(error: VshipError) -> StageError {
    StageError::with_source(format!("{STAGE} could not reset between frames"), error)
}

fn missing_frame_error(input: &str) -> StageError {
    StageError::msg(format!("{STAGE} was not given a {input} frame"))
}

struct CvvdpInstance {
    metric: avtbx_vship::Cvvdp,
    map: Option<MapOutput>,
    temporal: bool,
    summary: Summary,
    last_score: Option<f64>,
}

impl CvvdpInstance {
    /// The score the summary reports, the last running score or the mean frame score.
    fn final_score(&self) -> Option<f64> {
        if self.temporal {
            return self.last_score;
        }

        self.summary.mean()
    }
}

impl StageInstance for CvvdpInstance {}

impl Filter<HostSide> for CvvdpInstance {
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

        let reference_vship = VshipPlanes::of(&reference)?;
        let reference_planes = reference_vship.planes();
        let distorted_vship = VshipPlanes::of(&distorted)?;
        let distorted_planes = distorted_vship.planes();
        let computed = match &mut self.map {
            Some(output) => {
                self.metric
                    .compute_with_map(&reference_planes, &distorted_planes, &mut output.map)
            },
            None => self.metric.compute(&reference_planes, &distorted_planes),
        };
        let score = computed.map_err(compute_error)?;

        if !self.temporal {
            self.metric.reset().map_err(reset_error)?;
            self.metric.reset_score().map_err(reset_error)?;
        }

        self.summary.push(score);
        self.last_score = Some(score);

        let Some(output) = &self.map else {
            distorted.meta_mut().attach("score", MetaValue::Float(score));
            return Ok(vec![distorted]);
        };

        let mut meta = distorted.meta().clone();
        meta.attach("score", MetaValue::Float(score));
        let frame = map_frame(context, &output.format, meta, &output.map, JOD_LOST_PER_MAP_UNIT)?;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<HostFrame>, StageError> {
        let frames = i64::try_from(self.summary.frames()).unwrap_or(i64::MAX);
        let score = self.final_score();

        let record = context.record("summary");
        record.attach("frames", MetaValue::Int(frames));

        if let Some(score) = score {
            record.attach("score", MetaValue::Float(score));
        }

        Ok(Vec::new())
    }
}
