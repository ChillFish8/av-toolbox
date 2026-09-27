use av_toolbox_core::StageError;
use av_toolbox_gpu::{Channel, ColourRange, FrameFormat, SampleFormat};
use schemars::JsonSchema;
use serde::Deserialize;

const STAGE: &str = "deband/placebo";

/// The most iterations a deband runs.
const MAX_ITERATIONS: u32 = 16;

/// One value for every plane, or one value per plane.
#[derive(Debug, Deserialize, JsonSchema, Clone, PartialEq)]
#[serde(untagged)]
pub(crate) enum PerPlane {
    Every(f32),
    Each(Vec<f32>),
}

fn default_iterations() -> u32 {
    1
}

fn default_threshold() -> PerPlane {
    PerPlane::Every(3.0)
}

fn default_radius() -> f32 {
    16.0
}

fn default_grain() -> PerPlane {
    PerPlane::Every(4.0)
}

/// Arguments for `deband/placebo`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct PlaceboArgs {
    /// How many rounds of averaging run, each reaching further than the last.
    #[serde(default = "default_iterations")]
    pub(crate) iterations: u32,
    /// How close an average must be to a sample to replace it, in thousandths of full scale.
    #[serde(default = "default_threshold")]
    pub(crate) threshold: PerPlane,
    /// How far the first round reaches, in samples of the plane.
    #[serde(default = "default_radius")]
    pub(crate) radius: f32,
    /// How much grain is added, in thousandths of full scale.
    #[serde(default = "default_grain")]
    pub(crate) grain: PerPlane,
    /// Changes the random pattern without changing its strength.
    #[serde(default)]
    pub(crate) seed: u32,
}

/// What one plane is debanded with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct PlaneSettings {
    pub(crate) channel: Channel,
    pub(crate) threshold: f32,
    pub(crate) grain: f32,
    /// The code value grain fades toward.
    pub(crate) neutral: f32,
    /// Whether the plane passes through untouched.
    pub(crate) skipped: bool,
}

fn too_many_iterations(iterations: u32) -> StageError {
    let message = format!("{STAGE} needs iterations between 0 and {MAX_ITERATIONS}, got {iterations}");

    StageError::msg(message)
}

fn bad_radius(radius: f32) -> StageError {
    let message = format!("{STAGE} needs a finite radius above 0, got {radius}");

    StageError::msg(message)
}

fn bad_value(name: &str, value: f32) -> StageError {
    let message = format!("{STAGE} needs every {name} value finite and at least 0, got {value}");

    StageError::msg(message)
}

fn bad_length(name: &str, length: usize, planes: usize) -> StageError {
    let message =
        format!("{STAGE} needs {name} to hold one value per plane, got {length} for {planes} planes");

    StageError::msg(message)
}

/// `value` spread over `planes` planes, once every value is checked.
fn per_plane(value: &PerPlane, name: &str, planes: usize) -> Result<Vec<f32>, StageError> {
    let values = match value {
        PerPlane::Every(single) => vec![*single; planes],
        PerPlane::Each(each) => each.clone(),
    };

    if values.len() != planes {
        return Err(bad_length(name, values.len(), planes));
    }

    let invalid = values.iter().find(|value| !value.is_finite() || **value < 0.0);
    if let Some(value) = invalid {
        return Err(bad_value(name, *value));
    }

    Ok(values)
}

/// The bits one sample of `sample` holds.
pub(crate) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        _ => 8,
    }
}

/// The code value grain fades toward on `channel`, as libplacebo's renderer sets it.
///
/// That is black for luma and RGB, and the centre for chroma.
pub(crate) fn neutral(channel: Channel, range: ColourRange, bits: u32) -> f32 {
    match (channel, range) {
        (Channel::U | Channel::V, _) => (1u32 << (bits - 1)) as f32,
        (Channel::Y, ColourRange::Full) => 0.0,
        (Channel::Y, _) => (16u32 << (bits - 8)) as f32,
        _ => 0.0,
    }
}

/// Every plane's settings for `input`, in the layout's channel order, once the arguments are checked.
pub(crate) fn plane_settings(
    args: &PlaceboArgs,
    input: &FrameFormat,
) -> Result<Vec<PlaneSettings>, StageError> {
    if args.iterations > MAX_ITERATIONS {
        return Err(too_many_iterations(args.iterations));
    }

    if !args.radius.is_finite() || args.radius <= 0.0 {
        return Err(bad_radius(args.radius));
    }

    let channels = input.channels();
    let thresholds = per_plane(&args.threshold, "threshold", channels.len())?;
    let grains = per_plane(&args.grain, "grain", channels.len())?;
    let bits = bits_of(input.sample);

    let settings = channels
        .iter()
        .zip(thresholds)
        .zip(grains)
        .map(|((channel, threshold), grain)| {
            let smooths = threshold > 0.0 && args.iterations > 0;
            let skipped = !smooths && grain == 0.0;

            PlaneSettings {
                channel: *channel,
                threshold,
                grain,
                neutral: neutral(*channel, input.colour.range, bits),
                skipped,
            }
        })
        .collect();

    Ok(settings)
}
