use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::deband::{DebandParams, DebandPlanes, deband, salt};
use av_toolbox_cubek::depth::Element;
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ColourRange,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::bits_of;
use crate::sizing::Sizing;

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

fn not_frames() -> StageError {
    let message = format!("{STAGE} debands frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_channel(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

/// Smooths banding by averaging random nearby samples, then adds grain, the way libplacebo does.
pub(crate) struct Placebo;

impl StageDefinition for Placebo {
    type Args = PlaceboArgs;

    fn package() -> &'static str {
        "deband"
    }

    fn identifier() -> &'static str {
        "placebo"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Placebo {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(not_frames());
        };

        if !matches!(format.sample, SampleFormat::U8 | SampleFormat::U16 { .. }) {
            return Ok(FormatSet::new());
        }

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        plane_settings(args, &input)?;

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
        let input = checked_input(inputs)?;
        let planes = plane_settings(&args, &input)?;
        let skip_all = planes.iter().all(|plane| plane.skipped);

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let client = context.client().clone();
        let debander = Debander {
            client,
            output: input,
            planes,
            element,
            iterations: args.iterations,
            radius: args.radius,
            peak,
            seed: args.seed,
            skip_all,
        };

        Ok(Box::new(debander))
    }
}

/// Debands every plane of a frame that has something to do, sharing the rest.
struct Debander<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
    planes: Vec<PlaneSettings>,
    element: Element,
    iterations: u32,
    radius: f32,
    peak: f32,
    seed: u32,
    skip_all: bool,
}

impl<R: Runtime> Debander<R> {
    /// Debands `source` into a fresh plane of the same size and hands that plane back.
    fn deband_plane(
        &self,
        context: &FrameContext,
        source: &Plane,
        settings: &PlaneSettings,
        plane_index: u32,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let (width, height) = source.dimensions();
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.output
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let Some(output) = target.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        let source_handle = source.buffer().handle();
        let output_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let output_plane = unsafe { DevicePlane::new(output_handle, width, height) };

        let planes = DebandPlanes {
            input: source_plane,
            output: output_plane,
        };
        let params = DebandParams {
            element: self.element,
            iterations: self.iterations,
            radius: self.radius,
            threshold: settings.threshold,
            grain: settings.grain,
            neutral: settings.neutral,
            peak: self.peak,
        };
        let plane_salt = salt(meta.index, self.seed, plane_index);
        let sizing = Sizing(context);

        deband(&self.client, &sizing, planes, &params, plane_salt);

        Ok(output.clone())
    }
}

impl<R: Runtime> StageInstance for Debander<R> {}

impl<R: Runtime> Filter for Debander<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        if self.skip_all {
            return Ok(input);
        }

        let meta = input.meta().clone();
        let mut planes = Vec::with_capacity(self.planes.len());

        for (plane_index, settings) in self.planes.iter().enumerate() {
            let Some(source) = input.plane(settings.channel) else {
                return Err(missing_channel(settings.channel));
            };

            if settings.skipped {
                planes.push(source.clone());
                continue;
            }

            let debanded = self.deband_plane(context, source, settings, plane_index as u32, &meta)?;
            planes.push(debanded);
        }

        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(output)
    }
}
