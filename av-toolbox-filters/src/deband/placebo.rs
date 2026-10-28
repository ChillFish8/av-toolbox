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
use av_toolbox_cubek::depth::Source;
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

use crate::codes::CodeValues;
use crate::scratch::Scratch;
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
    let codes = CodeValues::new(input, SampleFormat::F32);
    let bits = codes.bits();

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

    fn description() -> &'static str {
        "Smooths banding in gradients, then adds grain to hide it, as libplacebo does.\n\n\
         Banding is the visible steps that compression leaves in smooth gradients such as skies."
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

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        let planes = plane_settings(args, &input)?;
        let sample = output_sample(&planes, &input);

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([sample]),
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
        let skip_all = skips_every_plane(&planes);
        let sample = output_sample(&planes, &input);

        let codes = CodeValues::new(&input, sample);
        let output = FrameFormat { sample, ..input };
        let client = context.client().clone();
        let debander = Debander {
            client,
            output,
            planes,
            codes,
            source: source_of(input.sample),
            iterations: args.iterations,
            radius: args.radius,
            seed: args.seed,
            skip_all,
            shared: input.sample == SampleFormat::F32,
        };

        Ok(Box::new(debander))
    }
}

fn skips_every_plane(planes: &[PlaneSettings]) -> bool {
    planes.iter().all(|plane| plane.skipped)
}

/// The sample `input` is stored as once debanded with `planes`.
///
/// It is F32, unless every plane is skipped and the frame passes through in its own sample.
pub(crate) fn output_sample(planes: &[PlaneSettings], input: &FrameFormat) -> SampleFormat {
    if skips_every_plane(planes) {
        return input.sample;
    }

    SampleFormat::F32
}

/// How the deband kernel reads samples of `sample`. F32 is read as code values widened from it.
fn source_of(sample: SampleFormat) -> Source {
    match sample {
        SampleFormat::U8 => Source::U8,
        SampleFormat::U16 { .. } => Source::U16,
        SampleFormat::F32 => Source::F32,
    }
}

/// Debands every plane of a frame that has something to do into F32.
///
/// The rest are shared when the input is already F32, and converted when not. A frame with every
/// plane skipped passes through.
struct Debander<R: Runtime> {
    client: ComputeClient<R>,
    output: FrameFormat,
    planes: Vec<PlaneSettings>,
    codes: CodeValues,
    source: Source,
    iterations: u32,
    radius: f32,
    seed: u32,
    skip_all: bool,
    /// Whether planes left alone can be shared, which they can when the input is F32.
    shared: bool,
}

impl<R: Runtime> Debander<R> {
    /// Stores `source` as an F32 plane of the same size, debanded unless `settings` skips it.
    fn output_plane(
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
        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, width, height) };

        let sizing = Sizing(context);

        if settings.skipped {
            self.codes.convert(
                &self.client,
                &sizing,
                source_plane,
                target_plane,
                settings.channel,
            );
            return Ok(output.clone());
        }

        let values = Scratch::new::<f32>(context, width, height);
        let values_plane = values.plane();
        self.deband_values(context, source_plane, values_plane, settings, plane_index, meta);
        self.codes.store(
            &self.client,
            &sizing,
            values_plane,
            target_plane,
            settings.channel,
        );

        Ok(output.clone())
    }

    /// Debands `source` into f32 code values held in `values`.
    fn deband_values(
        &self,
        context: &FrameContext,
        source: DevicePlane<'_>,
        values: DevicePlane<'_>,
        settings: &PlaneSettings,
        plane_index: u32,
        meta: &FrameMeta,
    ) {
        let sizing = Sizing(context);
        let widened = if self.source == Source::F32 {
            Some(Scratch::new::<f32>(context, source.width(), source.height()))
        } else {
            None
        };

        // F32 samples are read as the code values they widen to.
        let input = match &widened {
            Some(scratch) => {
                let widened_plane = scratch.plane();
                self.codes
                    .widen(&self.client, &sizing, source, widened_plane, settings.channel);
                widened_plane
            },
            None => source,
        };

        let planes = DebandPlanes {
            input,
            output: values,
        };
        let params = DebandParams {
            element: self.source,
            iterations: self.iterations,
            radius: self.radius,
            threshold: settings.threshold,
            grain: settings.grain,
            neutral: settings.neutral,
            peak: self.codes.peak(),
        };
        let plane_salt = salt(meta.index, self.seed, plane_index);

        deband(&self.client, &sizing, planes, &params, plane_salt);
    }
}

impl<R: Runtime> StageInstance for Debander<R> {}

impl<R: Runtime> Filter for Debander<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        if self.skip_all {
            return Ok(vec![input]);
        }

        let meta = input.meta().clone();
        let mut planes = Vec::with_capacity(self.planes.len());

        for (plane_index, settings) in self.planes.iter().enumerate() {
            let Some(source) = input.plane(settings.channel) else {
                return Err(missing_channel(settings.channel));
            };

            if settings.skipped && self.shared {
                planes.push(source.clone());
                continue;
            }

            let output = self.output_plane(context, source, settings, plane_index as u32, &meta)?;
            planes.push(output);
        }

        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(vec![output])
    }
}
