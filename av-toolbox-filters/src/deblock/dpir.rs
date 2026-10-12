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
use av_toolbox_cubek::cnn::{BodyPath, CnnError, LayerRun, Precision};
use av_toolbox_cubek::dpir::{
    Dpir as Network,
    DpirMaps,
    DpirPlanes,
    PACK,
    ScratchSizes,
    Variant,
    WIDTHS,
    Weights,
    deblock,
    scratch_sizes,
    warm_up,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
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
    PlaneBuffer,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

use super::planes::plane_like;
use crate::checks::{accepts_integer_or_f32_frames, between_f32};
use crate::cnn::{
    PrecisionName,
    allocation_cap,
    body_path,
    broken_model,
    checked_precision,
    failed_run,
    kernel_summary,
    path_name,
    unreadable_model,
    warmed_path,
};
use crate::deblock::staging::{Staging, centre_of, stage_in, stage_out};
use crate::planes::ChannelName;
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

pub(crate) const STAGE: &str = "deblock/dpir";

const DEFAULT_STRENGTH: f32 = 10.0;

/// The feature width of each run of eight ResBlock convolutions, in forward order.
const LEVEL_FEATURES: [u32; 7] = [64, 128, 256, 512, 256, 128, 64];

const CONVOLUTIONS_PER_LEVEL: usize = 8;

/// How strongly to deblock, on an 8-bit scale, once for every plane or once per plane.
#[derive(Debug, Clone, PartialEq, Deserialize, JsonSchema)]
#[serde(untagged)]
pub(crate) enum StrengthArg {
    One(f32),
    PerPlane(Vec<f32>),
}

impl Default for StrengthArg {
    fn default() -> Self {
        Self::One(DEFAULT_STRENGTH)
    }
}

/// Which DPIR network runs, as the stage's arguments name it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelChoice {
    /// The colour network on RGB, the gray network otherwise.
    #[default]
    Auto,
    Gray,
    Colour,
}

/// Arguments for `deblock/dpir`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DpirArgs {
    /// How strongly to deblock, between 0 and 255. A list gives one per selected plane, gray model only.
    #[serde(default)]
    pub(crate) strength: StrengthArg,
    /// `colour` runs the RGB planes together, `gray` runs each plane on its own, `auto` picks by layout.
    #[serde(default)]
    pub(crate) model: ModelChoice,
    /// The planes the gray model runs on, every plane by default.
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// The precision of the feature maps. `f32` is slower and more exact than `f16`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
}

/// What each frame runs through the network as.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Runs {
    /// The three RGB planes together through the colour network.
    Colour { strength: f32 },
    /// Each plane on its own through the gray network, one entry per plane in channel order, `None` for a
    /// plane left alone.
    Gray { strengths: Vec<Option<f32>> },
}

impl Runs {
    pub(crate) fn variant(&self) -> Variant {
        match self {
            Self::Colour { .. } => Variant::Colour,
            Self::Gray { .. } => Variant::Gray,
        }
    }
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} deblocks frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn colour_needs_rgb(layout: FrameLayout) -> StageError {
    let message = format!(
        "{STAGE}'s colour model runs on rgb frames, not {layout} frames. Convert with resample/chroma to \
         4:4:4 and resample/matrix to rgb first, or set model to \"gray\""
    );

    StageError::msg(message)
}

fn colour_on_grey() -> StageError {
    let message =
        format!("{STAGE}'s colour model runs on rgb frames, not grey frames. Set model to \"gray\"");

    StageError::msg(message)
}

fn planes_with_colour() -> StageError {
    let message = format!(
        "{STAGE}'s colour model runs on all three rgb planes together, so it takes no planes. Set model to \
         \"gray\" to choose planes"
    );

    StageError::msg(message)
}

fn strength_list_with_colour() -> StageError {
    let message =
        format!("{STAGE}'s colour model takes one strength, not a list. Set model to \"gray\" for a list");

    StageError::msg(message)
}

fn strength_count(given: usize, planes: usize) -> StageError {
    let message = format!("{STAGE} needs one strength per selected plane, got {given} for {planes} planes");

    StageError::msg(message)
}

fn model_missing(variant: Variant, missing: MissingModel) -> StageError {
    let name = model_argument(variant);
    let message = format!("{STAGE} cannot find the weights for the {name} model. {missing}");

    StageError::msg(message)
}

fn model_argument(variant: Variant) -> &'static str {
    match variant {
        Variant::Gray => "gray",
        Variant::Colour => "colour",
    }
}

fn checked_strength(strength: f32) -> Result<f32, StageError> {
    between_f32(STAGE, "strength", strength, 0.0, 255.0)?;

    Ok(strength)
}

/// The input format, once it is known to be frames.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

/// U8, U16 and F32 frames, refusing masks.
pub(crate) fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(samples) = accepts_integer_or_f32_frames(offered) else {
        return Err(not_frames());
    };

    Ok(samples)
}

fn colour_runs(args: &DpirArgs, layout: FrameLayout) -> Result<Runs, StageError> {
    if layout == FrameLayout::Grey {
        return Err(colour_on_grey());
    }

    if layout != FrameLayout::Rgb {
        return Err(colour_needs_rgb(layout));
    }

    if args.planes.is_some() {
        return Err(planes_with_colour());
    }

    let StrengthArg::One(strength) = args.strength else {
        return Err(strength_list_with_colour());
    };

    let strength = checked_strength(strength)?;

    Ok(Runs::Colour { strength })
}

/// The plane indices to run, in the order `planes` lists them, or in channel order without it.
fn ordered_planes(args: &DpirArgs, layout: FrameLayout) -> Result<Vec<usize>, StageError> {
    let requested = args.planes.as_deref();
    selected_planes(STAGE, requested, layout, PlaneDefault::Every)?;
    let channels = layout.channels();

    let Some(names) = requested else {
        let every = (0..channels.len()).collect();

        return Ok(every);
    };

    let ordered = names
        .iter()
        .filter_map(|name| channels.iter().position(|channel| *channel == name.channel()))
        .collect();

    Ok(ordered)
}

fn gray_runs(args: &DpirArgs, layout: FrameLayout) -> Result<Runs, StageError> {
    let ordered = ordered_planes(args, layout)?;
    let mut strengths = vec![None; layout.channels().len()];

    let given = match &args.strength {
        StrengthArg::One(strength) => vec![*strength; ordered.len()],
        StrengthArg::PerPlane(list) => list.clone(),
    };

    if given.len() != ordered.len() {
        return Err(strength_count(given.len(), ordered.len()));
    }

    for (index, strength) in ordered.into_iter().zip(given) {
        let checked = checked_strength(strength)?;
        strengths[index] = Some(checked);
    }

    Ok(Runs::Gray { strengths })
}

/// Checks the arguments against the input, giving what each frame runs through the network as.
pub(crate) fn checked_runs(args: &DpirArgs, input: &FrameFormat) -> Result<Runs, StageError> {
    let model = match (args.model, input.layout) {
        (ModelChoice::Auto, FrameLayout::Rgb) => ModelChoice::Colour,
        (ModelChoice::Auto, _) => ModelChoice::Gray,
        (chosen, _) => chosen,
    };

    match model {
        ModelChoice::Colour => colour_runs(args, input.layout),
        _ => gray_runs(args, input.layout),
    }
}

/// Finds `variant`'s weight file in `store`.
pub(crate) fn locate(store: &Store, variant: Variant) -> Result<Located, StageError> {
    let found = store.find(PACK, variant.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(variant, missing)),
    }
}

fn missing_frame() -> StageError {
    let message = format!("{STAGE} expected a frame in slot 0");

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

/// The kernels `layers` ran on, grouped by the feature width of their level.
///
/// `layers` holds the ResBlock convolutions of whole forward passes in order, eight to a level. It panics
/// when `layers` is not a whole number of passes, so a change in how many runs the warm-up gives cannot
/// silently mislabel the levels.
pub(super) fn level_summary(layers: &[LayerRun]) -> String {
    let per_pass = LEVEL_FEATURES.len() * CONVOLUTIONS_PER_LEVEL;
    assert!(
        layers.len().is_multiple_of(per_pass),
        "{STAGE} expected the warm-up to give {per_pass} ResBlock convolutions per pass, got {} in all",
        layers.len(),
    );

    let parts: Vec<String> = WIDTHS
        .iter()
        .map(|width| {
            let at_width: Vec<LayerRun> = layers
                .chunks(CONVOLUTIONS_PER_LEVEL)
                .zip(LEVEL_FEATURES.iter().cycle())
                .filter(|(_, features)| *features == width)
                .flat_map(|(chunk, _)| chunk.iter().copied())
                .collect();
            let summary = kernel_summary(&at_width);

            format!("{width}: {summary}")
        })
        .collect();

    parts.join(", ")
}

/// Reads, parses and uploads `variant`'s weights.
fn load_network<R: Runtime>(
    client: &ComputeClient<R>,
    variant: Variant,
    precision: Precision,
) -> Result<Network, StageError> {
    let store = Store::from_environment();
    let located = locate(&store, variant)?;
    let file = variant.file_name();
    let read = located.read();
    let bytes = match read {
        Ok(bytes) => bytes,
        Err(error) => return Err(unreadable_model(STAGE, file, error)),
    };

    let parsed = Weights::parse(&bytes, variant);
    let weights = match parsed {
        Ok(weights) => weights,
        Err(error) => return Err(broken_model(STAGE, file, error)),
    };

    let network = Network::upload(client, &weights, precision);

    Ok(network)
}

/// The distinct plane sizes each frame runs the network at.
fn run_sizes(runs: &Runs, input: &FrameFormat) -> Vec<(u32, u32)> {
    let channels = input.layout.channels();
    let mut sizes = Vec::new();

    for (index, channel) in channels.iter().enumerate() {
        let runs_here = match runs {
            Runs::Colour { .. } => index == 0,
            Runs::Gray { strengths } => strengths[index].is_some(),
        };
        let dimensions = input
            .layout
            .channel_dimensions(*channel, input.width, input.height);

        if !runs_here {
            continue;
        }

        let Some(size) = dimensions else {
            continue;
        };

        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }

    sizes
}

/// The sixteen feature maps one run works in, four of each level's size, and the im2col columns.
struct MapBuffers {
    levels: [[PlaneBuffer; 4]; 4],
    columns: Option<PlaneBuffer>,
}

impl MapBuffers {
    fn new(context: &FrameContext, sizes: &ScratchSizes) -> Self {
        let levels = sizes
            .level_bytes
            .map(|bytes| [(); 4].map(|_| context.scratch(bytes)));
        let columns = match sizes.column_bytes {
            0 => None,
            bytes => Some(context.scratch(bytes)),
        };

        Self { levels, columns }
    }

    fn maps(&self) -> DpirMaps<'_> {
        let levels = self
            .levels
            .each_ref()
            .map(|level| level.each_ref().map(PlaneBuffer::handle));
        let columns = self.columns.as_ref().map(PlaneBuffer::handle);

        DpirMaps { levels, columns }
    }
}

/// Warms the network up at every size on `path`, giving every run's layers in turn.
fn warm_every_size<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    network: &Network,
    path: BodyPath,
    sizes: &[(u32, u32)],
    cap: usize,
) -> Result<Vec<LayerRun>, CnnError> {
    let sizing = Sizing(context);
    let precision = network.precision();
    let mut layers = Vec::new();

    for (width, height) in sizes {
        let scratch = scratch_sizes(precision, path, *width, *height, cap);
        let buffers = MapBuffers::new(context, &scratch);
        let maps = buffers.maps();

        let warmed = warm_up(
            client,
            &sizing,
            network,
            path,
            *width,
            *height,
            scratch.tile,
            maps,
        )?;
        layers.extend(warmed);
    }

    Ok(layers)
}

/// Removes compression blocking with the DPIR network, on RGB together or on each selected plane.
pub(crate) struct Dpir;

impl StageDefinition for Dpir {
    type Args = DpirArgs;

    fn package() -> &'static str {
        "deblock"
    }

    fn identifier() -> &'static str {
        "dpir"
    }

    fn description() -> &'static str {
        "Removes compression blocking with the DPIR neural network."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Stream
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Dpir {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepted_samples(offered)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        let runs = checked_runs(args, &input)?;
        checked_precision(STAGE, capabilities, args.precision)?;

        let store = Store::from_environment();
        let variant = runs.variant();
        locate(&store, variant)?;

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
        let runs = checked_runs(&args, &input)?;
        let capabilities = context.capabilities();
        let precision = checked_precision(STAGE, capabilities, args.precision)?;

        let client = context.client();
        let variant = runs.variant();
        let network = load_network(client, variant, precision)?;

        let frame_context = context.frame_context();
        let sizes = run_sizes(&runs, &input);
        let chosen = body_path(capabilities, precision);
        let cap = allocation_cap(capabilities);
        let warmed = warmed_path(STAGE, chosen, |path| {
            warm_every_size(client, &frame_context, &network, path, &sizes, cap)
        })?;
        let (path, layers) = warmed;

        let path_label = path_name(path);
        let summary = level_summary(&layers);
        tracing::info!(
            stage = STAGE,
            "warmed up on the {path_label} path, with the ResBlock convolutions on {summary}",
        );

        let staging = Staging::of(input.sample);
        let deblocker = Deblocker {
            client: client.clone(),
            network,
            path,
            runs,
            format: input,
            staging,
            cap,
        };

        Ok(Box::new(deblocker))
    }
}

/// Runs `deblock/dpir` on every frame, with the network uploaded and warmed up at build.
struct Deblocker<R: Runtime> {
    client: ComputeClient<R>,
    network: Network,
    path: BodyPath,
    runs: Runs,
    format: FrameFormat,
    staging: Staging,
    cap: usize,
}

impl<R: Runtime> Deblocker<R> {
    /// A fresh plane of `source`'s size in the clip's sample format.
    fn fresh_plane(
        &self,
        context: &FrameContext,
        source: &Plane,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let Some(plane) = plane_like(context, &self.format, source, meta) else {
            return Err(missing_channel(Channel::Y));
        };

        Ok(plane)
    }

    /// Runs `sources`, all one size, through the network at `strength` into fresh planes, in order.
    fn run(
        &self,
        context: &FrameContext,
        sources: &[(Channel, &Plane)],
        strength: f32,
        meta: &FrameMeta,
    ) -> Result<Vec<Plane>, StageError> {
        let (width, height) = sources[0].1.dimensions();
        let centres: Vec<f32> = sources
            .iter()
            .map(|(channel, _)| centre_of(self.format.sample, *channel))
            .collect();

        let staged: Vec<Scratch> = sources
            .iter()
            .zip(&centres)
            .map(|((_, plane), centre)| {
                // Safety: `staging` was made from the clip's sample format, which every source plane holds.
                unsafe { stage_in(&self.client, context, self.staging, plane, *centre) }
            })
            .collect();
        let results: Vec<Scratch> = sources
            .iter()
            .map(|_| Scratch::new::<f32>(context, width, height))
            .collect();
        let inputs: Vec<DevicePlane<'_>> = staged.iter().map(Scratch::plane).collect();
        let outputs: Vec<DevicePlane<'_>> = results.iter().map(Scratch::plane).collect();

        let planes = DpirPlanes {
            inputs: &inputs,
            outputs: &outputs,
        };
        let precision = self.network.precision();
        let scratch = scratch_sizes(precision, self.path, width, height, self.cap);
        let buffers = MapBuffers::new(context, &scratch);
        let maps = buffers.maps();
        let sizing = Sizing(context);
        let peak = self.staging.peak();
        let ran = deblock(
            &self.client,
            &sizing,
            &self.network,
            self.path,
            planes,
            strength,
            peak,
            scratch.tile,
            maps,
        );
        if let Err(error) = ran {
            return Err(failed_run(STAGE, self.path, error));
        }

        let mut written = Vec::with_capacity(sources.len());
        for (index, (_, source)) in sources.iter().enumerate() {
            let target = self.fresh_plane(context, source, meta)?;

            // Safety: `staging` was made from the clip's sample format, which `fresh_plane` allocates in.
            unsafe {
                stage_out(
                    &self.client,
                    context,
                    self.staging,
                    outputs[index],
                    &target,
                    centres[index],
                );
            }

            written.push(target);
        }

        Ok(written)
    }
}

impl<R: Runtime> StageInstance for Deblocker<R> {}

impl<R: Runtime> Filter for Deblocker<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let channels = self.format.channels();
        let mut sources = Vec::with_capacity(channels.len());
        for channel in channels {
            let Some(plane) = input.plane(*channel) else {
                return Err(missing_channel(*channel));
            };

            sources.push((*channel, plane));
        }

        let planes = match &self.runs {
            Runs::Colour { strength } => self.run(context, &sources, *strength, &meta)?,
            Runs::Gray { strengths } => {
                let mut planes = Vec::with_capacity(sources.len());

                for (source, strength) in sources.iter().zip(strengths) {
                    let Some(strength) = strength else {
                        planes.push(source.1.clone());
                        continue;
                    };

                    let ran = self.run(context, &[*source], *strength, &meta)?;
                    planes.extend(ran);
                }

                planes
            },
        };

        match Frame::from_planes(self.format, planes, meta) {
            Ok(output) => Ok(vec![output]),
            Err(error) => Err(bad_frame(error)),
        }
    }
}
