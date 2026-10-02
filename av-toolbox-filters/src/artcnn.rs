use std::io;

use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyPath,
    Kernels,
    LayerRun,
    Maps,
    Model,
    Network,
    Precision,
    TILE_BUDGET,
    Weights,
    WeightsError,
    artcnn,
    fits_whole,
    scratch_sizes,
    tile_side,
    warm_up,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ComputeClient,
    ElementType,
    FrameContext,
    PlaneBuffer,
    Runtime,
};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::sizing::Sizing;

/// The variable that pins ArtCNN's body convolutions to one fixed kernel.
pub(crate) const KERNEL_ENV: &str = "AVTB_ARTCNN_KERNEL";

/// An ArtCNN network, as a stage's arguments name it.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelName {
    #[default]
    R8f64,
    R16f96,
}

/// The precision of ArtCNN's feature maps, as a stage's arguments name it.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum PrecisionName {
    #[default]
    F16,
    F32,
}

impl ModelName {
    pub(crate) fn model(self) -> Model {
        match self {
            Self::R8f64 => Model::R8F64,
            Self::R16f96 => Model::R16F96,
        }
    }

    fn argument(self) -> &'static str {
        match self {
            Self::R8f64 => "r8f64",
            Self::R16f96 => "r16f96",
        }
    }
}

impl PrecisionName {
    pub(crate) fn precision(self) -> Precision {
        match self {
            Self::F16 => Precision::F16,
            Self::F32 => Precision::F32,
        }
    }
}

fn model_missing(stage: &str, name: ModelName, missing: MissingModel) -> StageError {
    let argument = name.argument();
    let message = format!("{stage} cannot find the weights for model = \"{argument}\". {missing}");

    StageError::msg(message)
}

fn no_f16_arithmetic(stage: &str) -> StageError {
    let message = format!(
        "{stage} uses precision f16 by default, which needs f16 arithmetic this device lacks. \
         Use precision = \"f32\"",
    );

    StageError::msg(message)
}

fn unreadable_model(stage: &str, file: &str, error: io::Error) -> StageError {
    let message = format!("{stage} could not read its model {file}: {error}");

    StageError::msg(message)
}

fn broken_model(stage: &str, file: &str, error: WeightsError) -> StageError {
    let message = format!("{stage} could not load its model {file}: {error}");

    StageError::msg(message)
}

fn unknown_kernel(stage: &str, value: &str) -> StageError {
    let message = format!(
        "{stage} does not understand {KERNEL_ENV}=\"{value}\". Set it to \"fixed\" to pin one kernel, or \
         leave it unset or empty to tune"
    );

    StageError::msg(message)
}

/// How the body convolutions choose their kernel, from the value of [KERNEL_ENV].
///
/// `fixed` runs one fixed kernel, the same on every device that offers it. Unset or empty, autotune chooses.
pub(crate) fn kernels_from_value(stage: &str, value: Option<&str>) -> Result<Kernels, StageError> {
    let Some(text) = value.map(str::trim).filter(|text| !text.is_empty()) else {
        return Ok(Kernels::Tuned);
    };

    if text.eq_ignore_ascii_case("fixed") {
        return Ok(Kernels::Fixed);
    }

    Err(unknown_kernel(stage, text))
}

fn path_name(path: BodyPath) -> &'static str {
    match path {
        BodyPath::Convolution => "fused convolution",
        BodyPath::UnfusedConvolution => "unfused convolution",
        BodyPath::Im2col => "im2col",
    }
}

fn failed_warm_up(stage: &str, path: BodyPath, error: ArtcnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{stage} could not warm up on the {name} path: {error}");

    StageError::msg(message)
}

fn failed_run(stage: &str, path: BodyPath, error: ArtcnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{stage} could not run on the {name} path: {error}");

    StageError::msg(message)
}

/// The error for a warm-up that failed on every path it tried, naming each path and its cause in turn.
pub(crate) fn failed_every_path(stage: &str, failures: Vec<(BodyPath, ArtcnnError)>) -> StageError {
    let mut causes: Vec<String> = failures
        .iter()
        .map(|(path, error)| {
            let name = path_name(*path);

            format!("on the {name} path ({error})")
        })
        .collect();
    let last = causes.pop().unwrap_or_default();
    let earlier = causes.join(", ");
    let message = match causes.is_empty() {
        true => format!("{stage} could not warm up {last}"),
        false => format!("{stage} could not warm up {earlier} or {last}"),
    };

    StageError::msg(message)
}

/// The paths a warm-up tries in turn from `chosen`, each falling back to the next.
pub(crate) fn fallbacks(chosen: BodyPath) -> &'static [BodyPath] {
    match chosen {
        BodyPath::Convolution => &[
            BodyPath::Convolution,
            BodyPath::UnfusedConvolution,
            BodyPath::Im2col,
        ],
        BodyPath::UnfusedConvolution => &[BodyPath::UnfusedConvolution, BodyPath::Im2col],
        BodyPath::Im2col => &[BodyPath::Im2col],
    }
}

/// Finds `name`'s weight file in `store`, or the copy built into the binary.
pub(crate) fn locate(stage: &str, store: &Store, name: ModelName) -> Result<Located, StageError> {
    let model = name.model();
    let found = store.find("artcnn", model.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(stage, name, missing)),
    }
}

/// The feature maps' precision, refusing f16 on a device without f16 arithmetic.
pub(crate) fn checked_precision(
    stage: &str,
    capabilities: &Capabilities,
    precision: PrecisionName,
) -> Result<Precision, StageError> {
    let computes_f16 = capabilities.can_compute(ElementType::F16);

    if precision == PrecisionName::F16 && !computes_f16 {
        return Err(no_f16_arithmetic(stage));
    }

    let chosen = precision.precision();
    Ok(chosen)
}

/// The body path for `precision`, taking the convolution path when the device has cooperative matrices
/// for it.
pub(crate) fn body_path(capabilities: &Capabilities, precision: Precision) -> BodyPath {
    let operands = match precision {
        Precision::F16 => ElementType::F16,
        Precision::F32 => ElementType::F32,
    };
    let offered = capabilities.has_cooperative_matrix(operands, ElementType::F32);

    if offered {
        BodyPath::Convolution
    } else {
        BodyPath::Im2col
    }
}

/// The device's largest single buffer, treating an unset limit of 0 as no cap.
pub(crate) fn allocation_cap(capabilities: &Capabilities) -> usize {
    let limit = capabilities.max_allocation;

    if limit == 0 {
        return usize::MAX;
    }

    usize::try_from(limit).unwrap_or(usize::MAX)
}

/// Reads, parses and uploads `name`'s weights in `precision`'s type.
fn load_network<R: Runtime>(
    stage: &str,
    client: &ComputeClient<R>,
    store: &Store,
    name: ModelName,
    precision: Precision,
) -> Result<Network, StageError> {
    let model = name.model();
    let file = model.file_name();
    let located = locate(stage, store, name)?;
    let read = located.read();
    let bytes = match read {
        Ok(bytes) => bytes,
        Err(error) => return Err(unreadable_model(stage, file, error)),
    };

    let parsed = Weights::parse(&bytes, model.features(), model.blocks());
    let weights = match parsed {
        Ok(weights) => weights,
        Err(error) => return Err(broken_model(stage, file, error)),
    };

    let uploaded = Network::upload_model(client, model, &weights, precision);
    match uploaded {
        Ok(network) => Ok(network),
        Err(error) => Err(broken_model(stage, file, error)),
    }
}

/// The tile side a `width` by `height` frame runs at on `path`.
///
/// That is the frame's longer side, one tile, when the whole frame fits the scratch budget and `cap`, the
/// device's largest buffer. Otherwise it is the largest square tile they allow.
pub(crate) fn frame_tile(
    features: u32,
    halo: u32,
    precision: Precision,
    path: BodyPath,
    cap: usize,
    width: u32,
    height: u32,
) -> u32 {
    let whole = fits_whole(features, precision, path, width, height, TILE_BUDGET, cap);

    if whole {
        return width.max(height);
    }

    tile_side(features, halo, precision, path, TILE_BUDGET, cap)
}

fn tile_for(network: &Network, path: BodyPath, cap: usize, width: u32, height: u32) -> u32 {
    let features = network.features();
    let halo = network.halo();
    let precision = network.precision();

    frame_tile(features, halo, precision, path, cap, width, height)
}

/// The four feature maps and, on the im2col path, the columns that a frame's tiles work in.
struct MapBuffers {
    conv0: PlaneBuffer,
    first: PlaneBuffer,
    second: PlaneBuffer,
    third: PlaneBuffer,
    columns: Option<PlaneBuffer>,
}

impl MapBuffers {
    /// Takes scratch for a `width` by `height` frame split into tiles of `tile`.
    fn new(
        context: &FrameContext,
        network: &Network,
        path: BodyPath,
        tile: u32,
        width: u32,
        height: u32,
    ) -> Self {
        let sizes = scratch_sizes(network, path, tile, width, height);
        let conv0 = context.scratch(sizes.conv0_bytes);
        let first = context.scratch(sizes.map_bytes);
        let second = context.scratch(sizes.map_bytes);
        let third = context.scratch(sizes.map_bytes);
        let columns = match path {
            BodyPath::Convolution | BodyPath::UnfusedConvolution => None,
            BodyPath::Im2col => Some(context.scratch(sizes.column_bytes)),
        };

        Self {
            conv0,
            first,
            second,
            third,
            columns,
        }
    }

    fn maps(&self) -> Maps<'_> {
        let columns = self.columns.as_ref().map(PlaneBuffer::handle);

        Maps {
            conv0: self.conv0.handle(),
            first: self.first.handle(),
            second: self.second.handle(),
            third: self.third.handle(),
            columns,
        }
    }
}

/// What a warm-up chose: the tile side, and how each body layer of each tile ran.
struct Warmed {
    tile: u32,
    layers: Vec<LayerRun>,
}

/// Warms the network up on `path` at the frame's size.
fn warm_up_on<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    network: &Network,
    path: BodyPath,
    cap: usize,
    width: u32,
    height: u32,
) -> Result<Warmed, ArtcnnError> {
    let tile = tile_for(network, path, cap, width, height);
    let buffers = MapBuffers::new(context, network, path, tile, width, height);
    let maps = buffers.maps();
    let sizing = Sizing(context);

    let layers = warm_up(client, &sizing, network, path, width, height, tile, maps)?;

    Ok(Warmed { tile, layers })
}

/// The kernels `layers` ran on, each named once with how many layers ran on it.
fn kernel_summary(layers: &[LayerRun]) -> String {
    let mut counts: Vec<(String, usize)> = Vec::new();

    for layer in layers {
        let name = match layer.candidate {
            Some(candidate) => candidate.name(),
            None => "cubek".to_string(),
        };

        match counts.iter_mut().find(|(known, _)| *known == name) {
            Some((_, count)) => *count += 1,
            None => counts.push((name, 1)),
        }
    }

    let parts: Vec<String> = counts
        .iter()
        .map(|(name, count)| format!("{name} x{count}"))
        .collect();

    parts.join(", ")
}

/// Warms up on each of [fallbacks](crate::artcnn::fallbacks) from `chosen` in turn with `warm`, giving the
/// first path that succeeds and what its warm-up gave.
pub(crate) fn warmed_path<T>(
    stage: &'static str,
    chosen: BodyPath,
    mut warm: impl FnMut(BodyPath) -> Result<T, ArtcnnError>,
) -> Result<(BodyPath, T), StageError> {
    let ladder = fallbacks(chosen);
    let mut failures = Vec::new();

    for (index, path) in ladder.iter().copied().enumerate() {
        let warmed = warm(path);

        let error = match warmed {
            Ok(outcome) => return Ok((path, outcome)),
            Err(error) => error,
        };

        if let Some(next) = ladder.get(index + 1) {
            let name = path_name(path);
            let next_name = path_name(*next);
            tracing::warn!(
                stage = stage,
                reason = %error,
                "the {name} path could not warm up, so the stage falls back to the {next_name} path",
            );
        }

        failures.push((path, error));
    }

    if failures.len() == 1 {
        let (path, error) = failures.remove(0);

        return Err(failed_warm_up(stage, path, error));
    }

    Err(failed_every_path(stage, failures))
}

/// ArtCNN, read, uploaded and warmed up for planes of one size.
pub(crate) struct Supersampler {
    stage: &'static str,
    network: Network,
    path: BodyPath,
    tile: u32,
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    layers: Vec<LayerRun>,
}

impl Supersampler {
    /// Finds and reads `model` at `precision`, then warms it up on a `width` by `height` plane.
    ///
    /// The fused convolution path is taken when the device offers it, falling back to the unfused
    /// convolution and then to im2col when it cannot run there. Its f16 body convolutions choose their
    /// kernel by autotune, which benchmarks during the warm-up, unless [KERNEL_ENV] pins one.
    pub(crate) fn load<R: Runtime>(
        stage: &'static str,
        context: &BuildContext<R>,
        model: ModelName,
        precision: PrecisionName,
        width: u32,
        height: u32,
    ) -> Result<Self, StageError> {
        let pinned = std::env::var(KERNEL_ENV).ok();
        let kernels = kernels_from_value(stage, pinned.as_deref())?;

        Self::load_with(stage, context, model, precision, width, height, kernels)
    }

    /// The same as [Supersampler::load](crate::artcnn::Supersampler::load), with the body convolutions
    /// choosing their kernel by `kernels`.
    pub(crate) fn load_with<R: Runtime>(
        stage: &'static str,
        context: &BuildContext<R>,
        model: ModelName,
        precision: PrecisionName,
        width: u32,
        height: u32,
        kernels: Kernels,
    ) -> Result<Self, StageError> {
        let capabilities = context.capabilities();
        let precision = checked_precision(stage, capabilities, precision)?;

        let client = context.client();
        let store = Store::from_environment();
        let uploaded = load_network(stage, client, &store, model, precision)?;
        let network = uploaded.with_kernels(kernels);

        let frame_context = context.frame_context();
        let chosen = body_path(capabilities, precision);
        let cap = allocation_cap(capabilities);
        let warmed = warmed_path(stage, chosen, |path| {
            warm_up_on(client, &frame_context, &network, path, cap, width, height)
        })?;
        let (path, outcome) = warmed;

        let path_label = path_name(path);
        let summary = kernel_summary(&outcome.layers);
        tracing::info!(
            stage = stage,
            "warmed up on the {path_label} path, with the body convolutions on {summary}",
        );

        Ok(Self {
            stage,
            network,
            path,
            tile: outcome.tile,
            #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
            layers: outcome.layers,
        })
    }

    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn path(&self) -> BodyPath {
        self.path
    }

    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn tile(&self) -> u32 {
        self.tile
    }

    /// How the body convolutions choose their kernel.
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn kernels(&self) -> &Kernels {
        self.network.kernels()
    }

    /// The model the network was uploaded as.
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn model(&self) -> Option<Model> {
        self.network.model()
    }

    /// How each body layer of each tile ran in the warm-up.
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    pub(crate) fn layers(&self) -> &[LayerRun] {
        &self.layers
    }

    /// Doubles `input` into `output`, both f32 code values up to `peak`.
    ///
    /// The output is rounded to code values. `input` must be the size given to
    /// [Supersampler::load](crate::artcnn::Supersampler::load), since the tile is chosen there.
    pub(crate) fn run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        context: &FrameContext,
        input: DevicePlane<'_>,
        peak: f32,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError> {
        let width = input.width();
        let height = input.height();
        let buffers = MapBuffers::new(context, &self.network, self.path, self.tile, width, height);
        let maps = buffers.maps();
        let sizing = Sizing(context);

        let ran = artcnn(
            client,
            &sizing,
            &self.network,
            self.path,
            input,
            peak,
            self.tile,
            maps,
            output,
        );

        match ran {
            Ok(()) => Ok(()),
            Err(error) => Err(failed_run(self.stage, self.path, error)),
        }
    }
}
