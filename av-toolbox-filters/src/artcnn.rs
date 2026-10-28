use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::artcnn::{
    BodyPath,
    CnnError,
    Kernels,
    LayerRun,
    Maps,
    Model,
    Network,
    Precision,
    TILE_BUDGET,
    Weights,
    artcnn,
    fits_whole,
    scratch_sizes,
    tile_side,
    warm_up,
};
use av_toolbox_gpu::{BuildContext, ComputeClient, FrameContext, PlaneBuffer, Runtime};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

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

fn model_missing(stage: &str, name: ModelName, missing: MissingModel) -> StageError {
    let argument = name.argument();
    let message = format!("{stage} cannot find the weights for model = \"{argument}\". {missing}");

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

/// Finds `name`'s weight file in `store`, or the copy built into the binary.
pub(crate) fn locate(stage: &str, store: &Store, name: ModelName) -> Result<Located, StageError> {
    let model = name.model();
    let found = store.find("artcnn", model.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(stage, name, missing)),
    }
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

/// Warms the network up on `path` at the frame's size, with the last write rounded when `rounded` is set.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the warm-up"
)]
fn warm_up_on<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    network: &Network,
    path: BodyPath,
    cap: usize,
    width: u32,
    height: u32,
    rounded: bool,
) -> Result<Warmed, CnnError> {
    let tile = tile_for(network, path, cap, width, height);
    let buffers = MapBuffers::new(context, network, path, tile, width, height);
    let maps = buffers.maps();
    let sizing = Sizing(context);

    let layers = warm_up(client, &sizing, network, path, width, height, rounded, tile, maps)?;

    Ok(Warmed { tile, layers })
}

/// ArtCNN, read, uploaded and warmed up for planes of one size.
pub(crate) struct Supersampler {
    stage: &'static str,
    network: Network,
    path: BodyPath,
    tile: u32,
    rounded: bool,
    #[cfg(all(test, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    layers: Vec<LayerRun>,
}

impl Supersampler {
    /// Finds and reads `model` at `precision`, then warms it up on a `width` by `height` plane.
    ///
    /// Its output is rounded to whole codes when `rounded` is set. The fused convolution path is taken
    /// when the device offers it, falling back to the unfused convolution and then to im2col when it
    /// cannot run there. Its f16 body convolutions choose their kernel by autotune, which benchmarks
    /// during the warm-up, unless [KERNEL_ENV] pins one.
    pub(crate) fn load<R: Runtime>(
        stage: &'static str,
        context: &BuildContext<R>,
        model: ModelName,
        precision: PrecisionName,
        width: u32,
        height: u32,
        rounded: bool,
    ) -> Result<Self, StageError> {
        let pinned = std::env::var(KERNEL_ENV).ok();
        let kernels = kernels_from_value(stage, pinned.as_deref())?;

        Self::load_with(stage, context, model, precision, width, height, rounded, kernels)
    }

    /// The same as [Supersampler::load](crate::artcnn::Supersampler::load), with the body convolutions
    /// choosing their kernel by `kernels`.
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is a distinct input of the load"
    )]
    pub(crate) fn load_with<R: Runtime>(
        stage: &'static str,
        context: &BuildContext<R>,
        model: ModelName,
        precision: PrecisionName,
        width: u32,
        height: u32,
        rounded: bool,
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
            warm_up_on(
                client,
                &frame_context,
                &network,
                path,
                cap,
                width,
                height,
                rounded,
            )
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
            rounded,
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
    /// The output is rounded to whole codes when the supersampler was loaded to round. `input` must be
    /// the size given to [Supersampler::load](crate::artcnn::Supersampler::load), since the tile is
    /// chosen there.
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
            self.rounded,
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
