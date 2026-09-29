use std::io;

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
use av_toolbox_cubek::artcnn::{
    ArtcnnError,
    BodyPath,
    Maps,
    Model,
    Network,
    Precision,
    TILE_BUDGET,
    Weights,
    WeightsError,
    artcnn,
    scratch_sizes,
    tile_side,
    warm_up,
};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::resize::{Kernel, Samples};
use av_toolbox_cubek::values::{narrow, widen};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
    DeviceSide,
    ElementType,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    Plane,
    PlaneBuffer,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_models::{Located, MissingModel, Store};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::bits::bits_of;
use crate::resize::{Tables, Window, geometry, resize_plane, samples_for};
use crate::scratch::Scratch;
use crate::sizing::Sizing;

const STAGE: &str = "upscale/artcnn";

/// Catrom, the bicubic the chroma planes are resized with.
const CATROM: Kernel = Kernel::Bicubic { b: 0.0, c: 0.5 };

/// Arguments for `upscale/artcnn`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtcnnArgs {
    /// The network: `r8f64`, or the larger and slower `r16f96`.
    #[serde(default)]
    pub(crate) model: ModelName,

    /// The feature maps' precision: `f16`, or the slower and more exact `f32`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
}

#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum ModelName {
    #[default]
    R8f64,
    R16f96,
}

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
}

impl PrecisionName {
    pub(crate) fn precision(self) -> Precision {
        match self {
            Self::F16 => Precision::F16,
            Self::F32 => Precision::F32,
        }
    }
}

fn upscalable_layout(layout: FrameLayout) -> bool {
    matches!(layout, FrameLayout::Grey | FrameLayout::Yuv { .. })
}

fn upscalable_sample(sample: SampleFormat) -> bool {
    matches!(sample, SampleFormat::U8 | SampleFormat::U16 { .. })
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} upscales frames, not masks");

    StageError::msg(message)
}

fn unusable_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} upscales grey or yuv frames, not {layout} frames");

    StageError::msg(message)
}

fn unusable_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} upscales U8 or U16 frames, not {sample} frames");

    StageError::msg(message)
}

fn model_missing(missing: MissingModel) -> StageError {
    let message = format!("{STAGE} cannot find its model. {missing}");

    StageError::msg(message)
}

fn no_f16_arithmetic() -> StageError {
    let message = format!(
        "{STAGE} was given precision f16, but this device has no f16 arithmetic. Use precision = \"f32\"",
    );

    StageError::msg(message)
}

fn unreadable_model(file: &str, error: io::Error) -> StageError {
    let message = format!("{STAGE} could not read its model {file}: {error}");

    StageError::msg(message)
}

fn broken_model(file: &str, error: WeightsError) -> StageError {
    let message = format!("{STAGE} could not load its model {file}: {error}");

    StageError::msg(message)
}

fn path_name(path: BodyPath) -> &'static str {
    match path {
        BodyPath::Convolution => "convolution",
        BodyPath::Im2col => "im2col",
    }
}

fn failed_warm_up(path: BodyPath, error: ArtcnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{STAGE} could not warm up on the {name} path: {error}");

    StageError::msg(message)
}

fn failed_run(path: BodyPath, error: ArtcnnError) -> StageError {
    let name = path_name(path);
    let message = format!("{STAGE} could not run on the {name} path: {error}");

    StageError::msg(message)
}

fn missing_frame() -> StageError {
    let message = format!("{STAGE} expected a frame in slot 0");

    StageError::msg(message)
}

fn missing_channel(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

/// Finds `model`'s weight file in `store`, or the copy built into the binary.
pub(crate) fn locate(store: &Store, model: Model) -> Result<Located, StageError> {
    let found = store.find("artcnn", model.file_name());

    match found {
        Ok(located) => Ok(located),
        Err(missing) => Err(model_missing(missing)),
    }
}

/// The input format, once it is known to be U8 or U16 frames in a grey or YUV layout.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Err(unusable_sample(format.sample));
    }

    Ok(*format)
}

/// The samples accepted from `offered`, refusing masks and layouts the stage cannot upscale.
fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

pub(crate) fn doubled(input: &FrameFormat) -> FrameFormat {
    FrameFormat {
        width: input.width * 2,
        height: input.height * 2,
        ..*input
    }
}

/// The feature maps' precision, refusing f16 on a device without f16 arithmetic.
pub(crate) fn checked_precision(
    capabilities: &Capabilities,
    precision: PrecisionName,
) -> Result<Precision, StageError> {
    let computes_f16 = capabilities.can_compute(ElementType::F16);

    if precision == PrecisionName::F16 && !computes_f16 {
        return Err(no_f16_arithmetic());
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

/// Reads, parses and uploads `model`'s weights in `precision`'s type.
fn load_network<R: Runtime>(
    client: &ComputeClient<R>,
    store: &Store,
    model: Model,
    precision: Precision,
) -> Result<Network, StageError> {
    let file = model.file_name();
    let located = locate(store, model)?;
    let bytes = match located.read() {
        Ok(bytes) => bytes,
        Err(error) => return Err(unreadable_model(file, error)),
    };

    let parsed = Weights::parse(&bytes, model.features(), model.blocks());
    let weights = match parsed {
        Ok(weights) => weights,
        Err(error) => return Err(broken_model(file, error)),
    };

    let uploaded = Network::upload(client, &weights, precision);
    match uploaded {
        Ok(network) => Ok(network),
        Err(error) => Err(broken_model(file, error)),
    }
}

/// The largest tile the scratch budget and the device's largest buffer allow on `path`.
fn tile_for(network: &Network, path: BodyPath, cap: usize) -> u32 {
    let features = network.features();
    let halo = network.halo();
    let precision = network.precision();

    tile_side(features, halo, precision, path, TILE_BUDGET, cap)
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
        let conv0 = context.scratch(sizes.map_bytes);
        let first = context.scratch(sizes.map_bytes);
        let second = context.scratch(sizes.map_bytes);
        let third = context.scratch(sizes.map_bytes);
        let columns = match path {
            BodyPath::Convolution => None,
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

/// Warms the network up on `path` at the frame's size, giving the tile side it used.
fn warm_up_on<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    network: &Network,
    path: BodyPath,
    cap: usize,
    width: u32,
    height: u32,
) -> Result<u32, ArtcnnError> {
    let tile = tile_for(network, path, cap);
    let buffers = MapBuffers::new(context, network, path, tile, width, height);
    let maps = buffers.maps();
    let sizing = Sizing(context);

    warm_up(client, &sizing, network, path, width, height, tile, maps)?;

    Ok(tile)
}

/// Warms the network up on `path`, giving the path that ran and its tile side.
///
/// When the convolution path cannot launch on this device, it falls back to im2col.
fn warmed_path<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    network: &Network,
    path: BodyPath,
    cap: usize,
    input: &FrameFormat,
) -> Result<(BodyPath, u32), StageError> {
    let warmed = warm_up_on(client, context, network, path, cap, input.width, input.height);

    let reason = match warmed {
        Ok(tile) => return Ok((path, tile)),
        Err(ArtcnnError::Convolution(reason)) if path == BodyPath::Convolution => reason,
        Err(error) => return Err(failed_warm_up(path, error)),
    };

    tracing::warn!(
        stage = STAGE,
        %reason,
        "the convolution path could not launch, so the stage falls back to im2col",
    );

    let fallback = BodyPath::Im2col;
    let rewarmed = warm_up_on(client, context, network, fallback, cap, input.width, input.height);
    match rewarmed {
        Ok(tile) => Ok((fallback, tile)),
        Err(error) => Err(failed_warm_up(fallback, error)),
    }
}

/// The tables that double a YUV input's chroma, keeping its siting. Grey has none.
///
/// 4:4:4 chroma is sited like luma, so it takes the luma plane's shape.
fn chroma_tables<R: Runtime>(client: &ComputeClient<R>, input: &FrameFormat) -> Option<Tables> {
    let FrameLayout::Yuv { .. } = input.layout else {
        return None;
    };

    let window = Window {
        width: input.width * 2,
        height: input.height * 2,
        src_left: 0.0,
        src_top: 0.0,
        src_width: None,
        src_height: None,
    };
    let shapes = geometry(input, &window);
    let shape = shapes.chroma.unwrap_or(shapes.luma);
    let tables = Tables::new(client, &shape, &CATROM);

    Some(tables)
}

/// Doubles a clip's size, running its luma through the ArtCNN network and its chroma through Catrom.
pub(crate) struct Artcnn;

impl StageDefinition for Artcnn {
    type Args = ArtcnnArgs;

    fn package() -> &'static str {
        "upscale"
    }

    fn identifier() -> &'static str {
        "artcnn"
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

impl FilterDefinition for Artcnn {
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
        checked_precision(capabilities, args.precision)?;

        let store = Store::from_environment();
        let model = args.model.model();
        locate(&store, model)?;

        let output = doubled(&input);
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
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let capabilities = context.capabilities();
        let precision = checked_precision(capabilities, args.precision)?;

        let client = context.client().clone();
        let store = Store::from_environment();
        let model = args.model.model();
        let network = load_network(&client, &store, model, precision)?;

        let frame_context = context.frame_context();
        let chosen = body_path(capabilities, precision);
        let cap = allocation_cap(capabilities);
        let (path, tile) = warmed_path(&client, &frame_context, &network, chosen, cap, &input)?;

        let chroma = chroma_tables(&client, &input);
        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let samples = samples_for(input.sample);

        let upscaler = Upscaler {
            client,
            network,
            path,
            tile,
            output: doubled(&input),
            chroma,
            element,
            peak,
            samples,
        };

        Ok(Box::new(upscaler))
    }
}

/// Runs `upscale/artcnn` on every frame, with the network, tables and tile chosen at build.
struct Upscaler<R: Runtime> {
    client: ComputeClient<R>,
    network: Network,
    path: BodyPath,
    tile: u32,
    output: FrameFormat,
    chroma: Option<Tables>,
    element: Element,
    peak: f32,
    samples: Samples,
}

impl<R: Runtime> Upscaler<R> {
    /// Widens `input` to f32, runs the network over it, and narrows the doubled result into `output`.
    fn upscale_luma(&self, context: &FrameContext, input: &Plane, output: &Plane) -> Result<(), StageError> {
        let (width, height) = input.dimensions();
        let (doubled_width, doubled_height) = output.dimensions();
        let sizing = Sizing(context);

        let input_values = Scratch::new::<f32>(context, width, height);
        let output_values = Scratch::new::<f32>(context, doubled_width, doubled_height);
        let buffers = MapBuffers::new(context, &self.network, self.path, self.tile, width, height);
        let input_plane = input_values.plane();
        let output_plane = output_values.plane();
        let maps = buffers.maps();

        let source_handle = input.buffer().handle();
        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, doubled_width, doubled_height) };

        widen(&self.client, &sizing, source_plane, input_plane, self.element);

        let ran = artcnn(
            &self.client,
            &sizing,
            &self.network,
            self.path,
            input_plane,
            self.peak,
            self.tile,
            maps,
            output_plane,
        );
        if let Err(error) = ran {
            return Err(failed_run(self.path, error));
        }

        narrow(
            &self.client,
            &sizing,
            output_plane,
            target_plane,
            self.element,
            self.peak,
        );

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Upscaler<R> {}

impl<R: Runtime> Filter for Upscaler<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);

        let Some(input_luma) = input.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        let Some(output_luma) = output.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        self.upscale_luma(context, input_luma, output_luma)?;

        let Some(tables) = &self.chroma else {
            return Ok(output);
        };

        for channel in [Channel::U, Channel::V] {
            let Some(input_plane) = input.plane(channel) else {
                return Err(missing_channel(channel));
            };

            let Some(output_plane) = output.plane(channel) else {
                return Err(missing_channel(channel));
            };

            resize_plane(
                &self.client,
                context,
                input_plane,
                output_plane,
                tables,
                self.samples,
            );
        }

        Ok(output)
    }
}
