use std::collections::BTreeSet;

use cubecl::features::MmaConfig;
use cubecl::ir::{AddressType, StorageType};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::components::{
    ConvSetupError,
    ConvolutionOperation,
    ConvolutionProblem,
    Dimensionality,
};
use cubek_convolution::definition::{ConvBlueprint, ForwardBlueprint};
use cubek_convolution::{AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy};
use cubek_matmul::components::stage::PartitionBuffering;
use cubek_matmul::components::tile::TileMatmulKind;
use cubek_matmul::definition::{
    MatmulElems,
    MatmulGlobalElems,
    MatmulSetupError,
    TilingBlueprint,
    TilingScheme,
};
use cubek_matmul::launch::Strategy as MatmulStrategy;
use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
use cubek_std::{InputBinding, MatrixLayout};

use super::fused::launch_fused;
use super::kernels::Candidate;
use super::network::map_bytes;
use crate::Dispatch;
use crate::artcnn::{Network, Precision, Region};

/// How the body's `F -> F` convolutions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPath {
    /// `cubek-convolution` on cooperative-matrix tiles, applying each layer's SiLU and residual as it writes.
    Convolution,
    /// `cubek-convolution` on cooperative-matrix tiles, with each layer's SiLU and residual in a separate
    /// pass.
    UnfusedConvolution,
    /// An im2col copy of the maps, then a `cubek-matmul` unit matmul.
    Im2col,
}

/// Why a forward pass could not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArtcnnError {
    #[error("the convolution path could not launch: {0}")]
    Convolution(String),

    #[error("the im2col path's matmul could not launch: {0}")]
    Matmul(String),

    #[error("the im2col path needs a columns buffer, but none was given")]
    MissingColumns,

    #[error("the device failed while running the network: {0}")]
    Device(String),
}

impl From<ServerError> for ArtcnnError {
    fn from(error: ServerError) -> Self {
        Self::Device(error.to_string())
    }
}

/// What the epilogue applies after adding the bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    None,
    Silu,
}

#[cube(launch_unchecked)]
fn epilogue_kernel<F: Float>(
    values: &mut Array<F>,
    bias: &Array<F>,
    residual: &Array<F>,
    features: u32,
    add_bias: u32,
    silu: u32,
    add_residual: u32,
) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let mut value = f32::cast_from(values[ABSOLUTE_POS]);

    if add_bias == 1 {
        let channel = ABSOLUTE_POS as u32 % features;
        value += f32::cast_from(bias[channel as usize]);
    }

    if silu == 1 {
        value = value / (1.0 + f32::exp(-value));
    }

    if add_residual == 1 {
        value += f32::cast_from(residual[ABSOLUTE_POS]);
    }

    values[ABSOLUTE_POS] = F::cast_from(value);
}

/// Adds `layer`'s bias to the NHWC maps in `values`, then applies `activation`, then adds `residual` if
/// given.
///
/// The maps hold `samples` samples of the network's features, and are updated in place. The layer must map
/// the features to themselves.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn epilogue<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    values: &Handle,
    layer: usize,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
) {
    run_epilogue(
        client, dispatch, network, values, layer, samples, activation, residual, true,
    );
}

/// Adds `layer`'s bias to the NHWC maps in `values` if `add_bias` is set, then applies `activation`, then
/// adds `residual` if given.
///
/// The maps hold `samples` samples of the network's features, and are updated in place. The layer must map
/// the features to themselves.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub(crate) fn run_epilogue<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    values: &Handle,
    layer: usize,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
    add_bias: bool,
) {
    let buffers = network.layer(layer);
    let features = network.features();
    assert!(
        buffers.in_channels == features && buffers.out_channels == features,
        "layer {layer} does not map the features to themselves"
    );

    let units = (samples * features) as usize;
    let bytes = (units * network.precision().bytes()) as u64;
    let residual_handle = residual.unwrap_or(values);
    assert!(
        values.size_in_used() >= bytes,
        "the values are smaller than the samples' maps"
    );
    assert!(
        residual_handle.size_in_used() >= bytes,
        "the residual is smaller than the values"
    );

    let bias_flag = u32::from(add_bias);
    let silu = u32::from(activation == Activation::Silu);
    let add_residual = u32::from(residual.is_some());
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the values and the residual hold `samples * F` values and the bias `F`. Without a residual, the
    // values are bound in its place and never read through it.
    unsafe {
        let values_array = ArrayArg::from_raw_parts(values.clone(), units);
        let bias = ArrayArg::from_raw_parts(buffers.bias.clone(), features as usize);
        let residual_array = ArrayArg::from_raw_parts(residual_handle.clone(), units);

        match network.precision() {
            Precision::F16 => epilogue_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                values_array,
                bias,
                residual_array,
                features,
                bias_flag,
                silu,
                add_residual,
            ),
            Precision::F32 => epilogue_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                values_array,
                bias,
                residual_array,
                features,
                bias_flag,
                silu,
                add_residual,
            ),
        }
    }
}

#[cube(launch_unchecked)]
fn im2col_kernel<F: Float>(input: &Array<F>, columns: &mut Array<F>, height: u32, width: u32, channels: u32) {
    if ABSOLUTE_POS >= columns.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row_length = channels * 9;
    let pixel = position / row_length;
    let within_row = position % row_length;
    let tap = within_row / channels;
    let channel = within_row % channels;
    let source_column = (pixel % width) as i32 + (tap % 3) as i32 - 1;
    let source_row = (pixel / width) as i32 + (tap / 3) as i32 - 1;
    let inside =
        source_column >= 0 && source_row >= 0 && source_column < width as i32 && source_row < height as i32;
    let mut value = F::new(0.0f32);

    if inside {
        let index = (source_row as u32 * width + source_column as u32) * channels + channel;
        value = input[index as usize];
    }

    columns[ABSOLUTE_POS] = value;
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights, into `output`. Gives
/// whether the forced blueprint ran.
///
/// The convolution paths add the bias inside the convolution. The
/// [BodyPath::Im2col](crate::artcnn::BodyPath::Im2col) path leaves the bias out, and stages the maps in
/// `columns`, which holds nine times as many values as a map.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub(crate) fn convolve<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    layer: usize,
    input: &Handle,
    output: &Handle,
    columns: Option<&Handle>,
    region: Region,
) -> Result<bool, ArtcnnError> {
    check_layer_maps(network, layer, input, output, region);

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => {
            let operands = body_operands(network, layer, true);

            convolve_on_tiles(client, &operands, input, output, region, true)
        },
        BodyPath::Im2col => {
            let columns = columns.ok_or(ArtcnnError::MissingColumns)?;
            let bytes = map_bytes(network, region);
            assert!(
                columns.size_in_used() >= bytes * 9,
                "the columns are smaller than nine of the region's maps"
            );

            match network.precision() {
                Precision::F16 => convolve_by_columns::<R, half::f16>(
                    client, dispatch, network, layer, input, output, columns, region,
                )?,
                Precision::F32 => convolve_by_columns::<R, f32>(
                    client, dispatch, network, layer, input, output, columns, region,
                )?,
            }

            Ok(false)
        },
    }
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights and bias on
/// cooperative-matrix tiles, into `output`. Gives whether the forced blueprint ran.
///
/// With `allow_forced` the convolution picks its blueprint as a forward pass does, and otherwise it keeps
/// the inferred blueprint.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn convolve_with_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> Result<bool, ArtcnnError> {
    if allow_forced {
        let path = BodyPath::Convolution;

        return convolve(
            client, dispatch, network, path, layer, input, output, None, region,
        );
    }

    check_layer_maps(network, layer, input, output, region);

    let operands = body_operands(network, layer, true);

    convolve_on_tiles(client, &operands, input, output, region, false)
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights on cooperative-matrix
/// tiles, into `output` without the bias. Gives whether the forced blueprint ran.
#[cfg(any(test, feature = "test-support"))]
pub fn convolve_without_bias<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
) -> Result<bool, ArtcnnError> {
    check_layer_maps(network, layer, input, output, region);

    let operands = body_operands(network, layer, false);

    convolve_on_tiles(client, &operands, input, output, region, true)
}

/// Asserts that `layer` maps the features to themselves, and that `input` and `output` each hold the
/// region's maps.
fn check_layer_maps(network: &Network, layer: usize, input: &Handle, output: &Handle, region: Region) {
    let buffers = network.layer(layer);
    let features = network.features();
    assert!(
        buffers.in_channels == features && buffers.out_channels == features,
        "layer {layer} does not map the features to themselves"
    );

    let bytes = map_bytes(network, region);
    assert!(
        input.size_in_used() >= bytes,
        "the input is smaller than the region's maps"
    );
    assert!(
        output.size_in_used() >= bytes,
        "the output is smaller than the region's maps"
    );
}

/// The fewest samples a region needs for the forced blueprint to beat the inferred one.
const FORCED_SAMPLES: usize = 8192;

/// The one plane width the forced blueprint is known to give correct results on.
const FORCED_PLANE_SIZE: u32 = 64;

/// The side of the cooperative-matrix tiles the forced blueprint multiplies.
const FORCED_TILE: u32 = 16;

/// The NHWC input, OHWI weight and NHWC output shapes of a 3x3 convolution over `region`, with their
/// contiguous strides.
struct ConvolutionShapes {
    map_shape: [usize; 4],
    map_strides: Vec<usize>,
    weight_shape: [usize; 4],
    weight_strides: Vec<usize>,
    out_shape: [usize; 4],
    out_strides: Vec<usize>,
}

fn convolution_shapes(region: Region, in_channels: usize, out_channels: usize) -> ConvolutionShapes {
    let height = region.height as usize;
    let width = region.width as usize;
    let map_shape = [1, height, width, in_channels];
    let weight_shape = [out_channels, 3, 3, in_channels];
    let out_shape = [1, height, width, out_channels];
    let map_strides = contiguous(&map_shape);
    let weight_strides = contiguous(&weight_shape);
    let out_strides = contiguous(&out_shape);

    ConvolutionShapes {
        map_shape,
        map_strides,
        weight_shape,
        weight_strides,
        out_shape,
        out_strides,
    }
}

/// The problem `cubek-convolution` solves for a 3x3 convolution of `region`'s NHWC maps, with one sample
/// of zero padding, from `in_channels` to `out_channels`.
pub(crate) fn convolution_problem(
    region: Region,
    in_channels: usize,
    out_channels: usize,
    globals: MatmulGlobalElems,
    address_type: AddressType,
) -> ConvolutionProblem {
    let height = region.height as usize;
    let width = region.width as usize;
    let shapes = convolution_shapes(region, in_channels, out_channels);

    ConvolutionProblem {
        m: height * width,
        n: out_channels,
        k: 9 * in_channels,
        lhs_strides: Strides::from(shapes.map_strides),
        rhs_strides: Strides::from(shapes.weight_strides),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
        kernel_size: vec![3, 3],
        stride: vec![1, 1],
        padding: vec![1, 1],
        dilation: vec![1, 1],
        batches: 1,
        channels: in_channels,
        out_channels,
        in_shape: Shape::from(vec![height, width]),
        out_shape: Shape::from(vec![height, width]),
        padded_channels: in_channels,
        operation: ConvolutionOperation::Forward,
        dimensionality: Dimensionality::Dim2,
        global_dtypes: globals,
        address_type,
    }
}

/// A blueprint of 16x16x16 cooperative-matrix tiles for `problem`, or `None` to let `cubek-convolution`
/// infer one.
///
/// Each workgroup computes a block of 256 samples by `n` channels in four planes. The blueprint
/// is only given where it is faster and known to be exact. That needs f16 operands, a device offering a
/// 16x16x16 f16 cooperative matrix accumulating in f32 (`offers_cmma_16`), 64 lanes per plane, `n` a
/// multiple of 16 and at least 8192 samples.
pub(crate) fn forced_blueprint(
    problem: &ConvolutionProblem,
    plane_size_max: u32,
    offers_cmma_16: bool,
) -> Option<ConvBlueprint> {
    let half_type = storage_type(Precision::F16);
    let operands_are_half = problem.global_dtypes.lhs == half_type && problem.global_dtypes.rhs == half_type;
    let tiled_channels = problem.n.is_multiple_of(FORCED_TILE as usize);
    let large = problem.m >= FORCED_SAMPLES;

    if !operands_are_half || !offers_cmma_16 || plane_size_max != FORCED_PLANE_SIZE {
        return None;
    }

    if !tiled_channels || !large {
        return None;
    }

    let channel_tiles = problem.n as u32 / FORCED_TILE;
    let scheme = TilingScheme::builder()
        .with_tile_size((FORCED_TILE, FORCED_TILE, FORCED_TILE).into())
        .with_partition_size((4, channel_tiles, 2).into())
        .with_stage_size((4, 1, 1).into())
        .build()
        .expect("every size of the scheme is set");
    let hypercube = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(4))
        .cube_count_strategy(CubeCountStrategy::FromProblem)
        .build();

    let matmul_problem = problem.as_matmul_problem();
    let matmul = TilingBlueprint::builder(TileMatmulKind::Cmma, scheme, plane_size_max, &matmul_problem)
        .partition_buffering(PartitionBuffering::Single)
        .hypercube_blueprint(hypercube)
        .build();
    let forward = ForwardBlueprint {
        matmul,
        dimensionality: Dimensionality::Dim2,
        has_bias: true,
    };

    Some(ConvBlueprint::Forward(forward))
}

/// Whether the device offers a 16x16x16 f16 cooperative matrix accumulating in f32.
pub fn offers_cmma_16<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let configs = &client.properties().features.matmul.cmma;

    offers_cmma_16_in(configs)
}

/// Whether `configs` holds a 16x16x16 f16 cooperative matrix accumulating in f32.
pub(crate) fn offers_cmma_16_in(configs: &BTreeSet<MmaConfig>) -> bool {
    let half_type = storage_type(Precision::F16);
    let float_type = storage_type(Precision::F32);
    let config = MmaConfig {
        a_type: half_type,
        b_type: half_type,
        cd_type: float_type,
        m: FORCED_TILE,
        n: FORCED_TILE,
        k: FORCED_TILE,
    };

    configs.contains(&config)
}

/// The blueprint a convolution solving `problem` takes on the client's device, or `None` to infer one.
fn chosen_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &ConvolutionProblem,
) -> Option<ConvBlueprint> {
    let plane_size_max = client.properties().hardware.plane_size_max;
    let offers_cmma = offers_cmma_16(client);

    forced_blueprint(problem, plane_size_max, offers_cmma)
}

/// Whether a body convolution of `region` takes the forced blueprint on the client's device.
#[cfg(any(test, feature = "test-support"))]
pub fn takes_forced_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    region: Region,
) -> bool {
    let features = network.features() as usize;
    let dtype = storage_type(network.precision());
    let globals = uniform_globals(dtype);
    let map_values = region.samples() as usize * features;
    let address_type = AddressType::from_len(map_values);
    let problem = convolution_problem(region, features, features, globals, address_type);
    let blueprint = chosen_blueprint(client, &problem);

    blueprint.is_some()
}

pub(crate) fn storage_type(precision: Precision) -> StorageType {
    match precision {
        Precision::F16 => half::f16::as_type_native_unchecked().storage_type(),
        Precision::F32 => f32::as_type_native_unchecked().storage_type(),
    }
}

/// Global types with `dtype` for both operands and the output.
fn uniform_globals(dtype: StorageType) -> MatmulGlobalElems {
    MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    }
}

/// Binds `handle` as a tensor of `shape` laid out at `strides`.
///
/// # Safety
///
/// The handle must hold every element the shape reaches at these strides.
unsafe fn tensor<R: Runtime>(handle: &Handle, shape: &[usize], strides: &[usize]) -> TensorBinding<R> {
    let shape_value = Shape::from(shape.to_vec());
    let strides_value = Strides::from(strides.to_vec());

    // Safety: the caller guarantees the handle holds the shape's elements at these strides.
    unsafe { TensorBinding::from_raw_parts(handle.clone(), strides_value, shape_value) }
}

fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }

    strides
}

/// Describes a convolution setup failure with `Debug`, since `ConvSetupError` has no `Display`.
pub(crate) fn convolution_failed(error: ConvSetupError) -> ArtcnnError {
    ArtcnnError::Convolution(format!("{error:?}"))
}

fn matmul_failed(error: MatmulSetupError) -> ArtcnnError {
    ArtcnnError::Matmul(error.to_string())
}

/// The weights and bias of one 3x3 convolution on the device, with its global types.
///
/// The bias is in the output type, which is how cubek-convolution reads it.
pub(crate) struct Operands<'a> {
    pub(crate) weights: &'a Handle,
    pub(crate) bias: Option<&'a Handle>,
    pub(crate) in_channels: usize,
    pub(crate) out_channels: usize,
    pub(crate) globals: MatmulGlobalElems,
}

impl<'a> Operands<'a> {
    /// The operands of an `F -> F` body convolution with `weights` and `bias` in `precision`'s type.
    pub(crate) fn body(weights: &'a Handle, bias: &'a Handle, features: usize, precision: Precision) -> Self {
        let dtype = storage_type(precision);
        let globals = uniform_globals(dtype);

        Operands {
            weights,
            bias: Some(bias),
            in_channels: features,
            out_channels: features,
            globals,
        }
    }
}

/// The operands of `layer`, a body convolution, with its bias if `with_bias` is set.
pub(crate) fn body_operands(network: &Network, layer: usize, with_bias: bool) -> Operands<'_> {
    let buffers = network.layer(layer);
    let features = network.features() as usize;
    let precision = network.precision();
    let mut operands = Operands::body(&buffers.weights, &buffers.bias, features, precision);

    if !with_bias {
        operands.bias = None;
    }

    operands
}

/// One 3x3 convolution's tensors bound for `cubek-convolution`, with its problem and the blueprint it takes.
pub(crate) struct BoundConvolution<R: Runtime> {
    pub(crate) problem: ConvolutionProblem,
    pub(crate) blueprint: Option<ConvBlueprint>,
    pub(crate) input: InputBinding<R>,
    pub(crate) weight: InputBinding<R>,
    pub(crate) bias: Option<InputBinding<R>>,
    pub(crate) output: TensorBinding<R>,
}

/// Binds a convolution of the NHWC maps of `region` in `input` with `operands`, into NHWC maps of the output
/// channels in `output`.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
fn bind_convolution<R: Runtime>(
    client: &ComputeClient<R>,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> BoundConvolution<R> {
    check_operand_sizes(operands, input, output, region);

    let globals = &operands.globals;
    let in_channels = operands.in_channels;
    let out_channels = operands.out_channels;

    let ConvolutionShapes {
        map_shape,
        map_strides,
        weight_shape,
        weight_strides,
        out_shape,
        out_strides,
    } = convolution_shapes(region, in_channels, out_channels);
    let bias_shape = [out_channels];
    let bias_strides = [1];
    // Safety: the input holds the region's `1 x H x W x in` maps contiguously, as `check_operand_sizes`
    // asserts.
    let input_tensor = unsafe { tensor::<R>(input, &map_shape, &map_strides) };
    // Safety: the weights hold `out x 3 x 3 x in` values contiguously, as `check_operand_sizes` asserts.
    let weight_tensor = unsafe { tensor::<R>(operands.weights, &weight_shape, &weight_strides) };
    // Safety: the output holds the region's `1 x H x W x out` maps contiguously, as `check_operand_sizes`
    // asserts.
    let output_tensor = unsafe { tensor::<R>(output, &out_shape, &out_strides) };
    // Safety: the bias holds `out` values contiguously, as `check_operand_sizes` asserts.
    let bias_tensor = operands
        .bias
        .map(|bias| unsafe { tensor::<R>(bias, &bias_shape, &bias_strides) });

    let input_binding = InputBinding::new(input_tensor, globals.lhs);
    let weight_binding = InputBinding::new(weight_tensor, globals.rhs);
    let bias_binding = bias_tensor.map(|tensor| InputBinding::new(tensor, globals.out));

    let address_type = required_address_type(
        &input_binding,
        &weight_binding,
        bias_binding.as_ref(),
        &output_tensor,
        globals.out,
    );
    let problem = convolution_problem(region, in_channels, out_channels, globals.clone(), address_type);
    let blueprint = if allow_forced {
        chosen_blueprint(client, &problem)
    } else {
        None
    };

    BoundConvolution {
        problem,
        blueprint,
        input: input_binding,
        weight: weight_binding,
        bias: bias_binding,
        output: output_tensor,
    }
}

/// The address type `cubek-convolution`'s forward launch picks for these buffers, wide enough to index the
/// largest of them.
fn required_address_type<R: Runtime>(
    input: &InputBinding<R>,
    weight: &InputBinding<R>,
    bias: Option<&InputBinding<R>>,
    output: &TensorBinding<R>,
    out_type: StorageType,
) -> AddressType {
    let input_address = input.required_address_type();
    let weight_address = weight.required_address_type();
    let bias_address = bias.map(InputBinding::required_address_type).unwrap_or_default();
    let output_address = output.required_address_type(out_type.size());

    input_address
        .max(weight_address)
        .max(bias_address)
        .max(output_address)
}

/// Convolves the NHWC maps of `region` in `input` with `operands` on cooperative-matrix tiles, into NHWC maps
/// of the output channels in `output`. Gives whether the forced blueprint ran.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
pub(crate) fn convolve_on_tiles<R: Runtime>(
    client: &ComputeClient<R>,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> Result<bool, ArtcnnError> {
    let bound = bind_convolution::<R>(client, operands, input, output, region, allow_forced);
    let forced = bound.blueprint.is_some();

    let inputs = ConvolutionInputs::Forward {
        input: bound.input,
        weight: bound.weight,
        bias: bound.bias,
        out: bound.output,
    };
    let arguments = ConvolutionArgs::<2> {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let strategy = convolution_strategy(bound.blueprint.as_ref());
    let dtypes = MatmulElems::from_globals(&operands.globals);
    let launched = cubek_convolution::launch_ref::<R, 2>(&strategy, client, inputs, arguments, dtypes);

    launched.map_err(convolution_failed)?;

    Ok(forced)
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights and bias on
/// cooperative-matrix tiles into `output`, applying `finish` as each value is written.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
fn convolve_fused<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<LayerRun, ArtcnnError> {
    check_layer_maps(network, layer, input, output, region);

    let operands = body_operands(network, layer, true);
    let forced = launch_fused_convolution(client, &operands, input, output, region, finish, allow_forced)?;

    Ok(LayerRun {
        forced,
        fused: true,
        candidate: None,
        winograd_offered: false,
    })
}

/// Convolves the NHWC maps of `region` in `input` with `operands` on cooperative-matrix tiles into `output`,
/// applying `finish` as each value is written. Gives whether the forced blueprint ran.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
pub(crate) fn launch_fused_convolution<R: Runtime>(
    client: &ComputeClient<R>,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<bool, ArtcnnError> {
    let bound = bind_convolution::<R>(client, operands, input, output, region, allow_forced);
    let forced = bound.blueprint.is_some();

    let out_channels = operands.out_channels;
    let shapes = convolution_shapes(region, operands.in_channels, out_channels);
    let samples = region.samples() as usize;
    let bytes = samples * out_channels * operands.globals.out.size();
    let residual = finish.residual.map(|handle| {
        assert!(
            handle.size_in_used() >= bytes as u64,
            "the residual is smaller than the region's maps"
        );

        // Safety: the residual holds the region's `1 x H x W x out` maps contiguously, as just asserted.
        unsafe { tensor::<R>(handle, &shapes.out_shape, &shapes.out_strides) }
    });

    launch_fused(client, bound, residual, finish.activation)?;

    Ok(forced)
}

/// How a body layer's convolution ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRun {
    /// Whether the convolution took the forced blueprint.
    pub forced: bool,
    /// Whether the layer's SiLU and residual were applied as the convolution wrote, rather than in a
    /// separate pass.
    pub fused: bool,
    /// The candidate that ran the convolution, if one was chosen for it.
    pub candidate: Option<Candidate>,
    /// Whether a Winograd variant was among the candidates the convolution's kernel was chosen from.
    pub winograd_offered: bool,
}

impl LayerRun {
    /// How `candidate` ran, as though it were the only candidate offered. `forced` is whether it took the
    /// forced blueprint.
    pub(crate) fn of_candidate(candidate: Candidate, forced: bool) -> Self {
        Self {
            forced,
            fused: true,
            candidate: Some(candidate),
            winograd_offered: matches!(candidate, Candidate::Winograd(_)),
        }
    }
}

/// What a body layer applies after its convolution and bias.
#[derive(Debug, Clone, Copy)]
pub struct Finish<'a> {
    pub activation: Activation,
    /// The maps added last, laid out as the layer's output.
    pub residual: Option<&'a Handle>,
}

/// Runs body layer `layer` over the NHWC maps of `region` in `input` on `path`, into `output`. That is its
/// convolution and bias, then `finish`.
///
/// [BodyPath::Im2col](crate::artcnn::BodyPath::Im2col) stages the maps in `columns`, which holds nine times
/// as many values as a map.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub(crate) fn body_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    layer: usize,
    input: &Handle,
    output: &Handle,
    columns: Option<&Handle>,
    region: Region,
    finish: Finish<'_>,
) -> Result<LayerRun, ArtcnnError> {
    if path == BodyPath::Convolution {
        return convolve_fused(client, network, layer, input, output, region, finish, true);
    }

    let forced = convolve(
        client, dispatch, network, path, layer, input, output, columns, region,
    )?;

    // The unfused convolution adds the bias inside the convolution, and im2col adds it here.
    let add_bias = path == BodyPath::Im2col;
    finish_separately(client, dispatch, network, layer, output, region, finish, add_bias);

    Ok(LayerRun {
        forced,
        fused: false,
        candidate: None,
        winograd_offered: false,
    })
}

/// Runs body layer `layer` on `path` as a forward pass does, keeping the inferred blueprint unless
/// `allow_forced` is set.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub fn body_layer_with_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    path: BodyPath,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<LayerRun, ArtcnnError> {
    assert!(
        path != BodyPath::Im2col,
        "only the convolution paths choose a blueprint"
    );

    if allow_forced {
        return body_layer(
            client, dispatch, network, path, layer, input, output, None, region, finish,
        );
    }

    if path == BodyPath::Convolution {
        return convolve_fused(client, network, layer, input, output, region, finish, false);
    }

    convolve_with_blueprint(client, dispatch, network, layer, input, output, region, false)?;
    finish_separately(client, dispatch, network, layer, output, region, finish, false);

    Ok(LayerRun {
        forced: false,
        fused: false,
        candidate: None,
        winograd_offered: false,
    })
}

/// Applies `finish` to `layer`'s maps of `region` in `output` in a separate pass, adding the bias first if
/// `add_bias` is set.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the pass"
)]
fn finish_separately<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    layer: usize,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    add_bias: bool,
) {
    let samples = region.samples();

    run_epilogue(
        client,
        dispatch,
        network,
        output,
        layer,
        samples,
        finish.activation,
        finish.residual,
        add_bias,
    );
}

/// Asserts that `input`, `output` and the operands each hold every value a convolution of `region` reads or
/// writes through them.
fn check_operand_sizes(operands: &Operands<'_>, input: &Handle, output: &Handle, region: Region) {
    let globals = &operands.globals;
    let samples = region.samples() as usize;
    let in_channels = operands.in_channels;
    let out_channels = operands.out_channels;
    let input_bytes = samples * in_channels * globals.lhs.size();
    let weight_bytes = out_channels * 9 * in_channels * globals.rhs.size();
    let output_bytes = samples * out_channels * globals.out.size();
    let bias_bytes = out_channels * globals.out.size();
    assert!(
        input.size_in_used() >= input_bytes as u64,
        "the input is smaller than the region's maps"
    );
    assert!(
        operands.weights.size_in_used() >= weight_bytes as u64,
        "the weights are smaller than the convolution's"
    );
    assert!(
        output.size_in_used() >= output_bytes as u64,
        "the output is smaller than the region's maps"
    );

    if let Some(bias) = operands.bias {
        assert!(
            bias.size_in_used() >= bias_bytes as u64,
            "the bias is smaller than the output channels"
        );
    }
}

fn convolution_strategy(blueprint: Option<&ConvBlueprint>) -> Strategy {
    let algorithm = ConvAlgorithm::SimpleSyncCyclic;

    match blueprint {
        Some(forced) => Strategy::Forced {
            algorithm,
            blueprint: forced.clone(),
        },
        None => Strategy::Inferred {
            algorithm,
            tile_kind: AcceleratedTileKind::Cmma,
        },
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
fn convolve_by_columns<R: Runtime, F: Float>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    columns: &Handle,
    region: Region,
) -> Result<(), ArtcnnError> {
    let buffers = network.layer(layer);
    let features = network.features() as usize;
    let samples = region.samples() as usize;
    let depth = 9 * features;
    let channels = network.features();
    let column_values = samples * depth;
    assert!(
        u32::try_from(column_values).is_ok(),
        "the columns hold more values than the kernel can index"
    );

    let dtype = F::as_type_native_unchecked().storage_type();
    let cubes = dispatch.cube_count(column_values);
    let unit = dispatch.cube_dim();

    // Safety: the input holds the region's samples times `F` values, and the columns nine times that.
    unsafe {
        let input_array = ArrayArg::from_raw_parts(input.clone(), samples * features);
        let columns_array = ArrayArg::from_raw_parts(columns.clone(), column_values);
        im2col_kernel::launch_unchecked::<F, R>(
            client,
            cubes,
            unit,
            input_array,
            columns_array,
            region.height,
            region.width,
            channels,
        );
    }

    let lhs_shape = [samples, depth];
    let lhs_strides = contiguous(&lhs_shape);
    let rhs_shape = [depth, features];
    let rhs_strides = [1, depth];
    let out_shape = [samples, features];
    let out_strides = contiguous(&out_shape);
    // Safety: the columns hold `samples x 9F` values contiguously, as `convolve` checks.
    let lhs_tensor = unsafe { tensor::<R>(columns, &lhs_shape, &lhs_strides) };
    // Safety: the layer maps `F -> F`, so its `F x 9F` weights read at strides `1, 9F` stay inside them.
    let rhs_tensor = unsafe { tensor::<R>(&buffers.weights, &rhs_shape, &rhs_strides) };
    // Safety: the output holds `samples x F` values contiguously, as `convolve` checks.
    let out_tensor = unsafe { tensor::<R>(output, &out_shape, &out_strides) };
    let lhs = InputBinding::new(lhs_tensor, dtype);
    let rhs = InputBinding::new(rhs_tensor, dtype);
    let globals = uniform_globals(dtype);
    let mut dtypes = MatmulElems::from_globals(&globals);
    // `DoubleUnit` gives wrong results when the depth of `9 * F` is 8 more than a multiple of 16.
    let strategy = MatmulStrategy::SimpleUnit(Default::default());
    let launched = cubek_matmul::launch::launch_ref(&strategy, client, lhs, rhs, out_tensor, &mut dtypes);

    launched.map_err(matmul_failed)
}
