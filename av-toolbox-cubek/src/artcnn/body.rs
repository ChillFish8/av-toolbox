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

use super::network::map_bytes;
use crate::Dispatch;
use crate::artcnn::{Network, Precision, Region};

/// How the body's `F -> F` convolutions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPath {
    /// `cubek-convolution` on cooperative-matrix tiles.
    Convolution,
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
    silu: u32,
    add_residual: u32,
) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let channel = ABSOLUTE_POS as u32 % features;
    let mut value = f32::cast_from(values[ABSOLUTE_POS]) + f32::cast_from(bias[channel as usize]);

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

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights, into `output` without
/// the bias. Gives whether the forced blueprint ran.
///
/// The [BodyPath::Im2col](crate::artcnn::BodyPath::Im2col) path stages the maps in `columns`, which holds
/// nine times as many values as a map.
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
        BodyPath::Convolution => {
            let blueprint = body_blueprint(client, network, region);
            let forced = blueprint.is_some();

            convolve_directly(client, network, layer, input, output, region, blueprint.as_ref())?;

            Ok(forced)
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

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights on cooperative-matrix
/// tiles, into `output` without the bias. Gives whether the forced blueprint ran.
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
    convolve_directly(client, network, layer, input, output, region, None)?;

    Ok(false)
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

/// The NHWC map and OHWI weight shapes of a 3x3 convolution over `region`, with their contiguous strides.
struct ConvolutionShapes {
    map_shape: [usize; 4],
    map_strides: Vec<usize>,
    weight_shape: [usize; 4],
    weight_strides: Vec<usize>,
}

fn convolution_shapes(region: Region, in_channels: usize, out_channels: usize) -> ConvolutionShapes {
    let height = region.height as usize;
    let width = region.width as usize;
    let map_shape = [1, height, width, in_channels];
    let weight_shape = [out_channels, 3, 3, in_channels];
    let map_strides = contiguous(&map_shape);
    let weight_strides = contiguous(&weight_shape);

    ConvolutionShapes {
        map_shape,
        map_strides,
        weight_shape,
        weight_strides,
    }
}

/// The problem `cubek-convolution` solves for a 3x3 convolution of `region`'s NHWC maps, with one sample
/// of zero padding, from `in_channels` to `out_channels`.
pub(crate) fn convolution_problem(
    region: Region,
    in_channels: usize,
    out_channels: usize,
    globals: MatmulGlobalElems,
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
        address_type: AddressType::U32,
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

/// The blueprint the body's convolutions over `region` take, or `None` to infer one.
fn body_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    region: Region,
) -> Option<ConvBlueprint> {
    let features = network.features() as usize;
    let dtype = storage_type(network.precision());
    let globals = uniform_globals(dtype);
    let problem = convolution_problem(region, features, features, globals);
    let plane_size_max = client.properties().hardware.plane_size_max;
    let offers_cmma = offers_cmma_16(client);

    forced_blueprint(&problem, plane_size_max, offers_cmma)
}

fn storage_type(precision: Precision) -> StorageType {
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
fn convolution_failed(error: ConvSetupError) -> ArtcnnError {
    ArtcnnError::Convolution(format!("{error:?}"))
}

fn matmul_failed(error: MatmulSetupError) -> ArtcnnError {
    ArtcnnError::Matmul(error.to_string())
}

/// Convolves on cooperative-matrix tiles with `blueprint`, or an inferred blueprint when it is `None`.
fn convolve_directly<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
    blueprint: Option<&ConvBlueprint>,
) -> Result<(), ArtcnnError> {
    let buffers = network.layer(layer);
    let features = network.features() as usize;
    let dtype = storage_type(network.precision());

    let ConvolutionShapes {
        map_shape,
        map_strides,
        weight_shape,
        weight_strides,
    } = convolution_shapes(region, features, features);
    // Safety: the input and output hold the region's `1 x H x W x F` maps contiguously, as every caller
    // checks through `check_layer_maps`.
    let input_tensor = unsafe { tensor::<R>(input, &map_shape, &map_strides) };
    // Safety: the layer maps `F -> F`, so its weights hold `F x 3 x 3 x F` values contiguously.
    let weight_tensor = unsafe { tensor::<R>(&buffers.weights, &weight_shape, &weight_strides) };
    // Safety: as for the input.
    let output_tensor = unsafe { tensor::<R>(output, &map_shape, &map_strides) };

    let inputs = ConvolutionInputs::Forward {
        input: InputBinding::new(input_tensor, dtype),
        weight: InputBinding::new(weight_tensor, dtype),
        bias: None,
        out: output_tensor,
    };
    let arguments = ConvolutionArgs::<2> {
        stride: [1, 1],
        padding: [1, 1],
        dilation: [1, 1],
    };
    let strategy = convolution_strategy(blueprint);
    let globals = uniform_globals(dtype);
    let dtypes = MatmulElems::from_globals(&globals);
    let launched = cubek_convolution::launch_ref::<R, 2>(&strategy, client, inputs, arguments, dtypes);

    launched.map_err(convolution_failed)
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
