use std::collections::BTreeSet;

use cubecl::features::MmaConfig;
use cubecl::ir::{AddressType, ElemType};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::components::{
    ConvSetupError,
    ConvolutionOperation,
    ConvolutionProblem,
    Dimensionality,
};
use cubek_convolution::definition::{ConvBlueprint, ForwardBlueprint};
use cubek_convolution::{AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy};
use cubek_matmul::definition::{AccumulatorOperand, MatmulElems, MatmulGlobalElems, MatmulSetupError};
use cubek_matmul::multi_level::components::stage::PartitionBuffering;
use cubek_matmul::multi_level::components::tile::TileMatmulKind;
use cubek_matmul::multi_level::definition::{BatchMatmulBlueprint, TilingScheme};
use cubek_std::cube_count::{CubeCountStrategy, GlobalOrder, HypercubeBlueprint};
use cubek_std::{InputBinding, MatrixLayout};

use super::fused::launch_fused;
use crate::cnn::body::Finish;
use crate::cnn::{CnnError, PADDED_CHANNELS, PaddedLastLayer, Precision, Region};

/// The fewest samples a region needs for the forced blueprint to beat the inferred one.
const FORCED_SAMPLES: usize = 8192;

/// The one plane width the forced blueprint is known to give correct results on.
const FORCED_PLANE_SIZE: u32 = 64;

/// The most channels the forced blueprint was measured at.
const FORCED_MAX_CHANNELS: usize = 96;

/// The side of the cooperative-matrix tiles the forced blueprint multiplies.
const FORCED_TILE: u32 = 16;

/// How many planes a workgroup of the forced blueprint runs.
const FORCED_PLANES: u32 = 4;

/// How many sample tiles each plane of the forced blueprint covers.
const FORCED_PARTITION_ROWS: u32 = 4;

/// How many input channel tiles each plane of the forced blueprint covers at once.
const FORCED_PARTITION_DEPTH: u32 = 2;

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

/// The shared memory in bytes one workgroup of the forced blueprint allocates for `channels` output channels,
/// with a bias.
///
/// A workgroup stages a block of samples by input channels in f16, the f16 weights of those input channels by
/// `channels`, an f32 accumulator of the block's samples by `channels`, and one f32 output tile per plane.
pub fn forced_shared_bytes(channels: usize) -> usize {
    let stage_samples = (FORCED_PLANES * FORCED_PARTITION_ROWS * FORCED_TILE) as usize;
    let stage_depth = (FORCED_PARTITION_DEPTH * FORCED_TILE) as usize;
    let tile_values = (FORCED_TILE * FORCED_TILE) as usize;
    let input = stage_samples * stage_depth * 2;
    let weights = stage_depth * channels * 2;
    let accumulator = stage_samples * channels * 4;
    let output = FORCED_PLANES as usize * tile_values * 4;

    input + weights + accumulator + output
}

/// A blueprint of 16x16x16 cooperative-matrix tiles for `problem`, or `None` to let `cubek-convolution`
/// infer one.
///
/// Each workgroup computes a block of 256 samples by `n` channels in four planes. The blueprint
/// is only given where it is faster and known to be exact. That needs f16 operands, a device offering a
/// 16x16x16 f16 cooperative matrix accumulating in f32 (`offers_cmma_16`), 64 lanes per plane, `n` a
/// multiple of 16 up to 96, at least 8192 samples, and a bias stage that fits in `max_shared_memory`.
pub(crate) fn forced_blueprint(
    problem: &ConvolutionProblem,
    plane_size_max: u32,
    offers_cmma_16: bool,
    max_shared_memory: usize,
) -> Option<ConvBlueprint> {
    let half_type = elem_type(Precision::F16);
    let operands_are_half = problem.global_dtypes.lhs == half_type && problem.global_dtypes.rhs == half_type;
    let tiled_channels = problem.n.is_multiple_of(FORCED_TILE as usize) && problem.n <= FORCED_MAX_CHANNELS;
    let large = problem.m >= FORCED_SAMPLES;

    if !operands_are_half || !offers_cmma_16 || plane_size_max != FORCED_PLANE_SIZE {
        return None;
    }

    if !tiled_channels || !large {
        return None;
    }

    if forced_shared_bytes(problem.n) > max_shared_memory {
        return None;
    }

    let channel_tiles = problem.n as u32 / FORCED_TILE;
    let scheme = TilingScheme::builder()
        .with_tile_size((FORCED_TILE, FORCED_TILE, FORCED_TILE).into())
        .with_partition_size((FORCED_PARTITION_ROWS, channel_tiles, FORCED_PARTITION_DEPTH).into())
        .with_stage_size((FORCED_PLANES, 1, 1).into())
        .build()
        .expect("every size of the scheme is set");
    let hypercube = HypercubeBlueprint::builder()
        .global_order(GlobalOrder::SwizzleRow(4))
        .cube_count_strategy(CubeCountStrategy::FromProblem)
        .build();

    let matmul_problem = problem.as_matmul_problem(AccumulatorOperand::Present);
    let matmul = BatchMatmulBlueprint::builder(TileMatmulKind::Cmma, scheme, plane_size_max, &matmul_problem)
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
pub fn offers_cmma_16(client: &Client) -> bool {
    let configs = &client.properties().features.matmul.cmma;

    offers_cmma_16_in(configs)
}

/// Whether `configs` holds a 16x16x16 f16 cooperative matrix accumulating in f32.
pub(crate) fn offers_cmma_16_in(configs: &BTreeSet<MmaConfig>) -> bool {
    let half_type = elem_type(Precision::F16);
    let float_type = elem_type(Precision::F32);
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
fn chosen_blueprint(
    client: &Client,
    problem: &ConvolutionProblem,
    max_shared_memory: usize,
) -> Option<ConvBlueprint> {
    let plane_size_max = client.properties().hardware.plane_size_max;
    let offers_cmma = offers_cmma_16(client);

    forced_blueprint(problem, plane_size_max, offers_cmma, max_shared_memory)
}

/// Whether a body convolution of `region` on maps of `features` channels in `precision`'s type takes the
/// forced blueprint on the client's device.
#[cfg(any(test, feature = "test-support"))]
pub fn takes_forced_blueprint(client: &Client, features: u32, precision: Precision, region: Region) -> bool {
    let features = features as usize;
    let dtype = elem_type(precision);
    let globals = uniform_globals(dtype);
    let map_values = region.samples() as usize * features;
    let address_type = AddressType::from_len(map_values);
    let problem = convolution_problem(region, features, features, globals, address_type);
    let max_shared_memory = client.properties().hardware.max_shared_memory_size;
    let blueprint = chosen_blueprint(client, &problem, max_shared_memory);

    blueprint.is_some()
}

/// Launches an f16 body convolution of `features` channels over `region` with the forced blueprint whatever
/// the device's shared memory, on zeroed-or-unwritten buffers. Gives whether the forced blueprint ran.
#[cfg(any(test, feature = "test-support"))]
pub fn convolve_forced_past_the_shared_memory_limit(
    client: &Client,
    features: u32,
    region: Region,
) -> Result<bool, CnnError> {
    let channels = features as usize;
    let samples = region.samples() as usize;
    let weights = client.empty(channels * 9 * channels * 2);
    let bias = client.empty(channels * 2);
    let input = client.empty(samples * channels * 2);
    let output = client.empty(samples * channels * 2);
    let operands = Operands::body(&weights, Some(&bias), channels, Precision::F16);

    convolve_on_tiles_within(client, &operands, &input, &output, region, true, usize::MAX)
}

pub(crate) fn elem_type(precision: Precision) -> ElemType {
    match precision {
        Precision::F16 => half::f16::elem_type_native(),
        Precision::F32 => f32::elem_type_native(),
    }
}

/// Global types with `dtype` for both operands and the output.
pub(crate) fn uniform_globals(dtype: ElemType) -> MatmulGlobalElems {
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
pub(crate) unsafe fn tensor(handle: &Handle, shape: &[usize], strides: &[usize]) -> TensorBinding {
    let shape_value = Shape::from(shape.to_vec());
    let strides_value = Strides::from(strides.to_vec());

    // Safety: the caller guarantees the handle holds the shape's elements at these strides.
    unsafe { TensorBinding::from_raw_parts(handle.clone(), strides_value, shape_value) }
}

pub(crate) fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }

    strides
}

/// Describes a convolution setup failure with `Debug`, since `ConvSetupError` has no `Display`.
pub(crate) fn convolution_failed(error: ConvSetupError) -> CnnError {
    CnnError::Convolution(format!("{error:?}"))
}

pub(crate) fn matmul_failed(error: MatmulSetupError) -> CnnError {
    CnnError::Matmul(error.to_string())
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
    /// The operands of an `F -> F` body convolution with `weights` and its `bias`, if any, in `precision`'s
    /// type.
    pub(crate) fn body(
        weights: &'a Handle,
        bias: Option<&'a Handle>,
        features: usize,
        precision: Precision,
    ) -> Self {
        let dtype = elem_type(precision);
        let globals = uniform_globals(dtype);

        Operands {
            weights,
            bias,
            in_channels: features,
            out_channels: features,
            globals,
        }
    }
}

/// One 3x3 convolution's tensors bound for `cubek-convolution`, with its problem and the blueprint it takes.
pub(crate) struct BoundConvolution {
    pub(crate) problem: ConvolutionProblem,
    pub(crate) blueprint: Option<ConvBlueprint>,
    pub(crate) input: InputBinding,
    pub(crate) weight: InputBinding,
    pub(crate) bias: Option<InputBinding>,
    pub(crate) output: TensorBinding,
}

/// Binds a convolution of the NHWC maps of `region` in `input` with `operands`, into NHWC maps of the output
/// channels in `output`.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
fn bind_convolution(
    client: &Client,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
    max_shared_memory: usize,
) -> BoundConvolution {
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
    let input_tensor = unsafe { tensor(input, &map_shape, &map_strides) };
    // Safety: the weights hold `out x 3 x 3 x in` values contiguously, as `check_operand_sizes` asserts.
    let weight_tensor = unsafe { tensor(operands.weights, &weight_shape, &weight_strides) };
    // Safety: the output holds the region's `1 x H x W x out` maps contiguously, as `check_operand_sizes`
    // asserts.
    let output_tensor = unsafe { tensor(output, &out_shape, &out_strides) };
    // Safety: the bias holds `out` values contiguously, as `check_operand_sizes` asserts.
    let bias_tensor = operands
        .bias
        .map(|bias| unsafe { tensor(bias, &bias_shape, &bias_strides) });

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
        chosen_blueprint(client, &problem, max_shared_memory)
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
fn required_address_type(
    input: &InputBinding,
    weight: &InputBinding,
    bias: Option<&InputBinding>,
    output: &TensorBinding,
    out_type: ElemType,
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
pub(crate) fn convolve_on_tiles(
    client: &Client,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    let max_shared_memory = client.properties().hardware.max_shared_memory_size;

    convolve_on_tiles_within(
        client,
        operands,
        input,
        output,
        region,
        allow_forced,
        max_shared_memory,
    )
}

/// The same as [convolve_on_tiles], with the forced blueprint judged against `max_shared_memory`.
fn convolve_on_tiles_within(
    client: &Client,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
    max_shared_memory: usize,
) -> Result<bool, CnnError> {
    let bound = bind_convolution(
        client,
        operands,
        input,
        output,
        region,
        allow_forced,
        max_shared_memory,
    );
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
    let launched = cubek_convolution::launch_ref::<2>(&strategy, client, inputs, arguments, dtypes);

    launched.map_err(convolution_failed)?;

    Ok(forced)
}

/// Runs `layer` over the NHWC maps of `region` in `input` on cooperative-matrix tiles, into `scratch` as NHWC
/// maps of 16 f32 channels. Gives whether the forced blueprint ran.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
pub fn run_padded_layer(
    client: &Client,
    layer: &PaddedLastLayer,
    input: &Handle,
    region: Region,
    scratch: &Handle,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    let dtype = elem_type(layer.precision);
    let float_type = elem_type(Precision::F32);
    let globals = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: float_type,
    };
    let operands = Operands {
        weights: &layer.weights,
        bias: Some(&layer.bias),
        in_channels: layer.in_channels as usize,
        out_channels: PADDED_CHANNELS,
        globals,
    };

    convolve_on_tiles(client, &operands, input, scratch, region, allow_forced)
}

/// Convolves the NHWC maps of `region` in `input` with `operands` on cooperative-matrix tiles into `output`,
/// applying `finish`'s activation and residual as each value is written. Gives whether the forced blueprint
/// ran.
///
/// `finish` must not carry a skip.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
pub(crate) fn launch_fused_convolution(
    client: &Client,
    operands: &Operands<'_>,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    debug_assert!(finish.skip.is_none(), "the fused convolution does not add a skip");

    let max_shared_memory = client.properties().hardware.max_shared_memory_size;
    let bound = bind_convolution(
        client,
        operands,
        input,
        output,
        region,
        allow_forced,
        max_shared_memory,
    );
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
        unsafe { tensor(handle, &shapes.out_shape, &shapes.out_strides) }
    });

    launch_fused(client, bound, residual, finish.activation)?;

    Ok(forced)
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
