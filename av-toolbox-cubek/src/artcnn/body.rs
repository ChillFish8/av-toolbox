// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};
use cubecl::zspace::{Shape, Strides};
use cubek_convolution::components::ConvSetupError;
use cubek_convolution::{AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy};
use cubek_matmul::definition::{MatmulElems, MatmulGlobalElems, MatmulSetupError};
use cubek_matmul::launch::Strategy as MatmulStrategy;
use cubek_std::InputBinding;

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
/// the bias.
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
) -> Result<(), ArtcnnError> {
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

    match path {
        BodyPath::Convolution => match network.precision() {
            Precision::F16 => {
                convolve_directly::<R, half::f16>(client, network, layer, input, output, region)
            },
            Precision::F32 => convolve_directly::<R, f32>(client, network, layer, input, output, region),
        },
        BodyPath::Im2col => {
            let columns = columns.ok_or(ArtcnnError::MissingColumns)?;
            assert!(
                columns.size_in_used() >= bytes * 9,
                "the columns are smaller than nine of the region's maps"
            );

            match network.precision() {
                Precision::F16 => convolve_by_columns::<R, half::f16>(
                    client, dispatch, network, layer, input, output, columns, region,
                ),
                Precision::F32 => convolve_by_columns::<R, f32>(
                    client, dispatch, network, layer, input, output, columns, region,
                ),
            }
        },
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

fn convolve_directly<R: Runtime, F: Float>(
    client: &ComputeClient<R>,
    network: &Network,
    layer: usize,
    input: &Handle,
    output: &Handle,
    region: Region,
) -> Result<(), ArtcnnError> {
    let buffers = network.layer(layer);
    let features = network.features() as usize;
    let height = region.height as usize;
    let width = region.width as usize;
    let dtype = F::as_type_native_unchecked().storage_type();

    let map_shape = [1, height, width, features];
    let weight_shape = [features, 3, 3, features];
    let map_strides = contiguous(&map_shape);
    let weight_strides = contiguous(&weight_shape);
    // Safety: the input and output hold the region's `1 x H x W x F` maps contiguously, as `convolve` checks.
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
    let strategy = Strategy::Inferred {
        algorithm: ConvAlgorithm::SimpleSyncCyclic,
        tile_kind: AcceleratedTileKind::Cmma,
    };
    let globals = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    };
    let dtypes = MatmulElems::from_globals(&globals);
    let launched = cubek_convolution::launch_ref::<R, 2>(&strategy, client, inputs, arguments, dtypes);

    launched.map_err(convolution_failed)
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
    let globals = MatmulGlobalElems {
        lhs: dtype,
        rhs: dtype,
        out: dtype,
    };
    let mut dtypes = MatmulElems::from_globals(&globals);
    // `DoubleUnit` gives wrong results when the depth of `9 * F` is 8 more than a multiple of 16.
    let strategy = MatmulStrategy::SimpleUnit(Default::default());
    let launched = cubek_matmul::launch::launch_ref(&strategy, client, lhs, rhs, out_tensor, &mut dtypes);

    launched.map_err(matmul_failed)
}
