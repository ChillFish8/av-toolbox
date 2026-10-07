// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::{Handle, ServerError};
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::launch::Strategy as MatmulStrategy;
use cubek_std::InputBinding;

use super::kernels::{BodyConvolution, Candidate, Epilogue, Kernels, run_tuned, takes_tuner};
use crate::Dispatch;
use crate::cnn::convolution::{
    Operands,
    contiguous,
    convolve_on_tiles,
    launch_fused_convolution,
    matmul_failed,
    tensor,
    uniform_globals,
};
use crate::cnn::{ConvLayer, Precision, Region};

/// How the body's `F -> F` convolutions run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BodyPath {
    /// `cubek-convolution` on cooperative-matrix tiles, applying each layer's activation and residual as it
    /// writes.
    Convolution,
    /// `cubek-convolution` on cooperative-matrix tiles, with each layer's activation and residual in a
    /// separate pass.
    UnfusedConvolution,
    /// An im2col copy of the maps, then a `cubek-matmul` unit matmul.
    Im2col,
}

/// Why a forward pass could not run.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CnnError {
    #[error("the convolution path could not launch: {0}")]
    Convolution(String),

    #[error("the im2col path's matmul could not launch: {0}")]
    Matmul(String),

    #[error("the im2col path needs a columns buffer, but none was given")]
    MissingColumns,

    #[error("the device failed while running the network: {0}")]
    Device(String),
}

impl From<ServerError> for CnnError {
    fn from(error: ServerError) -> Self {
        Self::Device(error.to_string())
    }
}

/// What the epilogue applies after adding the bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    None,
    Silu,
    Relu,
}

#[cube(launch_unchecked)]
fn epilogue_kernel<F: Float>(
    values: &mut Array<F>,
    bias: &Array<F>,
    residual: &Array<F>,
    features: u32,
    add_bias: u32,
    silu: u32,
    relu: u32,
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

    if relu == 1 {
        value = f32::max(value, 0.0);
    }

    if add_residual == 1 {
        value += f32::cast_from(residual[ABSOLUTE_POS]);
    }

    values[ABSOLUTE_POS] = F::cast_from(value);
}

/// Adds `layer`'s bias to the NHWC maps in `values`, then applies `activation`, then adds `residual` if
/// given.
///
/// The maps hold `samples` samples of the layer's features, and are updated in place.
#[cfg(any(test, feature = "test-support"))]
pub fn epilogue<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    values: &Handle,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
) {
    run_epilogue(
        client, dispatch, layer, values, samples, activation, residual, true,
    );
}

/// Adds `layer`'s bias to the NHWC maps in `values` if `add_bias` is set, then applies `activation`, then
/// adds `residual` if given.
///
/// The maps hold `samples` samples of the layer's features, and are updated in place.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub(crate) fn run_epilogue<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    values: &Handle,
    samples: u32,
    activation: Activation,
    residual: Option<&Handle>,
    add_bias: bool,
) {
    let features = layer.features;
    let units = (samples * features) as usize;
    let bytes = (units * layer.precision.bytes()) as u64;
    let residual_handle = residual.unwrap_or(values);
    // Without a bias the values stand in for it, and are never read through it.
    let bias_handle = layer.bias.as_ref().unwrap_or(values);
    let adds_bias = add_bias && layer.bias.is_some();
    assert!(
        values.size_in_used() >= bytes,
        "the values are smaller than the samples' maps"
    );
    assert!(
        residual_handle.size_in_used() >= bytes,
        "the residual is smaller than the values"
    );

    if let Some(bias) = layer.bias.as_ref() {
        let bias_bytes = u64::from(features) * layer.precision.bytes() as u64;
        assert!(
            bias.size_in_used() >= bias_bytes,
            "the bias is smaller than the features"
        );
    }

    let bias_flag = u32::from(adds_bias);
    let silu = u32::from(activation == Activation::Silu);
    let relu = u32::from(activation == Activation::Relu);
    let add_residual = u32::from(residual.is_some());
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the asserts above show the values and the residual hold `samples * F` values and the bias, when
    // there is one, `F`. Without a residual or a bias, the values are bound in its place and never read
    // through it.
    unsafe {
        let values_array = ArrayArg::from_raw_parts(values.clone(), units);
        let bias = ArrayArg::from_raw_parts(bias_handle.clone(), features as usize);
        let residual_array = ArrayArg::from_raw_parts(residual_handle.clone(), units);

        match layer.precision {
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
                relu,
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
                relu,
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

/// Convolves the NHWC maps of `region` in `input` with `layer`'s weights, into `output`. Gives whether the
/// forced blueprint ran.
///
/// The convolution paths add the bias inside the convolution. The
/// [BodyPath::Im2col](crate::cnn::BodyPath::Im2col) path leaves the bias out, and stages the maps in
/// `columns`, which holds nine times as many values as a map.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub(crate) fn convolve<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    path: BodyPath,
    layer: &ConvLayer,
    input: &Handle,
    output: &Handle,
    columns: Option<&Handle>,
    region: Region,
) -> Result<bool, CnnError> {
    check_layer_maps(layer, input, output, region);

    match path {
        BodyPath::Convolution | BodyPath::UnfusedConvolution => {
            let operands = body_operands(layer, true);

            convolve_on_tiles(client, &operands, input, output, region, true)
        },
        BodyPath::Im2col => {
            let columns = columns.ok_or(CnnError::MissingColumns)?;
            let bytes = layer.map_bytes(region);
            assert!(
                columns.size_in_used() >= bytes * 9,
                "the columns are smaller than nine of the region's maps"
            );

            match layer.precision {
                Precision::F16 => convolve_by_columns::<R, half::f16>(
                    client, dispatch, layer, input, output, columns, region,
                )?,
                Precision::F32 => {
                    convolve_by_columns::<R, f32>(client, dispatch, layer, input, output, columns, region)?
                },
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
pub fn convolve_with_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    input: &Handle,
    output: &Handle,
    region: Region,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    if allow_forced {
        let path = BodyPath::Convolution;

        return convolve(client, dispatch, path, layer, input, output, None, region);
    }

    check_layer_maps(layer, input, output, region);

    let operands = body_operands(layer, true);

    convolve_on_tiles(client, &operands, input, output, region, false)
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights on cooperative-matrix
/// tiles, into `output` without the bias. Gives whether the forced blueprint ran.
#[cfg(any(test, feature = "test-support"))]
pub fn convolve_without_bias<R: Runtime>(
    client: &ComputeClient<R>,
    layer: &ConvLayer,
    input: &Handle,
    output: &Handle,
    region: Region,
) -> Result<bool, CnnError> {
    check_layer_maps(layer, input, output, region);

    let operands = body_operands(layer, false);

    convolve_on_tiles(client, &operands, input, output, region, true)
}

/// Asserts that `input` and `output` each hold the region's maps of `layer`.
fn check_layer_maps(layer: &ConvLayer, input: &Handle, output: &Handle, region: Region) {
    let bytes = layer.map_bytes(region);
    assert!(
        input.size_in_used() >= bytes,
        "the input is smaller than the region's maps"
    );
    assert!(
        output.size_in_used() >= bytes,
        "the output is smaller than the region's maps"
    );
}

/// The operands of `layer`, with its bias if `with_bias` is set and the layer has one.
pub(crate) fn body_operands(layer: &ConvLayer, with_bias: bool) -> Operands<'_> {
    let bias = layer.bias.as_ref().filter(|_| with_bias);
    let features = layer.features as usize;

    Operands::body(&layer.weights, bias, features, layer.precision)
}

/// Convolves the NHWC maps of `region` in `input` with `layer`'s `F -> F` weights and bias on
/// cooperative-matrix tiles into `output`, applying `finish` as each value is written.
///
/// With `allow_forced` the blueprint is forced where that is known to be faster and exact, and otherwise
/// `cubek-convolution` infers one.
fn convolve_fused<R: Runtime>(
    client: &ComputeClient<R>,
    layer: &ConvLayer,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<LayerRun, CnnError> {
    check_layer_maps(layer, input, output, region);

    let operands = body_operands(layer, true);
    let forced = launch_fused_convolution(client, &operands, input, output, region, finish, allow_forced)?;

    Ok(LayerRun {
        forced,
        fused: true,
        candidate: None,
        winograd_offered: false,
    })
}

/// How a body layer's convolution ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerRun {
    /// Whether the convolution took the forced blueprint.
    pub forced: bool,
    /// Whether the layer's activation and residual were applied as the convolution wrote, rather than in a
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

/// Runs `layer` over the NHWC maps of `region` in `input` on `path`, into `output`. That is its convolution
/// and bias, then `finish`.
///
/// [BodyPath::Im2col](crate::cnn::BodyPath::Im2col) stages the maps in `columns`, which holds nine times as
/// many values as a map.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub(crate) fn body_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    path: BodyPath,
    input: &Handle,
    output: &Handle,
    columns: Option<&Handle>,
    region: Region,
    finish: Finish<'_>,
) -> Result<LayerRun, CnnError> {
    if path == BodyPath::Convolution {
        return convolve_fused(client, layer, input, output, region, finish, true);
    }

    let forced = convolve(client, dispatch, path, layer, input, output, columns, region)?;

    // The unfused convolution adds the bias inside the convolution, and im2col adds it here.
    let add_bias = path == BodyPath::Im2col;
    finish_separately(client, dispatch, layer, output, region, finish, add_bias);

    Ok(LayerRun {
        forced,
        fused: false,
        candidate: None,
        winograd_offered: false,
    })
}

/// The maps one layer reads and writes.
///
/// `columns` holds nine times as many values as a map, and is needed on
/// [BodyPath::Im2col](crate::cnn::BodyPath::Im2col) only.
#[derive(Clone, Copy)]
pub struct LayerMaps<'a> {
    pub input: &'a Handle,
    pub output: &'a Handle,
    pub columns: Option<&'a Handle>,
}

/// Runs `layer` over the NHWC maps of `region` on `path`, that is its convolution and bias, then `finish`.
/// Gives how it ran.
///
/// On the convolution path with f16 maps whose features are a multiple of 16, the layer runs with the
/// candidate `kernels` chooses, tuned under `epilogue`.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub fn run_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    kernels: &Kernels,
    path: BodyPath,
    layer: &ConvLayer,
    maps: LayerMaps<'_>,
    region: Region,
    finish: Finish<'_>,
    epilogue: Epilogue,
) -> Result<LayerRun, CnnError> {
    if takes_tuner(layer, path) {
        let convolution = BodyConvolution::of_conv(layer, maps.input, maps.output, region, finish);
        let tuned = run_tuned(client, kernels, &convolution, epilogue)?;

        return Ok(tuned.run);
    }

    body_layer(
        client,
        dispatch,
        layer,
        path,
        maps.input,
        maps.output,
        maps.columns,
        region,
        finish,
    )
}

/// Runs `layer` on `path` as a forward pass does, keeping the inferred blueprint unless `allow_forced` is
/// set.
#[cfg(any(test, feature = "test-support"))]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the layer"
)]
pub fn body_layer_with_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    path: BodyPath,
    input: &Handle,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    allow_forced: bool,
) -> Result<LayerRun, CnnError> {
    assert!(
        path != BodyPath::Im2col,
        "only the convolution paths choose a blueprint"
    );

    if allow_forced {
        return body_layer(client, dispatch, layer, path, input, output, None, region, finish);
    }

    if path == BodyPath::Convolution {
        return convolve_fused(client, layer, input, output, region, finish, false);
    }

    convolve_with_blueprint(client, dispatch, layer, input, output, region, false)?;
    finish_separately(client, dispatch, layer, output, region, finish, false);

    Ok(LayerRun {
        forced: false,
        fused: false,
        candidate: None,
        winograd_offered: false,
    })
}

/// Applies `finish` to `layer`'s maps of `region` in `output` in a separate pass, adding the bias first if
/// `add_bias` is set.
fn finish_separately<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    output: &Handle,
    region: Region,
    finish: Finish<'_>,
    add_bias: bool,
) {
    let samples = region.samples();

    run_epilogue(
        client,
        dispatch,
        layer,
        output,
        samples,
        finish.activation,
        finish.residual,
        add_bias,
    );
}

fn convolve_by_columns<R: Runtime, F: Float>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &ConvLayer,
    input: &Handle,
    output: &Handle,
    columns: &Handle,
    region: Region,
) -> Result<(), CnnError> {
    let channels = layer.features;
    let features = channels as usize;
    let samples = region.samples() as usize;
    let depth = 9 * features;
    let column_values = samples * depth;
    assert!(
        u32::try_from(column_values).is_ok(),
        "the columns hold more values than the kernel can index"
    );

    let weight_bytes = (features * depth * layer.precision.bytes()) as u64;
    assert!(
        layer.weights.size_in_used() >= weight_bytes,
        "the weights are smaller than the layer's"
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
    // Safety: the weights hold `F x 9F` values, as just asserted, so reads at strides `1, 9F` stay inside
    // them.
    let rhs_tensor = unsafe { tensor::<R>(&layer.weights, &rhs_shape, &rhs_strides) };
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
