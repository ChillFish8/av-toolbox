// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;
use cubek_matmul::definition::MatmulElems;
use cubek_matmul::launch::Strategy as MatmulStrategy;
use cubek_std::InputBinding;

use crate::cnn::convolution::{
    contiguous,
    matmul_failed,
    offers_cmma_16,
    storage_type,
    tensor,
    uniform_globals,
};
use crate::cnn::{
    BodyPath,
    CnnError,
    PADDED_CHANNELS,
    PaddedLastLayer,
    Precision,
    Region,
    add_taps,
    encode,
    run_padded_layer,
};
use crate::dpir::{LayerKind, WIDTHS, WeightLayer};
use crate::{DevicePlane, Dispatch, assert_plane, mirror};

/// The features the head writes.
const HEAD_FEATURES: u32 = WIDTHS[0];

/// The most image planes a frame has.
const MAX_IMAGE_CHANNELS: usize = 3;

/// The scale of the strength channel, the 8-bit code range.
const STRENGTH_SCALE: f32 = 255.0;

/// Where a padded frame's region sits, and how it maps back onto the planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placement {
    /// The region, in padded frame samples.
    pub region: Region,
    /// The padding before the plane's first column.
    pub left: u32,
    /// The padding before the plane's first row.
    pub top: u32,
}

/// DRUNet's first 3x3 convolution on the device, from the image and strength channels to 64 features.
///
/// The weights are laid out by output channel, row, column and input channel, in the feature maps' type.
pub struct HeadLayer {
    weights: Handle,
    in_channels: u32,
    precision: Precision,
}

impl HeadLayer {
    /// Uploads `layer` in `precision`'s type.
    ///
    /// Panics if `layer` is not a 3x3 convolution from one or three image channels and the strength channel
    /// to 64 features.
    pub fn upload<R: Runtime>(client: &ComputeClient<R>, layer: &WeightLayer, precision: Precision) -> Self {
        let image_channels = layer.in_channels.saturating_sub(1) as usize;
        let shaped = layer.kind == LayerKind::Conv && layer.out_channels == HEAD_FEATURES;
        assert!(
            shaped && (1..=MAX_IMAGE_CHANNELS).contains(&image_channels),
            "the head is not a 3x3 convolution from the image and strength channels to 64 features"
        );

        let count = layer.kind.value_count(layer.in_channels, layer.out_channels);
        assert_eq!(
            layer.values.len(),
            count,
            "the head's values do not match its channels"
        );

        let bytes = encode(&layer.values, precision);
        let weights = client.create_from_slice(&bytes);

        Self {
            weights,
            in_channels: layer.in_channels,
            precision,
        }
    }
}

/// The sample at `index` of image plane `channel`.
#[cube]
fn plane_sample(
    first: &Array<f32>,
    second: &Array<f32>,
    third: &Array<f32>,
    channel: u32,
    index: usize,
) -> f32 {
    let mut value = first[index];
    if channel == 1 {
        value = second[index];
    } else if channel == 2 {
        value = third[index];
    }

    value
}

#[cube(launch_unchecked)]
fn head_kernel<F: Float>(
    first: &Array<f32>,
    second: &Array<f32>,
    third: &Array<f32>,
    weights: &Array<F>,
    output: &mut Array<F>,
    plane_width: u32,
    plane_height: u32,
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    left: u32,
    top: u32,
    image_channels: u32,
    features: u32,
    peak: f32,
    strength: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let feature = position % features;
    let sample = position / features;
    let column = (sample % region_width) as i32;
    let row = (sample / region_width) as i32;
    let in_channels = image_channels + 1;
    let mut sum = 0.0f32;

    for tap in 0..9u32 {
        let source_column = column + (tap % 3) as i32 - 1;
        let source_row = row + (tap / 3) as i32 - 1;
        let inside = source_column >= 0
            && source_row >= 0
            && source_column < region_width as i32
            && source_row < region_height as i32;

        if inside {
            let padded_column = region_x as i32 + source_column - left as i32;
            let padded_row = region_y as i32 + source_row - top as i32;
            let plane_column = mirror(padded_column, plane_width as i32) as u32;
            let plane_row = mirror(padded_row, plane_height as i32) as u32;
            let index = (plane_row * plane_width + plane_column) as usize;
            let tap_start = (feature * 9 + tap) * in_channels;

            for channel in 0..image_channels {
                let sample_value = plane_sample(first, second, third, channel, index);
                let weight = f32::cast_from(weights[(tap_start + channel) as usize]);
                sum += weight * (sample_value / peak);
            }

            let strength_weight = f32::cast_from(weights[(tap_start + image_channels) as usize]);
            sum += strength_weight * strength;
        }
    }

    output[ABSOLUTE_POS] = F::cast_from(sum);
}

#[cube(launch_unchecked)]
fn gather_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>, width: u32, channels: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let gathered = 4 * channels;
    let gathered_channel = position % gathered;
    let sample = position / gathered;
    let half_width = width / 2;
    let tap = gathered_channel / channels;
    let channel = gathered_channel % channels;
    let tap_row = tap / 2;
    let tap_column = tap % 2;
    let source_row = (sample / half_width) * 2 + tap_row;
    let source_column = (sample % half_width) * 2 + tap_column;
    let source = (source_row * width + source_column) * channels + channel;

    output[ABSOLUTE_POS] = input[source as usize];
}

#[cube(launch_unchecked)]
fn scatter_kernel<F: Float>(input: &Array<F>, output: &mut Array<F>, width: u32, channels: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let channel = position % channels;
    let sample = position / channels;
    let doubled_width = width * 2;
    let out_column = sample % doubled_width;
    let out_row = sample / doubled_width;
    let tap_row = out_row % 2;
    let tap_column = out_column % 2;
    let source_sample = (out_row / 2) * width + out_column / 2;
    let tap = tap_row * 2 + tap_column;
    let source = source_sample * 4 * channels + tap * channels + channel;

    output[ABSOLUTE_POS] = input[source as usize];
}

#[cube(launch_unchecked)]
fn add_kernel<F: Float>(values: &mut Array<F>, addend: &Array<F>) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let sum = f32::cast_from(values[ABSOLUTE_POS]) + f32::cast_from(addend[ABSOLUTE_POS]);

    values[ABSOLUTE_POS] = F::cast_from(sum);
}

#[cube(launch_unchecked)]
fn direct_tail_kernel<F: Float>(
    input: &Array<F>,
    weights: &Array<F>,
    bias: &Array<f32>,
    scratch: &mut Array<f32>,
    region_width: u32,
    region_height: u32,
    in_channels: u32,
    out_channels: u32,
    scratch_channels: u32,
) {
    if ABSOLUTE_POS >= scratch.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let channel = position % scratch_channels;
    let sample = position / scratch_channels;
    let column = (sample % region_width) as i32;
    let row = (sample / region_width) as i32;
    let mut sum = 0.0f32;

    if channel < out_channels {
        let bias_value = bias[channel as usize];
        sum = add_taps::<F>(
            bias_value,
            input,
            weights,
            column,
            row,
            region_width,
            region_height,
            channel,
            in_channels,
        );
    }

    scratch[ABSOLUTE_POS] = sum;
}

/// Writes `value` to `index` of image plane `channel`.
#[cube]
fn write_plane(
    first: &mut Array<f32>,
    second: &mut Array<f32>,
    third: &mut Array<f32>,
    channel: u32,
    index: usize,
    value: f32,
) {
    if channel == 0 {
        first[index] = value;
    } else if channel == 1 {
        second[index] = value;
    } else {
        third[index] = value;
    }
}

#[cube(launch_unchecked)]
fn write_out_kernel(
    scratch: &Array<f32>,
    first: &mut Array<f32>,
    second: &mut Array<f32>,
    third: &mut Array<f32>,
    plane_width: u32,
    plane_height: u32,
    region_x: u32,
    region_y: u32,
    region_width: u32,
    left: u32,
    top: u32,
    core_x: u32,
    core_y: u32,
    core_width: u32,
    core_samples: u32,
    channels: u32,
    scratch_channels: u32,
    peak: f32,
) {
    if ABSOLUTE_POS >= (core_samples * channels) as usize {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let channel = position % channels;
    let sample = position / channels;
    let padded_column = core_x + sample % core_width;
    let padded_row = core_y + sample / core_width;
    let plane_column = padded_column as i32 - left as i32;
    let plane_row = padded_row as i32 - top as i32;
    let inside = plane_column >= 0
        && plane_row >= 0
        && plane_column < plane_width as i32
        && plane_row < plane_height as i32;

    if inside {
        let local = (padded_row - region_y) * region_width + padded_column - region_x;
        let value = scratch[(local * scratch_channels + channel) as usize];
        let scaled = f32::clamp(value, 0.0, 1.0) * peak;
        let index = (plane_row as u32 * plane_width + plane_column as u32) as usize;

        write_plane(first, second, third, channel, index, scaled);
    }
}

/// Asserts that `planes` holds between one and three planes of the same size, each held as f32, and gives
/// their width and height.
fn check_planes(planes: &[DevicePlane<'_>]) -> (u32, u32) {
    assert!(
        (1..=MAX_IMAGE_CHANNELS).contains(&planes.len()),
        "a frame has between one and three image planes"
    );

    let width = planes[0].width();
    let height = planes[0].height();
    assert!(width > 0 && height > 0, "the image planes are empty");

    for plane in planes {
        assert_plane(plane, width, height, "image");
    }

    (width, height)
}

/// The three planes a kernel binds, repeating the last of `planes` where the frame has fewer.
fn three_planes<'a>(planes: &[DevicePlane<'a>]) -> [DevicePlane<'a>; 3] {
    let last = planes.len() - 1;

    [planes[0], planes[1.min(last)], planes[2.min(last)]]
}

/// Asserts that `handle` holds `values` values of `precision`'s type, naming it `name`.
fn check_values(handle: &Handle, values: usize, precision: Precision, name: &str) {
    let bytes = (values * precision.bytes()) as u64;

    assert!(
        handle.size_in_used() >= bytes,
        "the {name} is smaller than {values} values"
    );
}

/// Runs the head over the placement's region into NHWC maps of 64 features in `output`.
///
/// The input is each of `planes` divided by `peak`, then a constant channel of `strength / 255`. Each region
/// sample maps onto the planes past the placement's padding, mirroring with the edge repeated, and samples
/// outside the region are zero in every channel.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn head<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &HeadLayer,
    planes: &[DevicePlane<'_>],
    peak: f32,
    strength: f32,
    placement: Placement,
    output: &Handle,
) {
    let (plane_width, plane_height) = check_planes(planes);
    let image_channels = planes.len() as u32;
    assert_eq!(
        layer.in_channels,
        image_channels + 1,
        "the head does not read one channel a plane and the strength"
    );

    let region = placement.region;
    let units = (region.samples() * HEAD_FEATURES) as usize;
    let weight_count = (HEAD_FEATURES * 9 * layer.in_channels) as usize;
    check_values(&layer.weights, weight_count, layer.precision, "head's weights");
    check_values(output, units, layer.precision, "output");

    let [first, second, third] = three_planes(planes);
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let third_array = third.array::<R>();
    let strength_value = strength / STRENGTH_SCALE;
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the weights hold `64 * 9 * in` values and the output the region times 64, as just asserted, and
    // every read of a plane is folded inside it by the mirror.
    unsafe {
        let weights = ArrayArg::from_raw_parts(layer.weights.clone(), weight_count);
        let output_array = ArrayArg::from_raw_parts(output.clone(), units);

        match layer.precision {
            Precision::F16 => head_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                first_array,
                second_array,
                third_array,
                weights,
                output_array,
                plane_width,
                plane_height,
                region.x,
                region.y,
                region.width,
                region.height,
                placement.left,
                placement.top,
                image_channels,
                HEAD_FEATURES,
                peak,
                strength_value,
            ),
            Precision::F32 => head_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                first_array,
                second_array,
                third_array,
                weights,
                output_array,
                plane_width,
                plane_height,
                region.x,
                region.y,
                region.width,
                region.height,
                placement.left,
                placement.top,
                image_channels,
                HEAD_FEATURES,
                peak,
                strength_value,
            ),
        }
    }
}

/// Gathers each 2x2 block of the `width` by `height` NHWC maps in `input` into one sample of `output`.
///
/// Channel `(row * 2 + column) * channels + c` of an output sample holds channel `c` of the block's sample
/// at that row and column.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn gather<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: &Handle,
    width: u32,
    height: u32,
    channels: u32,
    precision: Precision,
    output: &Handle,
) {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "the maps do not split into 2x2 blocks"
    );

    let count = (width * height * channels) as usize;
    check_values(input, count, precision, "input");
    check_values(output, count, precision, "output");

    let cubes = dispatch.cube_count(count);
    let unit = dispatch.cube_dim();

    // Safety: the input and output each hold the maps' values, as just asserted, and every read lands inside
    // the input's whole 2x2 blocks.
    unsafe {
        let input_array = ArrayArg::from_raw_parts(input.clone(), count);
        let output_array = ArrayArg::from_raw_parts(output.clone(), count);

        match precision {
            Precision::F16 => gather_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                input_array,
                output_array,
                width,
                channels,
            ),
            Precision::F32 => gather_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                input_array,
                output_array,
                width,
                channels,
            ),
        }
    }
}

/// The matmul a down or up layer's product runs on.
pub(super) fn product_strategy(precision: Precision, offers_matrix: bool) -> MatmulStrategy {
    if precision == Precision::F16 && offers_matrix {
        return MatmulStrategy::SimpleCyclicCmma(Default::default());
    }

    MatmulStrategy::SimpleUnit(Default::default())
}

/// Multiplies the `rows` by `depth` matrix in `lhs` by the `depth` by `columns` matrix in `rhs` into the
/// `rows` by `columns` matrix in `output`, each held contiguously in `precision`'s type.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the product"
)]
pub fn matrix_product<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &Handle,
    rows: usize,
    depth: usize,
    rhs: &Handle,
    columns: usize,
    precision: Precision,
    output: &Handle,
) -> Result<(), CnnError> {
    check_values(lhs, rows * depth, precision, "left matrix");
    check_values(rhs, depth * columns, precision, "right matrix");
    check_values(output, rows * columns, precision, "output");

    let dtype = storage_type(precision);
    let lhs_shape = [rows, depth];
    let rhs_shape = [depth, columns];
    let out_shape = [rows, columns];
    let lhs_strides = contiguous(&lhs_shape);
    let rhs_strides = contiguous(&rhs_shape);
    let out_strides = contiguous(&out_shape);
    // Safety: the left matrix holds `rows x depth` values contiguously, as just asserted.
    let lhs_tensor = unsafe { tensor::<R>(lhs, &lhs_shape, &lhs_strides) };
    // Safety: the right matrix holds `depth x columns` values contiguously, as just asserted.
    let rhs_tensor = unsafe { tensor::<R>(rhs, &rhs_shape, &rhs_strides) };
    // Safety: the output holds `rows x columns` values contiguously, as just asserted.
    let out_tensor = unsafe { tensor::<R>(output, &out_shape, &out_strides) };

    let lhs_binding = InputBinding::new(lhs_tensor, dtype);
    let rhs_binding = InputBinding::new(rhs_tensor, dtype);
    let globals = uniform_globals(dtype);
    let mut dtypes = MatmulElems::from_globals(&globals);
    let offers_matrix = offers_cmma_16(client);
    let strategy = product_strategy(precision, offers_matrix);
    let launched = cubek_matmul::launch::launch_ref(
        &strategy,
        client,
        lhs_binding,
        rhs_binding,
        out_tensor,
        &mut dtypes,
    );

    launched.map_err(matmul_failed)
}

/// Scatters each sample of the `width` by `height` NHWC maps in `input` into a 2x2 block of `output`.
///
/// The output sample at a block's row and column takes input channels from `(row * 2 + column) * channels`,
/// so `output` is twice as wide and twice as tall, with a quarter of the channels.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn scatter<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: &Handle,
    width: u32,
    height: u32,
    channels: u32,
    precision: Precision,
    output: &Handle,
) {
    let count = (width * height * 4 * channels) as usize;
    check_values(input, count, precision, "input");
    check_values(output, count, precision, "output");

    let cubes = dispatch.cube_count(count);
    let unit = dispatch.cube_dim();

    // Safety: the input and output each hold the maps' values, as just asserted, and every read lands on the
    // input sample under the output's 2x2 block.
    unsafe {
        let input_array = ArrayArg::from_raw_parts(input.clone(), count);
        let output_array = ArrayArg::from_raw_parts(output.clone(), count);

        match precision {
            Precision::F16 => scatter_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                input_array,
                output_array,
                width,
                channels,
            ),
            Precision::F32 => scatter_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                input_array,
                output_array,
                width,
                channels,
            ),
        }
    }
}

/// Adds the first `count` values of `addend` to those of `values`, summing in f32.
pub fn add_in_place<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    values: &Handle,
    addend: &Handle,
    count: usize,
    precision: Precision,
) {
    check_values(values, count, precision, "values");
    check_values(addend, count, precision, "addend");

    let cubes = dispatch.cube_count(count);
    let unit = dispatch.cube_dim();

    // Safety: the values and the addend each hold `count` values, as just asserted.
    unsafe {
        let values_array = ArrayArg::from_raw_parts(values.clone(), count);
        let addend_array = ArrayArg::from_raw_parts(addend.clone(), count);

        match precision {
            Precision::F16 => {
                add_kernel::launch_unchecked::<half::f16, R>(client, cubes, unit, values_array, addend_array)
            },
            Precision::F32 => {
                add_kernel::launch_unchecked::<f32, R>(client, cubes, unit, values_array, addend_array)
            },
        }
    }
}

/// Writes `core` of the padded tail's maps of the placement's region in `scratch` into `outputs`.
///
/// Channel `c` of each core sample goes to plane `c`, clamped to `0..=1` and scaled by `peak`. A core sample
/// in the padding lies outside the planes and is not written.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn write_out<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    scratch: &Handle,
    placement: Placement,
    core: Region,
    channels: u32,
    peak: f32,
    outputs: &[DevicePlane<'_>],
) {
    let region = placement.region;
    assert!(region.contains(&core), "the core lies outside the region");

    let (plane_width, plane_height) = check_planes(outputs);
    assert_eq!(
        outputs.len(),
        channels as usize,
        "the outputs do not hold one plane a channel"
    );

    let scratch_count = region.samples() as usize * PADDED_CHANNELS;
    check_values(scratch, scratch_count, Precision::F32, "scratch");

    let core_samples = core.samples();
    let units = (core_samples * channels) as usize;
    let scratch_channels = PADDED_CHANNELS as u32;
    let [first, second, third] = three_planes(outputs);
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let third_array = third.array::<R>();
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the scratch holds the region times 16 values, as just asserted, every read stays inside the
    // region since the core lies inside it, and every write lands inside a plane, which the kernel checks.
    unsafe {
        let scratch_array = ArrayArg::from_raw_parts(scratch.clone(), scratch_count);

        write_out_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            scratch_array,
            first_array,
            second_array,
            third_array,
            plane_width,
            plane_height,
            region.x,
            region.y,
            region.width,
            placement.left,
            placement.top,
            core.x,
            core.y,
            core.width,
            core_samples,
            channels,
            scratch_channels,
            peak,
        );
    }
}

/// Runs `layer`, a 3x3 convolution from the features to at most 16 channels, over the NHWC maps of `region`
/// in `input`, into `scratch` as NHWC maps of 16 f32 channels.
///
/// The convolution paths run it on cooperative-matrix tiles. [BodyPath::Im2col](crate::cnn::BodyPath) runs a
/// direct kernel that needs none.
pub fn tail<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &PaddedLastLayer,
    path: BodyPath,
    input: &Handle,
    region: Region,
    scratch: &Handle,
) -> Result<(), CnnError> {
    match path {
        BodyPath::Im2col => direct_tail(client, dispatch, layer, input, region, scratch),
        BodyPath::Convolution | BodyPath::UnfusedConvolution => {
            run_padded_layer(client, layer, input, region, scratch, true)?;
        },
    }

    Ok(())
}

/// Runs `layer` over the NHWC maps of `region` in `input` one sample and channel a unit, into `scratch` as
/// NHWC maps of 16 f32 channels.
fn direct_tail<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    layer: &PaddedLastLayer,
    input: &Handle,
    region: Region,
    scratch: &Handle,
) {
    let samples = region.samples() as usize;
    let in_channels = layer.in_channels as usize;
    let input_count = samples * in_channels;
    let weight_count = PADDED_CHANNELS * 9 * in_channels;
    let units = samples * PADDED_CHANNELS;
    check_values(input, input_count, layer.precision, "input");
    check_values(&layer.weights, weight_count, layer.precision, "tail's weights");
    check_values(&layer.bias, PADDED_CHANNELS, Precision::F32, "tail's bias");
    check_values(scratch, units, Precision::F32, "scratch");

    let scratch_channels = PADDED_CHANNELS as u32;
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the input holds the region times `in` values, the weights `16 * 9 * in`, the bias 16 and the
    // scratch the region times 16, as just asserted, and every read of the input stays inside the region.
    unsafe {
        let input_array = ArrayArg::from_raw_parts(input.clone(), input_count);
        let weights = ArrayArg::from_raw_parts(layer.weights.clone(), weight_count);
        let bias = ArrayArg::from_raw_parts(layer.bias.clone(), PADDED_CHANNELS);
        let scratch_array = ArrayArg::from_raw_parts(scratch.clone(), units);

        match layer.precision {
            Precision::F16 => direct_tail_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                scratch_array,
                region.width,
                region.height,
                layer.in_channels,
                layer.out_channels,
                scratch_channels,
            ),
            Precision::F32 => direct_tail_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                scratch_array,
                region.width,
                region.height,
                layer.in_channels,
                layer.out_channels,
                scratch_channels,
            ),
        }
    }
}
