// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::artcnn::Network;
use crate::cnn::{CnnError, PADDED_CHANNELS, Precision, Region, add_taps, run_padded_layer};
use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn first_layer_kernel<F: Float>(
    input: &Array<f32>,
    weights: &Array<F>,
    bias: &Array<F>,
    output: &mut Array<F>,
    plane_width: u32,
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    features: u32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let channel = position % features;
    let sample = position / features;
    let column = (sample % region_width) as i32;
    let row = (sample / region_width) as i32;
    let mut sum = f32::cast_from(bias[channel as usize]);

    for tap in 0..9u32 {
        let source_column = column + (tap % 3) as i32 - 1;
        let source_row = row + (tap / 3) as i32 - 1;
        let inside = source_column >= 0
            && source_row >= 0
            && source_column < region_width as i32
            && source_row < region_height as i32;

        if inside {
            let plane_column = region_x + source_column as u32;
            let plane_row = region_y + source_row as u32;
            let value = input[(plane_row * plane_width + plane_column) as usize] / peak;
            let weight = f32::cast_from(weights[(channel * 9 + tap) as usize]);
            sum += weight * value;
        }
    }

    output[ABSOLUTE_POS] = F::cast_from(sum);
}

/// The plane sample and output channel of one unit of a core's doubled output, four units a sample.
#[derive(CubeType)]
struct CoreUnit {
    plane_column: u32,
    plane_row: u32,
    channel: u32,
}

#[cube]
impl CoreUnit {
    fn new(position: u32, core_x: u32, core_y: u32, core_width: u32) -> CoreUnit {
        let sample = position / 4;

        CoreUnit {
            plane_column: core_x + sample % core_width,
            plane_row: core_y + sample / core_width,
            channel: position % 4,
        }
    }
}

/// Clamps `value` to `0..=1`, scales it by `peak` and writes it to `unit`'s place in the doubled output,
/// rounded to a whole code when `rounded` is set.
#[cube]
fn write_doubled(
    output: &mut Array<f32>,
    output_width: u32,
    unit: &CoreUnit,
    value: f32,
    peak: f32,
    #[comptime] rounded: bool,
) {
    let clamped = f32::clamp(value, 0.0, 1.0);
    let mut scaled = clamped * peak;

    if rounded {
        scaled = f32::floor(scaled + 0.5);
    }

    let out_row = unit.plane_row * 2 + unit.channel / 2;
    let out_column = unit.plane_column * 2 + unit.channel % 2;
    output[(out_row * output_width + out_column) as usize] = scaled;
}

#[cube(launch_unchecked)]
fn last_layer_kernel<F: Float>(
    input: &Array<F>,
    weights: &Array<F>,
    bias: &Array<F>,
    output: &mut Array<f32>,
    output_width: u32,
    region_x: u32,
    region_y: u32,
    region_width: u32,
    region_height: u32,
    core_x: u32,
    core_y: u32,
    core_width: u32,
    core_samples: u32,
    features: u32,
    peak: f32,
    #[comptime] rounded: bool,
) {
    if ABSOLUTE_POS >= (core_samples * 4) as usize {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let unit = CoreUnit::new(position, core_x, core_y, core_width);
    let channel = unit.channel;
    let column = (unit.plane_column - region_x) as i32;
    let row = (unit.plane_row - region_y) as i32;
    let bias_value = f32::cast_from(bias[channel as usize]);
    let sum = add_taps::<F>(
        bias_value,
        input,
        weights,
        column,
        row,
        region_width,
        region_height,
        channel,
        features,
    );

    write_doubled(output, output_width, &unit, sum, peak, rounded);
}

#[cube(launch_unchecked)]
fn shuffle_kernel(
    scratch: &Array<f32>,
    output: &mut Array<f32>,
    output_width: u32,
    region_x: u32,
    region_y: u32,
    region_width: u32,
    core_x: u32,
    core_y: u32,
    core_width: u32,
    core_samples: u32,
    channels: u32,
    peak: f32,
    #[comptime] rounded: bool,
) {
    if ABSOLUTE_POS >= (core_samples * 4) as usize {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let unit = CoreUnit::new(position, core_x, core_y, core_width);
    let column = unit.plane_column - region_x;
    let row = unit.plane_row - region_y;
    let source = (row * region_width + column) * channels + unit.channel;
    let value = scratch[source as usize];

    write_doubled(output, output_width, &unit, value, peak, rounded);
}

/// Runs the network's first layer over `region` of `input`, divided by `peak`, into NHWC maps in `output`.
///
/// The region's edge is padded with zero.
pub fn first_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    input: DevicePlane<'_>,
    peak: f32,
    region: Region,
    output: &Handle,
) {
    assert!(
        region.x + region.width <= input.width(),
        "the region runs past the plane's right edge"
    );
    assert!(
        region.y + region.height <= input.height(),
        "the region runs past the plane's bottom edge"
    );

    let layer = network.first_layer();
    let features = network.features();
    assert!(
        layer.in_channels == 1 && layer.out_channels == features,
        "the first layer does not map one channel to the features"
    );

    let plane_width = input.width();
    let units = (region.samples() * features) as usize;
    let output_bytes = (units * network.precision().bytes()) as u64;
    assert!(
        output.size_in_used() >= output_bytes,
        "the output is smaller than the region's maps"
    );

    let input_array = input.array::<R>();
    let weight_count = (features * 9) as usize;
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the weights hold `F * 9` values and the bias `F`, the output holds the region times `F`, and
    // every read of the input is inside the region, which lies inside the plane.
    unsafe {
        let weights = ArrayArg::from_raw_parts(layer.weights.clone(), weight_count);
        let bias = ArrayArg::from_raw_parts(layer.bias.clone(), features as usize);
        let output_array = ArrayArg::from_raw_parts(output.clone(), units);

        match network.precision() {
            Precision::F16 => first_layer_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                output_array,
                plane_width,
                region.x,
                region.y,
                region.width,
                region.height,
                features,
                peak,
            ),
            Precision::F32 => first_layer_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                output_array,
                plane_width,
                region.x,
                region.y,
                region.width,
                region.height,
                features,
                peak,
            ),
        }
    }
}

/// Runs the network's last layer over the NHWC maps of `region` and writes `core` into the doubled `output`.
///
/// Each sample of `core` becomes a 2x2 block of the output, clamped to `0..=1` and scaled by `peak`. It is
/// rounded to whole codes when `rounded` is set.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn last_layer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    network: &Network,
    input: &Handle,
    region: Region,
    core: Region,
    peak: f32,
    rounded: bool,
    output: DevicePlane<'_>,
) {
    check_core(region, core, output);
    check_last_layer(network);

    let layer = network.last_layer();
    let features = network.features();
    let output_width = output.width();
    let core_samples = core.samples();
    let units = (core_samples * 4) as usize;
    let input_count = (region.samples() * features) as usize;
    let input_bytes = (input_count * network.precision().bytes()) as u64;
    assert!(
        input.size_in_used() >= input_bytes,
        "the input is smaller than the region's maps"
    );

    let weight_count = (4 * 9 * features) as usize;
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the weights hold `4 * 9 * F` values and the bias 4, the input holds the region times `F`, every
    // read stays inside the region, and every write lands in the doubled core, which lies inside the output
    // plane.
    unsafe {
        let input_array = ArrayArg::from_raw_parts(input.clone(), input_count);
        let weights = ArrayArg::from_raw_parts(layer.weights.clone(), weight_count);
        let bias = ArrayArg::from_raw_parts(layer.bias.clone(), 4);

        match network.precision() {
            Precision::F16 => last_layer_kernel::launch_unchecked::<half::f16, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                output_array,
                output_width,
                region.x,
                region.y,
                region.width,
                region.height,
                core.x,
                core.y,
                core.width,
                core_samples,
                features,
                peak,
                rounded,
            ),
            Precision::F32 => last_layer_kernel::launch_unchecked::<f32, R>(
                client,
                cubes,
                unit,
                input_array,
                weights,
                bias,
                output_array,
                output_width,
                region.x,
                region.y,
                region.width,
                region.height,
                core.x,
                core.y,
                core.width,
                core_samples,
                features,
                peak,
                rounded,
            ),
        }
    }
}

/// Asserts that the network's last layer maps the features to four channels.
fn check_last_layer(network: &Network) {
    let layer = network.last_layer();
    let features = network.features();
    assert!(
        layer.in_channels == features && layer.out_channels == 4,
        "the last layer does not map the features to four channels"
    );
}

/// Asserts that `core` lies inside `region`, and that the doubled core lies inside `output`.
fn check_core(region: Region, core: Region, output: DevicePlane<'_>) {
    assert!(region.contains(&core), "the core lies outside the region");
    assert!(
        (core.x + core.width) * 2 <= output.width(),
        "the doubled core runs past the output's right edge"
    );
    assert!(
        (core.y + core.height) * 2 <= output.height(),
        "the doubled core runs past the output's bottom edge"
    );
}

/// Runs the network's last layer over the NHWC maps of `region` in `input` on cooperative-matrix tiles,
/// into `scratch` as NHWC maps of 16 f32 channels. Gives whether the forced blueprint ran.
///
/// The first four channels are the last layer's, and the other twelve are zero.
pub fn padded_last_layer<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    input: &Handle,
    region: Region,
    scratch: &Handle,
) -> Result<bool, CnnError> {
    check_last_layer(network);

    let padded = network.padded_last_layer();

    run_padded_layer(client, padded, input, region, scratch, true)
}

/// The same as [padded_last_layer](crate::artcnn::padded_last_layer), keeping the inferred blueprint unless
/// `allow_forced` is set.
#[cfg(any(test, feature = "test-support"))]
pub fn padded_last_layer_with_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    network: &Network,
    input: &Handle,
    region: Region,
    scratch: &Handle,
    allow_forced: bool,
) -> Result<bool, CnnError> {
    if allow_forced {
        return padded_last_layer(client, network, input, region, scratch);
    }

    check_last_layer(network);

    let padded = network.padded_last_layer();

    run_padded_layer(client, padded, input, region, scratch, false)
}

/// Writes `core` of the padded last layer's maps of `region` in `scratch` into the doubled `output`.
///
/// Each sample of `core` becomes a 2x2 block of the output from its first four channels, clamped to `0..=1`
/// and scaled by `peak`. It is rounded to whole codes when `rounded` is set.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
pub fn shuffle<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    scratch: &Handle,
    region: Region,
    core: Region,
    peak: f32,
    rounded: bool,
    output: DevicePlane<'_>,
) {
    check_core(region, core, output);

    let scratch_count = region.samples() as usize * PADDED_CHANNELS;
    let scratch_bytes = (scratch_count * size_of::<f32>()) as u64;
    assert!(
        scratch.size_in_used() >= scratch_bytes,
        "the scratch is smaller than the region's padded maps"
    );

    let output_width = output.width();
    let core_samples = core.samples();
    let units = (core_samples * 4) as usize;
    let channels = PADDED_CHANNELS as u32;
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();

    // Safety: the scratch holds the region times 16 values, every read stays inside the region, and every
    // write lands in the doubled core, which lies inside the output plane.
    unsafe {
        let scratch_array = ArrayArg::from_raw_parts(scratch.clone(), scratch_count);

        shuffle_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            scratch_array,
            output_array,
            output_width,
            region.x,
            region.y,
            region.width,
            core.x,
            core.y,
            core.width,
            core_samples,
            channels,
            peak,
            rounded,
        );
    }
}
