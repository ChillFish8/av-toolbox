// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::artcnn::{Network, Precision};
use crate::{DevicePlane, Dispatch};

/// A rectangle of a plane, in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Region {
    pub fn samples(&self) -> u32 {
        self.width * self.height
    }

    fn contains(&self, other: &Region) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.x + other.width <= self.x + self.width
            && other.y + other.height <= self.y + self.height
    }
}

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
) {
    if ABSOLUTE_POS >= (core_samples * 4) as usize {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let channel = position % 4;
    let sample = position / 4;
    let plane_column = core_x + sample % core_width;
    let plane_row = core_y + sample / core_width;
    let column = (plane_column - region_x) as i32;
    let row = (plane_row - region_y) as i32;
    let mut sum = f32::cast_from(bias[channel as usize]);

    for tap in 0..9u32 {
        let source_column = column + (tap % 3) as i32 - 1;
        let source_row = row + (tap / 3) as i32 - 1;
        let inside = source_column >= 0
            && source_row >= 0
            && source_column < region_width as i32
            && source_row < region_height as i32;

        if inside {
            let source = (source_row as u32 * region_width + source_column as u32) * features;
            let tap_start = (channel * 9 + tap) * features;

            for feature in 0..features {
                let weight = f32::cast_from(weights[(tap_start + feature) as usize]);
                let value = f32::cast_from(input[(source + feature) as usize]);
                sum += weight * value;
            }
        }
    }

    let clamped = f32::clamp(sum, 0.0, 1.0);
    let rounded = f32::floor(clamped * peak + 0.5);
    let out_row = plane_row * 2 + channel / 2;
    let out_column = plane_column * 2 + channel % 2;
    output[(out_row * output_width + out_column) as usize] = rounded;
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

    let layer = network.layer(0);
    debug_assert_eq!(layer.in_channels, 1);

    let features = network.features();
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

    // Safety: the weights hold `F * 9` values and the bias `F`, the output holds the region times `F`, and every
    // read of the input is inside the region, which lies inside the plane.
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
/// Each sample of `core` becomes a 2x2 block of the output, clamped to `0..=1`, scaled by `peak` and rounded.
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
    output: DevicePlane<'_>,
) {
    assert!(region.contains(&core), "the core lies outside the region");
    assert!(
        (core.x + core.width) * 2 <= output.width(),
        "the doubled core runs past the output's right edge"
    );
    assert!(
        (core.y + core.height) * 2 <= output.height(),
        "the doubled core runs past the output's bottom edge"
    );

    let layer = network.last_layer();
    debug_assert_eq!(layer.out_channels, 4);

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

    // Safety: the weights hold `4 * 9 * F` values and the bias 4, the input holds the region times `F`, every read
    // stays inside the region, and every write lands in the doubled core, which lies inside the output plane.
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
            ),
        }
    }
}
