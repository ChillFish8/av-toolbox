// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::edges::{prewitt_across, prewitt_down, tap};
use crate::plane::assert_plane;
use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn binarize_prewitt_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    threshold: u32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;

    let top_left = tap(input, width, height, x - 1, y - 1);
    let top = tap(input, width, height, x, y - 1);
    let top_right = tap(input, width, height, x + 1, y - 1);
    let left = tap(input, width, height, x - 1, y);
    let right = tap(input, width, height, x + 1, y);
    let bottom_left = tap(input, width, height, x - 1, y + 1);
    let bottom = tap(input, width, height, x, y + 1);
    let bottom_right = tap(input, width, height, x + 1, y + 1);

    let across = prewitt_across(top_left, left, bottom_left, top_right, right, bottom_right);
    let down = prewitt_down(top_left, top, top_right, bottom_left, bottom, bottom_right);
    let gx = i32::abs(i32::cast_from(across));
    let gy = i32::abs(i32::cast_from(down));

    // A gradient of 65536 or more squares past every threshold on its own. Below it each square fits
    // a u32, and a sum that wraps has passed 2^32, which is above every threshold too.
    let mut set = gx >= 65536 || gy >= 65536;
    let gx_capped = u32::min(u32::cast_from(gx), 65535u32);
    let gy_capped = u32::min(u32::cast_from(gy), 65535u32);
    let gx_squared = gx_capped * gx_capped;
    let gy_squared = gy_capped * gy_capped;
    let sum = gx_squared + gy_squared;
    if sum < gx_squared || sum >= threshold {
        set = true;
    }

    let mut value = 0.0f32;
    if set {
        value = peak;
    }

    output[ABSOLUTE_POS] = value;
}

/// The first of the two input samples a doubled sample at `doubled` reads, before any clamp.
///
/// The second is the sample after it.
#[cube]
fn tap_start(doubled: u32) -> i32 {
    let half = doubled / 2u32;
    let odd = doubled % 2u32;

    (half + odd) as i32 - 1
}

/// `index` clamped inside a row of `length` samples.
#[cube]
fn clamp_tap(index: i32, length: u32) -> u32 {
    let last = length as i32 - 1;

    i32::clamp(index, 0, last) as u32
}

/// The weight of the first tap.
///
/// An even doubled sample lies nearer its second tap, so the first weighs 1/4. An odd one lies nearer
/// its first, which weighs 3/4.
#[cube]
fn first_weight(doubled: u32) -> f32 {
    let odd = doubled % 2u32;

    let mut weight = 0.25f32;
    if odd == 1u32 {
        weight = 0.75;
    }

    weight
}

#[cube(launch_unchecked)]
fn double_mclip_kernel(mask: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32, mask_peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let out_width = width * 2u32;
    let column = position % out_width;
    let row = position / out_width;

    let column_start = tap_start(column);
    let row_start = tap_start(row);
    let left = clamp_tap(column_start, width);
    let right = clamp_tap(column_start + 1, width);
    let top = clamp_tap(row_start, height);
    let bottom = clamp_tap(row_start + 1, height);
    let left_weight = first_weight(column);
    let right_weight = 1.0 - left_weight;
    let top_weight = first_weight(row);
    let bottom_weight = 1.0 - top_weight;

    let top_left = mask[(top * width + left) as usize];
    let top_right = mask[(top * width + right) as usize];
    let bottom_left = mask[(bottom * width + left) as usize];
    let bottom_right = mask[(bottom * width + right) as usize];
    let upper = left_weight * top_left + right_weight * top_right;
    let lower = left_weight * bottom_left + right_weight * bottom_right;
    let value = top_weight * upper + bottom_weight * lower;

    let mut clipped = 0.0f32;
    if 510.0 * value > mask_peak {
        clipped = 255.0;
    }

    output[ABSOLUTE_POS] = clipped;
}

#[cube(launch_unchecked)]
fn pscale_blend_kernel(
    aa: &Array<f32>,
    luma: &Array<f32>,
    down: &Array<f32>,
    output: &mut Array<f32>,
    keep: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let difference = luma[ABSOLUTE_POS] - down[ABSOLUTE_POS];
    let kept = keep * difference;

    output[ABSOLUTE_POS] = aa[ABSOLUTE_POS] + kept;
}

/// Writes `peak` where the Prewitt gradients of `input` satisfy `gx^2 + gy^2 >= threshold`, and 0
/// elsewhere.
///
/// `input` holds integer code values. The compare is exact in integers, and the window reflects at the
/// plane's edge as [edges](crate::edges::edges) does.
pub fn binarize_prewitt<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    threshold: u32,
    peak: f32,
) {
    let width = input.width();
    let height = input.height();

    assert_plane(&output, width, height, "output");

    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the asserted sizes make both planes hold `samples` f32 values, and every read is
    // reflected inside the plane.
    unsafe {
        binarize_prewitt_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            width,
            height,
            threshold,
            peak,
        );
    }
}

/// Doubles `mask` into the 8-bit clip EEDI3 searches within, holding 255 where a sample is set and 0
/// elsewhere.
///
/// Each doubled sample is the bilinear mix of its two nearest mask samples on each axis, weighted 3/4
/// and 1/4, repeating the edge sample past the plane's side. It is set when `510 * value > mask_peak`,
/// so a value of exactly half an 8-bit code is not set. `output` is twice the mask's size.
pub fn double_mclip<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    mask: DevicePlane<'_>,
    output: DevicePlane<'_>,
    mask_peak: f32,
) {
    let width = mask.width();
    let height = mask.height();

    assert_plane(&output, 2 * width, 2 * height, "output");

    let samples = output.samples();
    let mask_array = mask.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the mask holds `width * height` f32 values, the asserted output `samples`, and every
    // read is clamped inside the mask.
    unsafe {
        double_mclip_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            mask_array,
            output_array,
            width,
            height,
            mask_peak,
        );
    }
}

/// Writes `aa + keep * (luma - down)`, where `keep` is `1 - pscale`.
pub fn pscale_blend<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    aa: DevicePlane<'_>,
    luma: DevicePlane<'_>,
    down: DevicePlane<'_>,
    output: DevicePlane<'_>,
    keep: f32,
) {
    let width = output.width();
    let height = output.height();

    assert_plane(&aa, width, height, "aa");
    assert_plane(&luma, width, height, "luma");
    assert_plane(&down, width, height, "down");

    let samples = output.samples();
    let aa_array = aa.array::<R>();
    let luma_array = luma.array::<R>();
    let down_array = down.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the asserted sizes make all four planes hold `samples` f32 values.
    unsafe {
        pscale_blend_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            aa_array,
            luma_array,
            down_array,
            output_array,
            keep,
        );
    }
}
