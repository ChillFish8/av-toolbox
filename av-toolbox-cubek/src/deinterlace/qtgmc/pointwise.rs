// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{assert_bits, havsfunc_scale};
use crate::resize::Taps;
use crate::{DevicePlane, Dispatch, assert_plane};

/// How many samples each side of a sample fmtc's gaussian reaches at its default `taps` of 4.
const GAUSS_REACH: f64 = 4.0;

/// The `sw` and `sh` slack havsfunc's `Resize` adds so that fmtc resamples at an identity scale.
const GAUSS_SLACK: f64 = 1e-6;

/// fmtc's `gauss` kernel with `a1 = p` at an identity scale, as havsfunc's `Resize(..., kernel='gauss')`
/// runs it.
///
/// A sample `x` away is weighed by `2^(-p / 10 * x^2)` with `p` limited to `1..=100`, and zero beyond
/// `x` of 4 (`ContFirGauss.cpp`, constructor and `do_get_val`, and the default `taps` of 4 in
/// `Resample_vs.cpp`).
///
/// Each output's weights are scaled to sum to 1. Taps past the plane's edge count toward the edge sample, so
/// apply these with a clamped edge.
///
/// havsfunc's `sw = w + 1e-6` makes `Scaler.cpp` read 9 taps, with the kernel stretched by
/// `1 + 1e-6 / length`. That moves where each output sits by a hair, which only decides whether the 9th
/// sample, exactly 4 away, falls inside the kernel. It does for the first 4 outputs and not for the rest, so
/// away from the start the kernel reads from 3 samples before to 4 after.
pub fn fmtc_gauss_taps(length: u32, p: f32) -> Taps {
    let sharpness = f64::from(p).clamp(1.0, 100.0) / 10.0;
    let step = (f64::from(length) + GAUSS_SLACK) / f64::from(length);
    let support = GAUSS_REACH * step;
    let count = (support * 2.0).ceil() as i32;

    let mut starts = Vec::with_capacity(length as usize);
    let mut weights = Vec::with_capacity((length as i32 * count) as usize);
    for index in 0..length {
        let position = step * (f64::from(index) + 0.5) - 0.5;
        let start = (position + support).floor() as i32 - count + 1;

        let raw: Vec<f64> = (0..count)
            .map(|tap| {
                let distance = (position - f64::from(start + tap)) / step;
                if distance.abs() <= GAUSS_REACH {
                    2.0f64.powf(-sharpness * distance * distance)
                } else {
                    0.0
                }
            })
            .collect();
        let total: f64 = raw.iter().sum();

        starts.push(start);
        weights.extend(raw.iter().map(|weight| (weight / total) as f32));
    }

    Taps {
        starts,
        weights,
        count: count as u32,
    }
}

/// QTGMC's `sharpAdj` with temporal sharpness limiting and SMode 2.
///
/// `sharpness * 2 * (0.2 + tr1 * 0.15 + tr2 * 0.25)`.
pub fn sharpness_strength(sharpness: f32, tr1: u32, tr2: u32) -> f32 {
    let smoothing = 0.2 + tr1 as f32 * 0.15 + tr2 as f32 * 0.25;

    sharpness * 2.0 * smoothing
}

#[cube(launch_unchecked)]
fn search_kernel(
    blur: &[f32],
    repaired: &[f32],
    bob: &[f32],
    output: &mut [f32],
    tolerance: f32,
    near: f32,
    far: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let centre = repaired[ABSOLUTE_POS];
    let limited = f32::clamp(bob[ABSOLUTE_POS], centre - tolerance, centre + tolerance);
    let tweaked = f32::clamp(limited, 0.0, peak);

    let smoothed = blur[ABSOLUTE_POS];
    let mut result = (smoothed * 51.0 + tweaked * 49.0) / 100.0;
    if smoothed + far < tweaked {
        result = smoothed + near;
    }
    if smoothed - far > tweaked {
        result = smoothed - near;
    }

    output[ABSOLUTE_POS] = f32::clamp(result, 0.0, peak);
}

#[cube(launch_unchecked)]
fn resharpen_kernel(input: &[f32], blurred: &[f32], output: &mut [f32], strength: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let sample = input[ABSOLUTE_POS];
    let sharpened = sample + (sample - blurred[ABSOLUTE_POS]) * strength;

    output[ABSOLUTE_POS] = f32::clamp(sharpened, 0.0, peak);
}

#[cube(launch_unchecked)]
fn midpoint_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = (position / width) as i32;

    let last = height as i32 - 1;
    let above = i32::clamp(row - 1, 0, last) as u32;
    let below = i32::clamp(row + 1, 0, last) as u32;
    let top = input[(above * width + column) as usize];
    let centre = input[ABSOLUTE_POS];
    let bottom = input[(below * width + column) as usize];

    let largest = f32::max(centre, f32::max(top, bottom));
    let smallest = f32::min(centre, f32::min(top, bottom));

    output[ABSOLUTE_POS] = f32::floor((largest + smallest) / 2.0 + 0.5);
}

#[cube(launch_unchecked)]
fn subtract_kernel(input: &[f32], blurred: &[f32], output: &mut [f32], peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::clamp(input[ABSOLUTE_POS] - blurred[ABSOLUTE_POS], 0.0, peak);
}

#[cube(launch_unchecked)]
fn clamp_kernel(input: &[f32], upper: &[f32], lower: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let capped = f32::min(input[ABSOLUTE_POS], upper[ABSOLUTE_POS]);

    output[ABSOLUTE_POS] = f32::max(capped, lower[ABSOLUTE_POS]);
}

#[cube(launch_unchecked)]
fn merge_kernel(first: &[f32], second: &[f32], output: &mut [f32], weight: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = first[ABSOLUTE_POS] * (1.0 - weight) + second[ABSOLUTE_POS] * weight;
}

/// The code values `scale(3)`, `scale(7)` and `scale(2)` for `bits` deep samples.
pub(crate) fn search_levels(bits: u32) -> (f32, f32, f32) {
    let peak = ((1u32 << bits) - 1) as f32;

    (
        havsfunc_scale(3.0, peak),
        havsfunc_scale(7.0, peak),
        havsfunc_scale(2.0, peak),
    )
}

/// The search clip's final expressions, as QTGMC builds the clip motion is searched on.
///
/// `blur` is the smoothed clip already merged with `repaired`. It is nudged toward `bob` where they are
/// far apart and averaged with it where they are close.
pub fn search_clip(
    client: &Client,
    dispatch: &impl Dispatch,
    blur: DevicePlane<'_>,
    repaired: DevicePlane<'_>,
    bob: DevicePlane<'_>,
    output: DevicePlane<'_>,
    bits: u32,
) {
    assert_bits(bits);

    let (width, height) = (output.width(), output.height());
    assert_plane(&blur, width, height, "blur");
    assert_plane(&repaired, width, height, "repaired");
    assert_plane(&bob, width, height, "bob");
    assert_plane(&output, width, height, "output");

    let peak = ((1u32 << bits) - 1) as f32;
    let (tolerance, far, near) = search_levels(bits);
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let blur_array = blur.array();
    let repaired_array = repaired.array();
    let bob_array = bob.array();
    let output_array = output.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        search_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            blur_array,
            repaired_array,
            bob_array,
            output_array,
            tolerance,
            near,
            far,
            peak,
        );
    }
}

/// `x + (x − blurred) × strength`, clamped to the code values of `bits` deep samples.
pub fn resharpen(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    blurred: DevicePlane<'_>,
    output: DevicePlane<'_>,
    strength: f32,
    bits: u32,
) {
    assert_bits(bits);

    let (width, height) = (output.width(), output.height());
    assert_plane(&input, width, height, "input");
    assert_plane(&blurred, width, height, "blurred");
    assert_plane(&output, width, height, "output");

    let peak = ((1u32 << bits) - 1) as f32;
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array();
    let blurred_array = blurred.array();
    let output_array = output.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        resharpen_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            blurred_array,
            output_array,
            strength,
            peak,
        );
    }
}

/// `(vmax(x) + vmin(x)) / 2` rounded half up to a whole code value, the plane SMode 2 blurs before
/// resharpening.
///
/// `vmax` and `vmin` are the largest and smallest of a sample and the samples above and below it. The
/// rounding is how `std.Merge` stores the midpoint, and sharpening amplifies it.
pub fn vertical_midpoint(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let (width, height) = (output.width(), output.height());
    assert_plane(&input, width, height, "input");
    assert_plane(&output, width, height, "output");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array();
    let output_array = output.array();

    // Safety: both planes hold `width` by `height` f32 samples, and every row read is clamped inside
    // the plane.
    unsafe {
        midpoint_kernel::launch_unchecked(client, cubes, unit, input_array, output_array, width, height);
    }
}

/// `x − blurred`, the back-blend's final subtraction, clamped to the code values of `bits` deep samples.
pub fn subtract(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    blurred: DevicePlane<'_>,
    output: DevicePlane<'_>,
    bits: u32,
) {
    assert_bits(bits);

    let (width, height) = (output.width(), output.height());
    assert_plane(&input, width, height, "input");
    assert_plane(&blurred, width, height, "blurred");
    assert_plane(&output, width, height, "output");

    let peak = ((1u32 << bits) - 1) as f32;
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array();
    let blurred_array = blurred.array();
    let output_array = output.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        subtract_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            blurred_array,
            output_array,
            peak,
        );
    }
}

/// Clamps `input` between `lower` and `upper`, with `lower` winning where they cross.
pub fn clamp_between(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    upper: DevicePlane<'_>,
    lower: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let (width, height) = (output.width(), output.height());
    assert_plane(&input, width, height, "input");
    assert_plane(&upper, width, height, "upper");
    assert_plane(&lower, width, height, "lower");
    assert_plane(&output, width, height, "output");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array();
    let upper_array = upper.array();
    let lower_array = lower.array();
    let output_array = output.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        clamp_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            upper_array,
            lower_array,
            output_array,
        );
    }
}

/// `first × (1 − weight) + second × weight`.
pub fn merge(
    client: &Client,
    dispatch: &impl Dispatch,
    first: DevicePlane<'_>,
    second: DevicePlane<'_>,
    output: DevicePlane<'_>,
    weight: f32,
) {
    let (width, height) = (output.width(), output.height());
    assert_plane(&first, width, height, "first");
    assert_plane(&second, width, height, "second");
    assert_plane(&output, width, height, "output");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let first_array = first.array();
    let second_array = second.array();
    let output_array = output.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        merge_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            first_array,
            second_array,
            output_array,
            weight,
        );
    }
}
