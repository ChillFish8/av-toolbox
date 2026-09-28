// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::recursive::{RecursiveGaussian, RecursivePlanes, recursive_gaussian};
use crate::histogram::{PercentilePlanes, percentile};
use crate::reduce::{ReducePlanes, extremes};
use crate::{DevicePlane, Dispatch};

/// The multi-scale retinex's settings.
#[derive(Debug, Clone, PartialEq)]
pub struct RetinexSettings {
    /// The deviations the input is blurred at.
    pub sigmas: Vec<f32>,
    /// The fraction of samples clipped from the dark end of the response, not a percent.
    pub lower: f32,
    /// The fraction of samples clipped from the bright end of the response, not a percent.
    pub upper: f32,
    /// The value the stretched response is scaled up to.
    pub scale: f32,
}

/// Every plane the retinex works through. Plane-sized planes are f32 and the input's size.
#[derive(Debug, Clone, Copy)]
pub struct RetinexPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub normalised: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub msr: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    /// Each holds `reduce::scratch_len(samples)` f32 values.
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    /// Holds 2 f32 values.
    pub range: DevicePlane<'a>,
    /// Holds `histogram::BINS` u32 values.
    pub bins: DevicePlane<'a>,
    /// Each holds 1 f32 value.
    pub low: DevicePlane<'a>,
    pub high: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn normalise(input: &Array<f32>, range: &Array<f32>, normalised: &mut Array<f32>) {
    if ABSOLUTE_POS >= normalised.len() {
        terminate!();
    }

    let low = range[0];
    let high = range[1];

    let mut value = 0.0f32;
    if high > low {
        value = (input[ABSOLUTE_POS] - low) / (high - low);
    }

    normalised[ABSOLUTE_POS] = value;
}

#[cube(launch_unchecked)]
fn fill(msr: &mut Array<f32>, value: f32) {
    if ABSOLUTE_POS >= msr.len() {
        terminate!();
    }

    msr[ABSOLUTE_POS] = value;
}

#[cube(launch_unchecked)]
fn accumulate(normalised: &Array<f32>, blurred: &Array<f32>, msr: &mut Array<f32>) {
    if ABSOLUTE_POS >= msr.len() {
        terminate!();
    }

    let intensity = normalised[ABSOLUTE_POS];
    let gaussian = blurred[ABSOLUTE_POS];

    let mut factor = 1.0f32;
    if gaussian > 0.0 {
        factor = intensity / gaussian + 1.0;
    }

    msr[ABSOLUTE_POS] += f32::ln(factor);
}

#[cube(launch_unchecked)]
fn finish(msr: &mut Array<f32>, count: f32) {
    if ABSOLUTE_POS >= msr.len() {
        terminate!();
    }

    msr[ABSOLUTE_POS] = msr[ABSOLUTE_POS] / count;
}

#[cube(launch_unchecked)]
fn stretch(msr: &Array<f32>, low: &Array<f32>, high: &Array<f32>, output: &mut Array<f32>, scale: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let low_value = low[0];
    let high_value = high[0];

    let mut stretched = 0.0f32;
    if high_value > low_value {
        let fraction = (msr[ABSOLUTE_POS] - low_value) / (high_value - low_value);
        stretched = f32::clamp(fraction, 0.0, 1.0);
    }

    output[ABSOLUTE_POS] = stretched * scale;
}

/// Runs the multi-scale retinex of `planes.input` into `planes.output`.
///
/// The input is normalised by its own extremes, then blurred at each of `settings.sigmas` and
/// combined into a log response. The response is stretched between its `settings.lower` and
/// `settings.upper` percentiles and scaled by `settings.scale`. A flat input or crossed
/// percentiles give 0 everywhere.
pub fn retinex<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: RetinexPlanes<'_>,
    settings: &RetinexSettings,
) {
    debug_assert!(!settings.sigmas.is_empty(), "sigmas must not be empty");

    let samples = planes.input.samples();

    let input_range_planes = ReducePlanes {
        input: planes.input,
        first: planes.first,
        second: planes.second,
        result: planes.range,
    };
    extremes(client, dispatch, input_range_planes);

    let input = planes.input.array::<R>();
    let range = planes.range.array::<R>();
    let normalised = planes.normalised.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and normalised planes hold `samples` f32 values, and the range plane 2.
    unsafe {
        normalise::launch_unchecked::<R>(client, cubes, unit, input, range, normalised);
    }

    let msr = planes.msr.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr plane holds `samples` f32 values.
    unsafe {
        fill::launch_unchecked::<R>(client, cubes, unit, msr, 0.0);
    }

    for sigma in &settings.sigmas {
        let filter = RecursiveGaussian::new(*sigma);
        let gaussian_planes = RecursivePlanes {
            input: planes.normalised,
            output: planes.blurred,
        };
        recursive_gaussian(client, dispatch, gaussian_planes, filter);

        let normalised = planes.normalised.array::<R>();
        let blurred = planes.blurred.array::<R>();
        let msr = planes.msr.array::<R>();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();

        // Safety: the normalised, blurred and msr planes each hold `samples` f32 values.
        unsafe {
            accumulate::launch_unchecked::<R>(client, cubes, unit, normalised, blurred, msr);
        }
    }

    let count = settings.sigmas.len() as f32;
    let msr = planes.msr.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr plane holds `samples` f32 values.
    unsafe {
        finish::launch_unchecked::<R>(client, cubes, unit, msr, count);
    }

    let msr_range_planes = ReducePlanes {
        input: planes.msr,
        first: planes.first,
        second: planes.second,
        result: planes.range,
    };
    extremes(client, dispatch, msr_range_planes);

    let lower_percentile = settings.lower * 100.0;
    let low_planes = PercentilePlanes {
        input: planes.msr,
        range: planes.range,
        bins: planes.bins,
        result: planes.low,
    };
    percentile(client, dispatch, low_planes, lower_percentile);

    let upper_percentile = (1.0 - settings.upper) * 100.0;
    let high_planes = PercentilePlanes {
        input: planes.msr,
        range: planes.range,
        bins: planes.bins,
        result: planes.high,
    };
    percentile(client, dispatch, high_planes, upper_percentile);

    let msr = planes.msr.array::<R>();
    let low = planes.low.array::<R>();
    let high = planes.high.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr and output planes hold `samples` f32 values, and low and high each hold 1.
    unsafe {
        stretch::launch_unchecked::<R>(client, cubes, unit, msr, low, high, output, settings.scale);
    }
}
