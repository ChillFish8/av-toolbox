// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::recursive::{RecursiveGaussian, RecursivePlanes, recursive_gaussian};
use crate::blur::{BlurPlanes, Gaussian, MAX_SIGMA, Storage};
use crate::histogram::{PercentilePlanes, percentile};
use crate::reduce::{ReducePlanes, extremes};
use crate::{DevicePlane, Dispatch, Edge};

/// The widest sigma retinex blurs with the shared [Gaussian](crate::blur::Gaussian).
///
/// Wider sigmas run the [RecursiveGaussian](crate::retinex::RecursiveGaussian), whose cost does not
/// grow with sigma.
pub const GAUSSIAN_MAX_SIGMA: f32 = 500.0;

const _: () = assert!(GAUSSIAN_MAX_SIGMA <= MAX_SIGMA);

/// Whether retinex blurs at `sigma` with the shared [Gaussian](crate::blur::Gaussian) rather than the
/// [RecursiveGaussian](crate::retinex::RecursiveGaussian).
pub fn uses_gaussian(sigma: f32) -> bool {
    sigma <= GAUSSIAN_MAX_SIGMA
}

/// The multi-scale retinex's settings.
#[derive(Debug, Clone, PartialEq)]
pub struct RetinexSettings {
    sigmas: Vec<f32>,
    /// The fraction of samples clipped from the dark end of the response, not a percent.
    pub lower: f32,
    /// The fraction of samples clipped from the bright end of the response, not a percent.
    pub upper: f32,
    /// The value the stretched response is scaled up to.
    pub scale: f32,
}

impl RetinexSettings {
    /// Settings that blur at each of `sigmas`.
    ///
    /// `lower` and `upper` are the fractions of samples clipped from each end of the response, and
    /// `scale` is the value the stretched response is scaled up to.
    pub fn new(sigmas: Vec<f32>, lower: f32, upper: f32, scale: f32) -> Self {
        Self {
            sigmas,
            lower,
            upper,
            scale,
        }
    }

    /// The deviations the input is blurred at.
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }
}

/// How retinex blurs at one sigma.
#[derive(Debug)]
enum ScaleBlur {
    Gaussian(Gaussian),
    Recursive(RecursiveGaussian),
}

/// The blurs retinex runs, one for each of its settings' sigmas, built once for the device.
///
/// Each sigma up to [GAUSSIAN_MAX_SIGMA] blurs with the shared [Gaussian](crate::blur::Gaussian) and
/// replicate edges. Wider ones run the [RecursiveGaussian](crate::retinex::RecursiveGaussian).
#[derive(Debug)]
pub struct RetinexBlurs {
    sigmas: Vec<f32>,
    blurs: Vec<ScaleBlur>,
}

impl RetinexBlurs {
    pub fn new(client: &Client, settings: &RetinexSettings) -> Self {
        let blurs = settings
            .sigmas
            .iter()
            .map(|sigma| scale_blur(client, *sigma))
            .collect();
        let sigmas = settings.sigmas.clone();

        Self { sigmas, blurs }
    }
}

fn scale_blur(client: &Client, sigma: f32) -> ScaleBlur {
    if uses_gaussian(sigma) {
        let gaussian = Gaussian::new(client, sigma);
        return ScaleBlur::Gaussian(gaussian);
    }

    let recursive = RecursiveGaussian::new(sigma);

    ScaleBlur::Recursive(recursive)
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
fn normalise(input: &[f32], range: &[f32], normalised: &mut [f32]) {
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
fn fill(msr: &mut [f32], value: f32) {
    if ABSOLUTE_POS >= msr.len() {
        terminate!();
    }

    msr[ABSOLUTE_POS] = value;
}

#[cube(launch_unchecked)]
fn accumulate(normalised: &[f32], blurred: &[f32], msr: &mut [f32]) {
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
fn finish(msr: &mut [f32], count: f32) {
    if ABSOLUTE_POS >= msr.len() {
        terminate!();
    }

    msr[ABSOLUTE_POS] /= count;
}

#[cube(launch_unchecked)]
fn stretch(msr: &[f32], low: &[f32], high: &[f32], output: &mut [f32], scale: f32) {
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

/// Blurs `planes.normalised` into `planes.blurred` with `blur`.
///
/// The gaussian passes through `planes.output`, which is free until the response is stretched.
fn blur_scale(client: &Client, dispatch: &impl Dispatch, planes: RetinexPlanes<'_>, blur: &ScaleBlur) {
    match blur {
        ScaleBlur::Gaussian(gaussian) => {
            let blur_planes = BlurPlanes {
                input: planes.normalised,
                scratch: planes.output,
                output: planes.blurred,
            };
            gaussian.apply_with_edge(client, dispatch, blur_planes, Edge::Clamp, Storage::F32);
        },
        ScaleBlur::Recursive(filter) => {
            let recursive_planes = RecursivePlanes {
                input: planes.normalised,
                output: planes.blurred,
            };
            recursive_gaussian(client, dispatch, recursive_planes, *filter);
        },
    }
}

/// Runs the multi-scale retinex of `planes.input` into `planes.output`.
///
/// The input is normalised by its own extremes, then blurred with each of `blurs` and combined into
/// a log response. The response is stretched between its `settings.lower` and `settings.upper`
/// percentiles and scaled by `settings.scale`. A flat input or crossed percentiles give 0
/// everywhere. `blurs` must be built from `settings`.
pub fn retinex(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RetinexPlanes<'_>,
    settings: &RetinexSettings,
    blurs: &RetinexBlurs,
) {
    debug_assert!(!settings.sigmas.is_empty(), "sigmas must not be empty");
    assert_eq!(
        settings.sigmas, blurs.sigmas,
        "the blurs must be built from the settings"
    );

    let samples = planes.input.samples();

    let input_range_planes = ReducePlanes {
        input: planes.input,
        first: planes.first,
        second: planes.second,
        result: planes.range,
    };
    extremes(client, dispatch, input_range_planes);

    let input = planes.input.array();
    let range = planes.range.array();
    let normalised = planes.normalised.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and normalised planes hold `samples` f32 values, and the range plane 2.
    unsafe {
        normalise::launch_unchecked(client, cubes, unit, input, range, normalised);
    }

    let msr = planes.msr.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr plane holds `samples` f32 values.
    unsafe {
        fill::launch_unchecked(client, cubes, unit, msr, 0.0);
    }

    for blur in &blurs.blurs {
        blur_scale(client, dispatch, planes, blur);

        let normalised = planes.normalised.array();
        let blurred = planes.blurred.array();
        let msr = planes.msr.array();
        let cubes = dispatch.cube_count(samples);
        let unit = dispatch.cube_dim();

        // Safety: the normalised, blurred and msr planes each hold `samples` f32 values.
        unsafe {
            accumulate::launch_unchecked(client, cubes, unit, normalised, blurred, msr);
        }
    }

    let count = settings.sigmas.len() as f32;
    let msr = planes.msr.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr plane holds `samples` f32 values.
    unsafe {
        finish::launch_unchecked(client, cubes, unit, msr, count);
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

    let msr = planes.msr.array();
    let low = planes.low.array();
    let high = planes.high.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the msr and output planes hold `samples` f32 values, and low and high each hold 1.
    unsafe {
        stretch::launch_unchecked(client, cubes, unit, msr, low, high, output, settings.scale);
    }
}
