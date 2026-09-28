// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The Young and van Vliet direct-form recursive Gaussian feedback of `sigma`, as
/// `(gain, f1, f2, f3)`, computed in f64.
///
/// `f1`, `f2` and `f3` are the direct form's `b1 / b0`, `b2 / b0` and `b3 / b0`.
pub(crate) fn direct_form_coefficients(sigma: f32) -> (f64, f64, f64, f64) {
    let sigma = f64::from(sigma);
    let q = if sigma < 2.5 {
        3.97156 - 4.14554 * (1.0 - 0.26891 * sigma).sqrt()
    } else {
        0.98711 * sigma - 0.96330
    };

    let b0 = 1.57825 + 2.44413 * q + 1.4281 * q.powi(2) + 0.422205 * q.powi(3);
    let b1 = 2.44413 * q + 2.85619 * q.powi(2) + 1.26661 * q.powi(3);
    let b2 = -(1.4281 * q.powi(2) + 1.26661 * q.powi(3));
    let b3 = 0.422205 * q.powi(3);
    let gain = 1.0 - (b1 + b2 + b3) / b0;

    (gain, b1 / b0, b2 / b0, b3 / b0)
}

/// The real root, between 0 and 1, of `z^3 - f1 z^2 - f2 z - f3`, found by bisection.
///
/// The residual is negative at 0 and equal to the gain at 1, so the root lies between them for
/// any sigma of at least 0.5.
fn real_root(f1: f64, f2: f64, f3: f64) -> f64 {
    let residual = |z: f64| z.powi(3) - f1 * z.powi(2) - f2 * z - f3;

    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..100 {
        let midpoint = 0.5 * (low + high);
        let value = residual(midpoint);

        if value < 0.0 {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }

    0.5 * (low + high)
}

/// Factors direct-form feedback `f1`, `f2`, `f3` into a real pole and a complex conjugate pair,
/// as `(pole, pair_a, pair_b)`, so that `f1 = pair_a + pole`, `f2 = pair_b - pole * pair_a` and
/// `f3 = -pole * pair_b`.
pub(crate) fn factor(f1: f64, f2: f64, f3: f64) -> (f64, f64, f64) {
    let pole = real_root(f1, f2, f3);
    let pair_a = f1 - pole;
    let pair_b = f2 + pole * pair_a;

    (pole, pair_a, pair_b)
}

/// The coefficients of a recursive Gaussian, factored into a first-order section and a
/// second-order section.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RecursiveGaussian {
    pub real_gain: f32,
    pub real_pole: f32,
    pub pair_gain: f32,
    pub pair: [f32; 2],
}

impl RecursiveGaussian {
    /// Factors the Young and van Vliet recursive Gaussian of `sigma` into a first-order and a
    /// second-order section, computed in f64 and stored as f32.
    ///
    /// Each gain is taken from its section's rounded pole or pair, so each section sums to 1 in
    /// f32. `sigma` must be finite and at least 0.5.
    pub fn new(sigma: f32) -> Self {
        debug_assert!(
            sigma.is_finite() && sigma >= 0.5,
            "sigma must be finite and at least 0.5, got {sigma}"
        );

        let (_, f1, f2, f3) = direct_form_coefficients(sigma);
        let (pole, pair_a, pair_b) = factor(f1, f2, f3);

        let real_pole = pole as f32;
        let pair = [pair_a as f32, pair_b as f32];
        let real_gain = 1.0 - f64::from(real_pole);
        let pair_gain = 1.0 - f64::from(pair[0]) - f64::from(pair[1]);

        Self {
            real_gain: real_gain as f32,
            real_pole,
            pair_gain: pair_gain as f32,
            pair,
        }
    }
}

/// The planes one recursive Gaussian reads and writes, both f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RecursivePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn recursive_rows(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    real_gain: f32,
    real_pole: f32,
    pair_gain: f32,
    pair_a: f32,
    pair_b: f32,
) {
    let row = ABSOLUTE_POS as u32;
    if row >= height {
        terminate!();
    }

    let base = row * width;

    // First-order section, forward. Writes into `output`.
    let mut forward_first_previous = input[base as usize];
    for column in 0..width {
        let index = base + column;
        let value = input[index as usize];
        let filtered = real_gain * value + real_pole * forward_first_previous;

        output[index as usize] = filtered;
        forward_first_previous = filtered;
    }

    // Second-order section, forward. Reads and overwrites `output` in place.
    let forward_seed = output[base as usize];
    let mut forward_second_previous_1 = forward_seed;
    let mut forward_second_previous_2 = forward_seed;
    for column in 0..width {
        let index = base + column;
        let value = output[index as usize];
        let filtered =
            pair_gain * value + pair_a * forward_second_previous_1 + pair_b * forward_second_previous_2;

        output[index as usize] = filtered;
        forward_second_previous_2 = forward_second_previous_1;
        forward_second_previous_1 = filtered;
    }

    // First-order section, backward, walking from the end.
    let backward_seed = output[(base + width - 1) as usize];
    let mut backward_first_previous = backward_seed;
    for step in 0..width {
        let column = width - 1 - step;
        let index = base + column;
        let value = output[index as usize];
        let filtered = real_gain * value + real_pole * backward_first_previous;

        output[index as usize] = filtered;
        backward_first_previous = filtered;
    }

    // Second-order section, backward, walking from the end.
    let backward_second_seed = output[(base + width - 1) as usize];
    let mut backward_second_previous_1 = backward_second_seed;
    let mut backward_second_previous_2 = backward_second_seed;
    for step in 0..width {
        let column = width - 1 - step;
        let index = base + column;
        let value = output[index as usize];
        let filtered =
            pair_gain * value + pair_a * backward_second_previous_1 + pair_b * backward_second_previous_2;

        output[index as usize] = filtered;
        backward_second_previous_2 = backward_second_previous_1;
        backward_second_previous_1 = filtered;
    }
}

#[cube(launch_unchecked)]
fn recursive_columns(
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    real_gain: f32,
    real_pole: f32,
    pair_gain: f32,
    pair_a: f32,
    pair_b: f32,
) {
    let column = ABSOLUTE_POS as u32;
    if column >= width {
        terminate!();
    }

    // First-order section, forward.
    let mut forward_first_previous = output[column as usize];
    for row in 0..height {
        let index = row * width + column;
        let value = output[index as usize];
        let filtered = real_gain * value + real_pole * forward_first_previous;

        output[index as usize] = filtered;
        forward_first_previous = filtered;
    }

    // Second-order section, forward.
    let forward_seed = output[column as usize];
    let mut forward_second_previous_1 = forward_seed;
    let mut forward_second_previous_2 = forward_seed;
    for row in 0..height {
        let index = row * width + column;
        let value = output[index as usize];
        let filtered =
            pair_gain * value + pair_a * forward_second_previous_1 + pair_b * forward_second_previous_2;

        output[index as usize] = filtered;
        forward_second_previous_2 = forward_second_previous_1;
        forward_second_previous_1 = filtered;
    }

    // First-order section, backward, walking from the end.
    let backward_seed = output[((height - 1) * width + column) as usize];
    let mut backward_first_previous = backward_seed;
    for step in 0..height {
        let row = height - 1 - step;
        let index = row * width + column;
        let value = output[index as usize];
        let filtered = real_gain * value + real_pole * backward_first_previous;

        output[index as usize] = filtered;
        backward_first_previous = filtered;
    }

    // Second-order section, backward, walking from the end.
    let backward_second_seed = output[((height - 1) * width + column) as usize];
    let mut backward_second_previous_1 = backward_second_seed;
    let mut backward_second_previous_2 = backward_second_seed;
    for step in 0..height {
        let row = height - 1 - step;
        let index = row * width + column;
        let value = output[index as usize];
        let filtered =
            pair_gain * value + pair_a * backward_second_previous_1 + pair_b * backward_second_previous_2;

        output[index as usize] = filtered;
        backward_second_previous_2 = backward_second_previous_1;
        backward_second_previous_1 = filtered;
    }
}

/// Runs the recursive Gaussian `filter` over `planes`, rows forward and backward, then columns
/// forward and backward over the result.
///
/// Each pass runs the first-order section then the second-order section forward, then the same
/// two sections backward over that result.
pub fn recursive_gaussian<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: RecursivePlanes<'_>,
    filter: RecursiveGaussian,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(height as usize);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `width * height` f32 values, and each thread walks only its own row.
    unsafe {
        recursive_rows::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            output,
            width,
            height,
            filter.real_gain,
            filter.real_pole,
            filter.pair_gain,
            filter.pair[0],
            filter.pair[1],
        );
    }

    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(width as usize);
    let unit = dispatch.cube_dim();

    // Safety: the output plane holds `width * height` f32 values, and each thread walks only its
    // own column.
    unsafe {
        recursive_columns::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            output,
            width,
            height,
            filter.real_gain,
            filter.real_pole,
            filter.pair_gain,
            filter.pair[0],
            filter.pair[1],
        );
    }
}
