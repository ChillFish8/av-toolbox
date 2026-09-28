// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The real base pole of the recursive Gaussian, optimised for a sigma of 2.
const REAL_POLE: f64 = 1.86543;

/// The complex base pole of the recursive Gaussian as `(real, imaginary)`, optimised for a sigma
/// of 2. Its conjugate is the third pole.
const PAIR_POLE: (f64, f64) = (1.41650, 1.00829);

/// How many samples past a line's edge the filter's response takes to die away at `sigma`.
pub(crate) fn settling_length(sigma: f64) -> usize {
    let reach = (40.0 * sigma).ceil();

    reach as usize + 50
}

/// The complex base pole scaled by `q`, `d^(1/q)`, as `(modulus, argument)`.
fn scaled_pair_pole(q: f64) -> (f64, f64) {
    let (real, imaginary) = PAIR_POLE;
    let modulus = real.hypot(imaginary);
    let argument = imaginary.atan2(real);

    (modulus.powf(1.0 / q), argument / q)
}

/// The variance of the forward and backward filter whose base poles are scaled by `q`.
///
/// Each scaled pole `z` adds `2 z / (z - 1)²`, and the complex pair adds twice the real part of
/// its pole's term.
fn variance(q: f64) -> f64 {
    let real_pole = REAL_POLE.powf(1.0 / q);
    let real_term = 2.0 * real_pole / (real_pole - 1.0).powi(2);

    let (modulus, argument) = scaled_pair_pole(q);
    let pole_real = modulus * argument.cos();
    let pole_imaginary = modulus * argument.sin();
    let shifted_real = pole_real - 1.0;
    let squared_real = shifted_real * shifted_real - pole_imaginary * pole_imaginary;
    let squared_imaginary = 2.0 * shifted_real * pole_imaginary;
    let squared_norm = squared_real * squared_real + squared_imaginary * squared_imaginary;
    let pair_term = 2.0 * (pole_real * squared_real + pole_imaginary * squared_imaginary) / squared_norm;

    real_term + 2.0 * pair_term
}

/// The scale `q` whose scaled poles give a variance of `sigma²`, found by bisection.
///
/// The variance grows with `q`, so the upper bound doubles until it brackets `sigma²`.
pub(crate) fn solve_q(sigma: f64) -> f64 {
    let target = sigma * sigma;
    let mut low = 0.35;
    let mut high = 2.0;

    debug_assert!(
        variance(low) < target,
        "sigma {sigma} is below the smallest the poles can reach"
    );

    while variance(high) < target {
        high *= 2.0;
    }

    for _ in 0..200 {
        let midpoint = 0.5 * (low + high);
        let found = variance(midpoint);

        if (found - target).abs() <= 1e-12 * target {
            return midpoint;
        }

        if found < target {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }

    0.5 * (low + high)
}

/// The feedback of the recursive Gaussian of `sigma` as `(p, a, b)`, computed in f64.
///
/// `p` is the first-order section's pole, and `a` and `b` are the second-order section's weights
/// on its last two outputs.
pub(crate) fn sections(sigma: f64) -> (f64, f64, f64) {
    let q = solve_q(sigma);
    let pole = REAL_POLE.powf(-1.0 / q);

    let (modulus, argument) = scaled_pair_pole(q);
    let pair_modulus = 1.0 / modulus;
    let pair_argument = -argument;
    let pair_a = 2.0 * pair_modulus * pair_argument.cos();
    let pair_b = -(pair_modulus * pair_modulus);

    (pole, pair_a, pair_b)
}

/// The backward pass's starting state per unit of forward state deviation, for the feedback
/// `pole`, `pair_a` and `pair_b`, computed in f64.
///
/// Each column is the response to one unit deviation of the forward state `(h, y1, y1 - y2)`,
/// run on for `length` samples of zero input and then back through the backward sections from
/// their far end. The rows use the same `(h, y1, y1 - y2)` order.
fn edge_matrix(pole: f64, pair_a: f64, pair_b: f64, length: usize) -> [[f64; 3]; 3] {
    let real_gain = 1.0 - pole;
    let pair_gain = 1.0 - pair_a - pair_b;
    let deviations = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let mut matrix = [[0.0; 3]; 3];
    let mut continuation = vec![0.0f64; length];

    for (column, deviation) in deviations.iter().enumerate() {
        let [mut first, mut last, mut difference] = *deviation;

        // The continuation's input is 0, so the first-order section only decays.
        for value in continuation.iter_mut() {
            first -= real_gain * first;
            let step = pair_gain * (first - last) - pair_b * difference;

            last += step;
            difference = step;
            *value = last;
        }

        let mut backward_first = 0.0;
        let mut backward_last = 0.0;
        let mut backward_difference = 0.0;

        for value in continuation.iter().rev() {
            backward_first += real_gain * (value - backward_first);
            let step = pair_gain * (backward_first - backward_last) - pair_b * backward_difference;

            backward_last += step;
            backward_difference = step;
        }

        matrix[0][column] = backward_first;
        matrix[1][column] = backward_last;
        matrix[2][column] = backward_difference;
    }

    matrix
}

/// The coefficients of a recursive Gaussian, factored into a first-order section and a
/// second-order section.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct RecursiveGaussian {
    /// The first-order section's input weight.
    pub real_gain: f32,
    /// The first-order section's pole.
    pub real_pole: f32,
    /// The second-order section's input weight.
    pub pair_gain: f32,
    /// The second-order section's weights `[a, b]` on its last two outputs.
    pub pair: [f32; 2],
    /// The backward pass's starting state per unit of forward state deviation.
    ///
    /// Rows and columns are ordered as the first-order output `h`, the second-order last output
    /// `y1`, and the second-order difference `y1 - y2`.
    pub edge: [[f32; 3]; 3],
}

impl RecursiveGaussian {
    /// The van Vliet, Young and Verbeek recursive Gaussian of `sigma`, computed in f64 and stored
    /// as f32.
    ///
    /// Each gain is taken from its section's rounded pole or pair, so each section sums to 1 in
    /// f32. `sigma` must be finite and at least 0.5.
    pub fn new(sigma: f32) -> Self {
        debug_assert!(
            sigma.is_finite() && sigma >= 0.5,
            "sigma must be finite and at least 0.5, got {sigma}"
        );

        let sigma = f64::from(sigma);
        let (pole, pair_a, pair_b) = sections(sigma);
        let length = settling_length(sigma);
        let matrix = edge_matrix(pole, pair_a, pair_b, length);

        let real_pole = pole as f32;
        let pair = [pair_a as f32, pair_b as f32];
        let real_gain = 1.0 - f64::from(real_pole);
        let pair_gain = 1.0 - f64::from(pair[0]) - f64::from(pair[1]);
        let edge = matrix.map(|row| row.map(|value| value as f32));

        Self {
            real_gain: real_gain as f32,
            real_pole,
            pair_gain: pair_gain as f32,
            pair,
            edge,
        }
    }
}

/// The planes one recursive Gaussian reads and writes, both f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RecursivePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// One row of the edge matrix applied to the forward state's deviation from the line's edge.
#[cube]
fn edge_offset(
    weight_first: f32,
    weight_last: f32,
    weight_difference: f32,
    deviation_first: f32,
    deviation_last: f32,
    deviation_difference: f32,
) -> f32 {
    weight_first * deviation_first + weight_last * deviation_last + weight_difference * deviation_difference
}

#[cube(launch_unchecked)]
fn recursive_rows(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    real_gain: f32,
    pair_gain: f32,
    pair_b: f32,
    edge_00: f32,
    edge_01: f32,
    edge_02: f32,
    edge_10: f32,
    edge_11: f32,
    edge_12: f32,
    edge_20: f32,
    edge_21: f32,
    edge_22: f32,
) {
    let row = ABSOLUTE_POS as u32;
    if row >= height {
        terminate!();
    }

    let base = row * width;
    let start = input[base as usize];
    let edge = input[(base + width - 1) as usize];

    // First-order section, forward. Writes into `output`.
    let mut forward_first = start;
    for column in 0..width {
        let index = base + column;
        let value = input[index as usize];

        forward_first += real_gain * (value - forward_first);
        output[index as usize] = forward_first;
    }

    // Second-order section, forward. Reads and overwrites `output` in place.
    let mut forward_last = start;
    let mut forward_difference = 0.0f32;
    for column in 0..width {
        let index = base + column;
        let value = output[index as usize];
        let step = pair_gain * (value - forward_last) - pair_b * forward_difference;

        forward_last += step;
        forward_difference = step;
        output[index as usize] = forward_last;
    }

    // The backward pass starts where a row carried on by copies of `edge` would leave it.
    let deviation_first = forward_first - edge;
    let deviation_last = forward_last - edge;
    let offset_first = edge_offset(
        edge_00,
        edge_01,
        edge_02,
        deviation_first,
        deviation_last,
        forward_difference,
    );
    let offset_last = edge_offset(
        edge_10,
        edge_11,
        edge_12,
        deviation_first,
        deviation_last,
        forward_difference,
    );
    let seed_difference = edge_offset(
        edge_20,
        edge_21,
        edge_22,
        deviation_first,
        deviation_last,
        forward_difference,
    );

    // First-order section, backward, walking from the end.
    let mut backward_first = edge + offset_first;
    for step in 0..width {
        let column = width - 1 - step;
        let index = base + column;
        let value = output[index as usize];

        backward_first += real_gain * (value - backward_first);
        output[index as usize] = backward_first;
    }

    // Second-order section, backward, walking from the end.
    let mut backward_last = edge + offset_last;
    let mut backward_difference = seed_difference;
    for step in 0..width {
        let column = width - 1 - step;
        let index = base + column;
        let value = output[index as usize];
        let update = pair_gain * (value - backward_last) - pair_b * backward_difference;

        backward_last += update;
        backward_difference = update;
        output[index as usize] = backward_last;
    }
}

#[cube(launch_unchecked)]
fn recursive_columns(
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    real_gain: f32,
    pair_gain: f32,
    pair_b: f32,
    edge_00: f32,
    edge_01: f32,
    edge_02: f32,
    edge_10: f32,
    edge_11: f32,
    edge_12: f32,
    edge_20: f32,
    edge_21: f32,
    edge_22: f32,
) {
    let column = ABSOLUTE_POS as u32;
    if column >= width {
        terminate!();
    }

    // The forward pass overwrites the column, so both ends are read first.
    let start = output[column as usize];
    let edge = output[((height - 1) * width + column) as usize];

    // First-order section, forward.
    let mut forward_first = start;
    for row in 0..height {
        let index = row * width + column;
        let value = output[index as usize];

        forward_first += real_gain * (value - forward_first);
        output[index as usize] = forward_first;
    }

    // Second-order section, forward.
    let mut forward_last = start;
    let mut forward_difference = 0.0f32;
    for row in 0..height {
        let index = row * width + column;
        let value = output[index as usize];
        let step = pair_gain * (value - forward_last) - pair_b * forward_difference;

        forward_last += step;
        forward_difference = step;
        output[index as usize] = forward_last;
    }

    // The backward pass starts where a column carried on by copies of `edge` would leave it.
    let deviation_first = forward_first - edge;
    let deviation_last = forward_last - edge;
    let offset_first = edge_offset(
        edge_00,
        edge_01,
        edge_02,
        deviation_first,
        deviation_last,
        forward_difference,
    );
    let offset_last = edge_offset(
        edge_10,
        edge_11,
        edge_12,
        deviation_first,
        deviation_last,
        forward_difference,
    );
    let seed_difference = edge_offset(
        edge_20,
        edge_21,
        edge_22,
        deviation_first,
        deviation_last,
        forward_difference,
    );

    // First-order section, backward, walking from the end.
    let mut backward_first = edge + offset_first;
    for step in 0..height {
        let row = height - 1 - step;
        let index = row * width + column;
        let value = output[index as usize];

        backward_first += real_gain * (value - backward_first);
        output[index as usize] = backward_first;
    }

    // Second-order section, backward, walking from the end.
    let mut backward_last = edge + offset_last;
    let mut backward_difference = seed_difference;
    for step in 0..height {
        let row = height - 1 - step;
        let index = row * width + column;
        let value = output[index as usize];
        let update = pair_gain * (value - backward_last) - pair_b * backward_difference;

        backward_last += update;
        backward_difference = update;
        output[index as usize] = backward_last;
    }
}

/// Runs the recursive Gaussian `filter` over `planes`, rows forward and backward, then columns
/// forward and backward over the result.
///
/// Each pass runs the first-order section then the second-order section forward, then the same
/// two sections backward over that result. The backward pass starts from the state that a line
/// extended by copies of its last sample would reach, found from the forward pass's end state
/// through the filter's edge matrix.
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
    let [first_row, last_row, difference_row] = filter.edge;

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
            filter.pair_gain,
            filter.pair[1],
            first_row[0],
            first_row[1],
            first_row[2],
            last_row[0],
            last_row[1],
            last_row[2],
            difference_row[0],
            difference_row[1],
            difference_row[2],
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
            filter.pair_gain,
            filter.pair[1],
            first_row[0],
            first_row[1],
            first_row[2],
            last_row[0],
            last_row[1],
            last_row[2],
            difference_row[0],
            difference_row[1],
            difference_row[2],
        );
    }
}
