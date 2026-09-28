#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::neighbourhood::NeighbourPlanes;
use crate::{Dispatch, reflect};

/// Which edge operator computes a magnitude.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Operator {
    Prewitt,
    Sobel,
    Kirsch,
}

impl Operator {
    fn code(self) -> u32 {
        match self {
            Self::Prewitt => 0,
            Self::Sobel => 1,
            Self::Kirsch => 2,
        }
    }
}

/// How an edge magnitude is scaled and cut, in the plane's own code values.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct EdgeLimits {
    /// Scales the magnitude before it is cut.
    pub multi: f32,
    /// A magnitude below this becomes 0.
    pub low: f32,
    /// A magnitude above this becomes `peak`.
    pub high: f32,
    /// The ceiling every value is clamped to.
    pub peak: f32,
}

impl EdgeLimits {
    /// Limits that scale and cut nothing, up to `peak`.
    pub const fn open(peak: f32) -> Self {
        Self {
            multi: 1.0,
            low: 0.0,
            high: peak,
            peak,
        }
    }
}

#[cube]
fn tap(input: &Array<f32>, width: u32, height: u32, column: i32, row: i32) -> f32 {
    let x = reflect(column, width as i32);
    let y = reflect(row, height as i32);
    let index = y as u32 * width + x as u32;

    input[index as usize]
}

/// The neighbour `slot` steps clockwise from the top left, among the 8 already read taps.
#[cube]
fn ring_value(
    top_left: f32,
    top: f32,
    top_right: f32,
    right: f32,
    bottom_right: f32,
    bottom: f32,
    bottom_left: f32,
    left: f32,
    slot: u32,
) -> f32 {
    let mut value = top_left;

    if slot == 1u32 {
        value = top;
    } else if slot == 2u32 {
        value = top_right;
    } else if slot == 3u32 {
        value = right;
    } else if slot == 4u32 {
        value = bottom_right;
    } else if slot == 5u32 {
        value = bottom;
    } else if slot == 6u32 {
        value = bottom_left;
    } else if slot == 7u32 {
        value = left;
    }

    value
}

/// The largest `|8 * run - 3 * ring|` over the 8 runs of 3 that walk the ring clockwise.
#[cube]
fn kirsch_magnitude(
    top_left: f32,
    top: f32,
    top_right: f32,
    right: f32,
    bottom_right: f32,
    bottom: f32,
    bottom_left: f32,
    left: f32,
) -> f32 {
    let ring_total = top_left + top + top_right + right + bottom_right + bottom + bottom_left + left;
    let mut strongest = 0.0f32;

    for start in 0..8u32 {
        let second_slot = (start + 1u32) % 8u32;
        let third_slot = (start + 2u32) % 8u32;
        let first = ring_value(
            top_left,
            top,
            top_right,
            right,
            bottom_right,
            bottom,
            bottom_left,
            left,
            start,
        );
        let second = ring_value(
            top_left,
            top,
            top_right,
            right,
            bottom_right,
            bottom,
            bottom_left,
            left,
            second_slot,
        );
        let third = ring_value(
            top_left,
            top,
            top_right,
            right,
            bottom_right,
            bottom,
            bottom_left,
            left,
            third_slot,
        );
        let run = first + second + third;
        let value = 8.0 * run - 3.0 * ring_total;
        let value_size = f32::abs(value);

        strongest = f32::max(strongest, value_size);
    }

    strongest
}

#[cube(launch_unchecked)]
fn edges_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    operator: u32,
    multi: f32,
    low: f32,
    high: f32,
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

    let magnitude = if operator == 0u32 {
        let gx = top_left + left + bottom_left - top_right - right - bottom_right;
        let gy = top_left + top + top_right - bottom_left - bottom - bottom_right;
        f32::sqrt(gx * gx + gy * gy)
    } else if operator == 1u32 {
        let gx = top_left + 2.0 * left + bottom_left - top_right - 2.0 * right - bottom_right;
        let gy = top_left + 2.0 * top + top_right - bottom_left - 2.0 * bottom - bottom_right;
        f32::sqrt(gx * gx + gy * gy)
    } else {
        kirsch_magnitude(
            top_left,
            top,
            top_right,
            right,
            bottom_right,
            bottom,
            bottom_left,
            left,
        )
    };

    let scaled = magnitude * multi;
    let mut limited = scaled;
    if scaled < low {
        limited = 0.0;
    } else if scaled > high {
        limited = peak;
    }

    output[ABSOLUTE_POS] = f32::clamp(limited, 0.0, peak);
}

#[cube(launch_unchecked)]
fn binarize_kernel(input: &Array<f32>, output: &mut Array<f32>, threshold: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let mut binarized = peak;
    if value < threshold {
        binarized = 0.0;
    }

    output[ABSOLUTE_POS] = binarized;
}

/// Writes the edge strength of each sample, cut and filled by the limits.
pub fn edges<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    operator: Operator,
    limits: EdgeLimits,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let operator_code = operator.code();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, and every read is reflected inside the plane.
    unsafe {
        edges_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            output,
            width,
            height,
            operator_code,
            limits.multi,
            limits.low,
            limits.high,
            limits.peak,
        );
    }
}

/// Writes `peak` where a sample is at least `threshold`, otherwise 0.
pub fn binarize<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    threshold: f32,
    peak: f32,
) {
    let samples = planes.output.samples();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values.
    unsafe {
        binarize_kernel::launch_unchecked::<R>(client, cubes, unit, input, output, threshold, peak);
    }
}
