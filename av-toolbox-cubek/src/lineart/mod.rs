#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::neighbourhood::NeighbourPlanes;
use crate::{DevicePlane, Dispatch, reflect};

/// The planes one merge reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MergePlanes<'a> {
    /// The unfiltered plane.
    pub input: DevicePlane<'a>,
    /// The filtered plane, overwritten with the merge.
    pub filtered: DevicePlane<'a>,
    /// The weight of the filtered plane at each sample, between 0 and `weight_peak`.
    pub weight: DevicePlane<'a>,
}

/// Mixes the two planes as `input * (1 - share) + filtered * share`, with `share = weight / weight_peak`.
///
/// A weight of 0 or `weight_peak` gives a share of exactly 0 or 1, so those weights hand back the
/// input or the filtered value exactly.
#[cube(launch_unchecked)]
fn merge_kernel(input: &Array<f32>, filtered: &mut Array<f32>, weight: &Array<f32>, weight_peak: f32) {
    if ABSOLUTE_POS >= filtered.len() {
        terminate!();
    }

    let share = weight[ABSOLUTE_POS] / weight_peak;
    let kept = input[ABSOLUTE_POS] * (1.0 - share);
    let taken = filtered[ABSOLUTE_POS] * share;

    filtered[ABSOLUTE_POS] = kept + taken;
}

#[cube(launch_unchecked)]
fn vertical_121_kernel(input: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = (position / width) as i32;
    let above = reflect(row - 1, height as i32) as u32;
    let below = reflect(row + 1, height as i32) as u32;

    let top = input[(above * width + column) as usize];
    let centre = input[ABSOLUTE_POS];
    let bottom = input[(below * width + column) as usize];

    output[ABSOLUTE_POS] = (top + 2.0 * centre + bottom) / 4.0;
}

/// Writes `input + (filtered - input) * weight / weight_peak` into `filtered`.
pub fn merge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MergePlanes<'_>,
    weight_peak: f32,
) {
    let samples = planes.filtered.samples();
    let input = planes.input.array::<R>();
    let filtered = planes.filtered.array::<R>();
    let weight = planes.weight.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        merge_kernel::launch_unchecked::<R>(client, cubes, unit, input, filtered, weight, weight_peak);
    }
}

/// The vertical `1 2 1` blur over 4, reading past the top and bottom through `reflect`.
pub fn vertical_121<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, and every read is reflected inside the plane.
    unsafe {
        vertical_121_kernel::launch_unchecked::<R>(client, cubes, unit, input, output, width, height);
    }
}
