#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::dehalo::{DifferencePlanes, difference};
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

/// The settings of one vinverse pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VinverseParams {
    /// How strongly the remaining combing is countered.
    pub contra_str: f32,
    /// How much of a correction that fights the input is kept, between 0 and 1.
    pub scl: f32,
    /// The most a sample may move from the input, in code values. `None` leaves it unlimited.
    pub amnt: Option<f32>,
}

/// The planes one vinverse pass reads, writes and uses as scratch, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct VinversePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub diff: DevicePlane<'a>,
    pub diff_blurred: DevicePlane<'a>,
    pub sbr: DevicePlane<'a>,
}

/// Takes off the part of the comb a second blur removes, unless it points against the comb.
#[cube(launch_unchecked)]
fn sbr_kernel(input: &Array<f32>, diff: &Array<f32>, diff_blurred: &Array<f32>, sbr: &mut Array<f32>) {
    if ABSOLUTE_POS >= sbr.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let comb = diff[ABSOLUTE_POS];
    let residual = comb - diff_blurred[ABSOLUTE_POS];

    let mut removed = comb;
    if f32::abs(residual) < f32::abs(comb) {
        removed = residual;
    }

    let mut result = value - removed;
    if residual * comb < 0.0 {
        result = value;
    }

    sbr[ABSOLUTE_POS] = result;
}

/// Adds the contra-sharpening step onto `sbr` and limits the result to `limit` around `input`.
#[cube(launch_unchecked)]
fn vinverse_kernel(
    input: &Array<f32>,
    sbr: &Array<f32>,
    sbr_blurred: &Array<f32>,
    output: &mut Array<f32>,
    contra_str: f32,
    scl: f32,
    limit: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let base = sbr[ABSOLUTE_POS];
    let contra = (base - sbr_blurred[ABSOLUTE_POS]) * contra_str;
    let removed = value - base;

    let mut smaller = removed;
    if f32::abs(contra) < f32::abs(removed) {
        smaller = contra;
    }

    let mut step = smaller;
    if contra * removed < 0.0 {
        step = smaller * scl;
    }

    let result = base + step;
    let capped = f32::min(result, value + limit);

    output[ABSOLUTE_POS] = f32::max(capped, value - limit);
}

/// Removes residual combing from `input` the way vs-jetpack's vinverse does, writing `output`.
///
/// The comb is the part of `input` that a vertical `1 2 1` blur removes. The part of that comb a
/// second blur removes is taken off, unless it points against the comb. A contra-sharpening step
/// then restores detail the first step flattened.
pub fn vinverse<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: VinversePlanes<'_>,
    params: &VinverseParams,
) {
    let blur_input = NeighbourPlanes {
        input: planes.input,
        output: planes.blurred,
    };
    vertical_121(client, dispatch, blur_input);

    let comb = DifferencePlanes {
        first: planes.input,
        second: planes.blurred,
        output: planes.diff,
    };
    difference(client, dispatch, comb);

    let blur_comb = NeighbourPlanes {
        input: planes.diff,
        output: planes.diff_blurred,
    };
    vertical_121(client, dispatch, blur_comb);

    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let input = planes.input.array::<R>();
    let diff = planes.diff.array::<R>();
    let diff_blurred = planes.diff_blurred.array::<R>();
    let sbr = planes.sbr.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        sbr_kernel::launch_unchecked::<R>(client, cubes, unit, input, diff, diff_blurred, sbr);
    }

    let blur_sbr = NeighbourPlanes {
        input: planes.sbr,
        output: planes.blurred,
    };
    vertical_121(client, dispatch, blur_sbr);

    let cubes = dispatch.cube_count(samples);
    let input = planes.input.array::<R>();
    let sbr = planes.sbr.array::<R>();
    let sbr_blurred = planes.blurred.array::<R>();
    let output = planes.output.array::<R>();
    let limit = params.amnt.unwrap_or(f32::MAX);

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        vinverse_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            sbr,
            sbr_blurred,
            output,
            params.contra_str,
            params.scl,
            limit,
        );
    }
}
