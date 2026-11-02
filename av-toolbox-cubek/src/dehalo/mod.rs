pub mod alpha;
pub mod dering;
pub mod edge_clean;
pub mod fine;
#[cfg(test)]
mod tests;
pub mod yahr;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The amount the dehalo filters pad a plane by before warping it, as havsfunc does.
pub const PADDING: u32 = 6;

/// The planes one masked merge reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MergePlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub filtered: DevicePlane<'a>,
    pub mask: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn merge_masked_kernel(clip: &[f32], filtered: &[f32], mask: &[f32], output: &mut [f32], peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = clip[ABSOLUTE_POS];
    let change = filtered[ABSOLUTE_POS] - value;

    output[ABSOLUTE_POS] = value + change * mask[ABSOLUTE_POS] / peak;
}

/// Moves `clip` toward `filtered` by `mask / peak` at each sample.
pub fn merge_masked(client: &Client, dispatch: &impl Dispatch, planes: MergePlanes<'_>, peak: f32) {
    let samples = planes.output.samples();
    let clip = planes.clip.array();
    let filtered = planes.filtered.array();
    let mask = planes.mask.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        merge_masked_kernel::launch_unchecked(client, cubes, unit, clip, filtered, mask, output, peak);
    }
}

/// The planes one difference reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct DifferencePlanes<'a> {
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn difference_kernel(first: &[f32], second: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = first[ABSOLUTE_POS] - second[ABSOLUTE_POS];
}

/// Writes `first - second` at each sample, which may be negative.
pub fn difference(client: &Client, dispatch: &impl Dispatch, planes: DifferencePlanes<'_>) {
    let samples = planes.output.samples();
    let first = planes.first.array();
    let second = planes.second.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        difference_kernel::launch_unchecked(client, cubes, unit, first, second, output);
    }
}
