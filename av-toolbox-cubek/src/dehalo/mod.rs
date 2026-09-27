pub mod alpha;
#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The planes one masked merge reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MergePlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub filtered: DevicePlane<'a>,
    pub mask: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn merge_masked_kernel(
    clip: &Array<f32>,
    filtered: &Array<f32>,
    mask: &Array<f32>,
    output: &mut Array<f32>,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = clip[ABSOLUTE_POS];
    let change = filtered[ABSOLUTE_POS] - value;

    output[ABSOLUTE_POS] = value + change * mask[ABSOLUTE_POS] / peak;
}

/// Moves `clip` toward `filtered` by `mask / peak` at each sample.
pub fn merge_masked<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MergePlanes<'_>,
    peak: f32,
) {
    let samples = planes.output.samples();
    let clip = planes.clip.array::<R>();
    let filtered = planes.filtered.array::<R>();
    let mask = planes.mask.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        merge_masked_kernel::launch_unchecked::<R>(client, cubes, unit, clip, filtered, mask, output, peak);
    }
}
