// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{NeighbourPlanes, read};
use crate::Dispatch;

#[cube(launch_unchecked)]
fn prewitt_kernel(input: &[f32], output: &mut [f32], width: u32, height: u32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;

    let top_left = read(input, width, height, x - 1, y - 1);
    let top = read(input, width, height, x, y - 1);
    let top_right = read(input, width, height, x + 1, y - 1);
    let left = read(input, width, height, x - 1, y);
    let right = read(input, width, height, x + 1, y);
    let bottom_left = read(input, width, height, x - 1, y + 1);
    let bottom = read(input, width, height, x, y + 1);
    let bottom_right = read(input, width, height, x + 1, y + 1);

    let diagonal = top_left + top + left - right - bottom - bottom_right;
    let vertical = top_left + top + top_right - bottom_left - bottom - bottom_right;
    let horizontal = top_left - top_right + left - right + bottom_left - bottom_right;
    let anti_diagonal = left - top - top_right - right + bottom_left + bottom;

    let diagonal_size = f32::abs(diagonal);
    let vertical_size = f32::abs(vertical);
    let horizontal_size = f32::abs(horizontal);
    let anti_diagonal_size = f32::abs(anti_diagonal);
    let first = f32::max(diagonal_size, vertical_size);
    let second = f32::max(horizontal_size, anti_diagonal_size);
    let strongest = f32::max(first, second);

    output[ABSOLUTE_POS] = f32::min(strongest, peak);
}

/// Writes havsfunc's `AvsPrewitt` edge strength, the largest absolute response of four compass kernels.
///
/// Each response is clamped to `peak`, as the integer convolution it mirrors is.
pub fn prewitt(client: &Client, dispatch: &impl Dispatch, planes: NeighbourPlanes<'_>, peak: f32) {
    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, and every read is reflected inside the plane.
    unsafe {
        prewitt_kernel::launch_unchecked(client, cubes, unit, input, output, width, height, peak);
    }
}
