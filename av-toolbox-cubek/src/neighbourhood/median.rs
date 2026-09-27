// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{NeighbourPlanes, read};
use crate::Dispatch;

#[cube]
fn window_value(
    input: &Array<f32>,
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    size: u32,
    radius: i32,
    slot: u32,
) -> f32 {
    let offset_x = (slot % size) as i32 - radius;
    let offset_y = (slot / size) as i32 - radius;

    read(input, width, height, x + offset_x, y + offset_y)
}

#[cube(launch_unchecked)]
fn median_kernel(input: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32, radius: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let size = 2 * radius + 1;
    let count = size * size;
    let middle = count / 2;
    let reach = radius as i32;

    let mut chosen = 0.0f32;
    let mut found: u32 = 0u32;

    for candidate_slot in 0..count {
        let candidate = window_value(input, width, height, x, y, size, reach, candidate_slot);
        let mut below = 0u32;
        let mut equal = 0u32;

        for other_slot in 0..count {
            let other = window_value(input, width, height, x, y, size, reach, other_slot);

            if other < candidate {
                below += 1;
            } else if other == candidate {
                equal += 1;
            }
        }

        if found == 0u32 && below <= middle && middle < below + equal {
            chosen = candidate;
            found = 1u32;
        }
    }

    output[ABSOLUTE_POS] = chosen;
}

/// Writes the median of each sample's square window reaching `radius` samples each side.
///
/// `radius` is 1, 2 or 3, for a 3x3, 5x5 or 7x7 window.
pub fn median<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    radius: u32,
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
        median_kernel::launch_unchecked::<R>(client, cubes, unit, input, output, width, height, radius);
    }
}
