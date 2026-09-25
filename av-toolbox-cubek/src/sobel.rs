// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, reflect};

/// A plane and where its horizontal and vertical gradients land. All are f32 and of one size.
#[derive(Debug, Clone, Copy)]
pub struct SobelPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub horizontal: DevicePlane<'a>,
    pub vertical: DevicePlane<'a>,
}

#[cube]
fn tap(input: &Array<f32>, width: u32, height: u32, column: i32, row: i32) -> f32 {
    let x = reflect(column, width as i32);
    let y = reflect(row, height as i32);
    let index = y as u32 * width + x as u32;

    input[index as usize]
}

#[cube(launch_unchecked)]
fn gradients(
    input: &Array<f32>,
    horizontal: &mut Array<f32>,
    vertical: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    if ABSOLUTE_POS >= horizontal.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;

    let up_left = tap(input, width, height, x - 1, y - 1);
    let up = tap(input, width, height, x, y - 1);
    let up_right = tap(input, width, height, x + 1, y - 1);
    let left = tap(input, width, height, x - 1, y);
    let right = tap(input, width, height, x + 1, y);
    let down_left = tap(input, width, height, x - 1, y + 1);
    let down = tap(input, width, height, x, y + 1);
    let down_right = tap(input, width, height, x + 1, y + 1);

    horizontal[ABSOLUTE_POS] = (up_right + 2.0 * right + down_right) - (up_left + 2.0 * left + down_left);
    vertical[ABSOLUTE_POS] = (down_left + 2.0 * down + down_right) - (up_left + 2.0 * up + up_right);
}

/// Writes the 3x3 Sobel gradients of `input`.
pub fn sobel<R: Runtime>(client: &ComputeClient<R>, dispatch: &impl Dispatch, planes: SobelPlanes<'_>) {
    let samples = planes.input.samples();
    let width = planes.input.width();
    let height = planes.input.height();

    let input = planes.input.array::<R>();
    let horizontal = planes.horizontal.array::<R>();
    let vertical = planes.vertical.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    // Safety: all three planes hold `samples` f32 values and the kernel writes nothing past them.
    unsafe {
        gradients::launch_unchecked::<R>(client, cubes, shape, input, horizontal, vertical, width, height);
    }
}
