// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn pad_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    input_width: u32,
    input_height: u32,
    output_width: u32,
    amount: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % output_width) as i32;
    let y = (position / output_width) as i32;
    let signed_amount = amount as i32;

    let source_x = i32::clamp(x - signed_amount, 0, input_width as i32 - 1);
    let source_y = i32::clamp(y - signed_amount, 0, input_height as i32 - 1);
    let index = source_y as u32 * input_width + source_x as u32;

    output[ABSOLUTE_POS] = input[index as usize];
}

#[cube(launch_unchecked)]
fn crop_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    input_width: u32,
    output_width: u32,
    amount: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % output_width;
    let y = position / output_width;

    let source_x = x + amount;
    let source_y = y + amount;
    let index = source_y * input_width + source_x;

    output[ABSOLUTE_POS] = input[index as usize];
}

/// Grows `input` by `amount` on every side, each new sample repeating the nearest edge sample.
pub fn pad<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    amount: u32,
) {
    let input_width = input.width();
    let input_height = input.height();
    let output_width = output.width();
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the output holds its own samples, and every read clamps into the input.
    unsafe {
        pad_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            input_width,
            input_height,
            output_width,
            amount,
        );
    }
}

/// Removes `amount` from every side of `input`.
pub fn crop<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    amount: u32,
) {
    let input_width = input.width();
    let output_width = output.width();
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the output holds its own samples, and every read lands inside the input.
    unsafe {
        crop_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            input_width,
            output_width,
            amount,
        );
    }
}
