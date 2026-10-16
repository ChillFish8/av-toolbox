// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BlurPlanes, Gaussian, Storage};
use crate::{Dispatch, reflect};

#[cube(launch_unchecked)]
fn blur_rows<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    radius: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row_start = position - column;

    let mut total = 0.0f32;
    for tap in 0..2 * radius + 1 {
        let offset = tap as i32 - radius as i32;
        let source = reflect(column as i32 + offset, width as i32);
        let index = row_start + source as u32;
        total += f32::cast_from(input[index as usize]) * weights[tap as usize];
    }

    output[ABSOLUTE_POS] = F::cast_from(total);
}

#[cube(launch_unchecked)]
fn blur_columns<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    radius: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;

    let mut total = 0.0f32;
    for tap in 0..2 * radius + 1 {
        let offset = tap as i32 - radius as i32;
        let source = reflect(row as i32 + offset, height as i32);
        let index = source as u32 * width + column;
        total += f32::cast_from(input[index as usize]) * weights[tap as usize];
    }

    output[ABSOLUTE_POS] = F::cast_from(total);
}

/// Blurs rows then columns with one thread a sample, reading the taps straight from memory.
pub(super) fn apply<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    gaussian: &Gaussian,
    planes: BlurPlanes<'_>,
    storage: Storage,
) {
    let samples = planes.output.samples();
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    let radius = gaussian.radius();
    let input = planes.input.array::<R>();
    let scratch = planes.scratch.array::<R>();
    let weights = gaussian.weights::<R>();

    // Safety: every plane holds `samples` values of `storage` and the kernel writes nothing past the
    // output's length.
    unsafe {
        match storage {
            Storage::F32 => blur_rows::launch_unchecked::<f32, R>(
                client,
                cubes.clone(),
                shape,
                input,
                weights,
                scratch,
                width,
                radius,
            ),
            Storage::F16 => blur_rows::launch_unchecked::<half::f16, R>(
                client,
                cubes.clone(),
                shape,
                input,
                weights,
                scratch,
                width,
                radius,
            ),
        }
    }

    let scratch = planes.scratch.array::<R>();
    let output = planes.output.array::<R>();
    let weights = gaussian.weights::<R>();

    // Safety: every plane holds `samples` values of `storage` and the kernel writes nothing past the
    // output's length.
    unsafe {
        match storage {
            Storage::F32 => blur_columns::launch_unchecked::<f32, R>(
                client, cubes, shape, scratch, weights, output, width, height, radius,
            ),
            Storage::F16 => blur_columns::launch_unchecked::<half::f16, R>(
                client, cubes, shape, scratch, weights, output, width, height, radius,
            ),
        }
    }
}
