// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BlurPlanes, FilterPlanes, Gaussian, Storage};
use crate::border::fold;
use crate::{Dispatch, Edge};

#[cube(launch_unchecked)]
fn blur_rows<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    radius: u32,
    #[comptime] clamp: bool,
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
        let source = fold(column as i32 + offset, width, clamp);
        let index = row_start + source;
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
    #[comptime] clamp: bool,
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
        let source = fold(row as i32 + offset, height, clamp);
        let index = source * width + column;
        total += f32::cast_from(input[index as usize]) * weights[tap as usize];
    }

    output[ABSOLUTE_POS] = F::cast_from(total);
}

/// Blurs rows then columns with one thread a sample, reading the taps straight from memory.
///
/// It reads past the edge as `edge` says.
pub(super) fn apply<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    gaussian: &Gaussian,
    planes: BlurPlanes<'_>,
    edge: Edge,
    storage: Storage,
) {
    let samples = planes.output.samples();
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();

    let radius = gaussian.radius();
    let clamp = edge == Edge::Clamp;
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
                clamp,
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
                clamp,
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
                client, cubes, shape, scratch, weights, output, width, height, radius, clamp,
            ),
            Storage::F16 => blur_columns::launch_unchecked::<half::f16, R>(
                client, cubes, shape, scratch, weights, output, width, height, radius, clamp,
            ),
        }
    }
}

/// Blurs across then down for one sample, folding each read at the edge as `clamp` says.
///
/// Each row's sum across is finished before it is weighted down, in the order a two pass blur takes.
#[cube(launch_unchecked)]
fn blur_both_axes<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    radius: u32,
    #[comptime] clamp: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;

    let mut total = 0.0f32;
    for row_tap in 0..2 * radius + 1 {
        let row_offset = row_tap as i32 - radius as i32;
        let source_row = fold(row as i32 + row_offset, height, clamp);
        let row_start = source_row * width;

        let mut across = 0.0f32;
        for column_tap in 0..2 * radius + 1 {
            let column_offset = column_tap as i32 - radius as i32;
            let source_column = fold(column as i32 + column_offset, width, clamp);
            let sample = f32::cast_from(input[(row_start + source_column) as usize]);
            across += sample * weights[column_tap as usize];
        }

        total += across * weights[row_tap as usize];
    }

    output[ABSOLUTE_POS] = F::cast_from(total);
}

/// Blurs across then down in one dispatch, one thread a sample, reading past the edge as `edge` says.
///
/// Every read goes straight to memory, so it suits only short kernels.
pub(super) fn apply_in_one_pass<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    weights: ArrayArg<R>,
    radius: u32,
    planes: FilterPlanes<'_>,
    edge: Edge,
    storage: Storage,
) {
    let samples = planes.output.samples();
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = dispatch.cube_count(samples);
    let shape = dispatch.cube_dim();
    let clamp = edge == Edge::Clamp;
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: both planes hold `samples` values of `storage`, every read is folded into the plane and
    // the kernel writes nothing past the output's length.
    unsafe {
        match storage {
            Storage::F32 => blur_both_axes::launch_unchecked::<f32, R>(
                client, cubes, shape, input, weights, output, width, height, radius, clamp,
            ),
            Storage::F16 => blur_both_axes::launch_unchecked::<half::f16, R>(
                client, cubes, shape, input, weights, output, width, height, radius, clamp,
            ),
        }
    }
}
