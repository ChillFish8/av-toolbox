// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::depth::Element;
use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn widen_kernel<E: Int>(input: &Array<E>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::cast_from(input[ABSOLUTE_POS]);
}

#[cube(launch_unchecked)]
fn narrow_kernel<E: Int>(input: &Array<f32>, output: &mut Array<E>, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let rounded = f32::floor(input[ABSOLUTE_POS] + 0.5);
    let clamped = f32::clamp(rounded, 0.0, peak);

    output[ABSOLUTE_POS] = E::cast_from(clamped);
}

/// Copies an integer plane into f32 code values.
pub fn widen<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    element: Element,
) {
    match element {
        Element::U8 => launch_widen::<u8, R>(client, dispatch, input, output),
        Element::U16 => launch_widen::<u16, R>(client, dispatch, input, output),
    }
}

fn launch_widen<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input holds `samples` values of `E` and the output as many f32 values.
    unsafe {
        widen_kernel::launch_unchecked::<E, R>(client, cubes, unit, input_array, output_array);
    }
}

/// Rounds f32 code values to nearest, clamped to `0..=peak`, into an integer plane.
pub fn narrow<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    element: Element,
    peak: f32,
) {
    match element {
        Element::U8 => launch_narrow::<u8, R>(client, dispatch, input, output, peak),
        Element::U16 => launch_narrow::<u16, R>(client, dispatch, input, output, peak),
    }
}

fn launch_narrow<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    peak: f32,
) {
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input holds `samples` f32 values and the output as many values of `E`.
    unsafe {
        narrow_kernel::launch_unchecked::<E, R>(client, cubes, unit, input_array, output_array, peak);
    }
}
