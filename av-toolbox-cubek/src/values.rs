// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::depth::Element;
use crate::matrix::{MatrixSamples, SampleRange, scaling};
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

#[cube(launch_unchecked)]
fn offset_kernel(input: &Array<f32>, output: &mut Array<f32>, amount: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] + amount;
}

#[cube(launch_unchecked)]
fn float_to_codes_kernel(input: &Array<f32>, output: &mut Array<f32>, offset: f32, scale: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let sample = input[ABSOLUTE_POS];
    let defined = select(f32::is_nan(sample), 0.0, sample);
    let code = defined * scale + offset;

    output[ABSOLUTE_POS] = f32::clamp(code, 0.0, peak);
}

#[cube(launch_unchecked)]
fn codes_to_float_kernel(input: &Array<f32>, output: &mut Array<f32>, offset: f32, scale: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = (input[ABSOLUTE_POS] - offset) / scale;
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

/// Adds `amount` to every sample of an f32 plane, writing the result into `output`.
pub fn offset<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    amount: f32,
) {
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and the output each hold `samples` f32 values.
    unsafe {
        offset_kernel::launch_unchecked::<R>(client, cubes, unit, input_array, output_array, amount);
    }
}

/// How full range float samples map onto the code values of a plane.
///
/// A code is `sample * scale + offset`, and `peak` is the largest code.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FloatCodes {
    pub offset: f32,
    pub scale: f32,
    pub peak: f32,
}

impl FloatCodes {
    /// The mapping onto `bits` deep codes in `range`, for a chroma plane when `chroma` is set.
    pub fn new(bits: u32, range: SampleRange, chroma: bool) -> Self {
        let samples = MatrixSamples::U16 { bits };
        let (offset, scale) = scaling(samples, range, chroma);
        let peak = ((1u32 << bits) - 1) as f32;

        Self {
            offset: offset as f32,
            scale: scale as f32,
            peak,
        }
    }
}

/// Maps float samples to f32 code values held to `0..=peak`, without rounding. A NaN sample reads as 0.
pub fn float_to_codes<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    codes: FloatCodes,
) {
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and the output each hold `samples` f32 values.
    unsafe {
        float_to_codes_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            codes.offset,
            codes.scale,
            codes.peak,
        );
    }
}

/// Maps f32 code values back to float samples, neither rounded nor clamped.
///
/// It divides by the scale rather than multiplying by its inverse, which keeps black and white on 0 and 1.
pub fn codes_to_float<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    codes: FloatCodes,
) {
    let samples = output.samples();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and the output each hold `samples` f32 values.
    unsafe {
        codes_to_float_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            codes.offset,
            codes.scale,
        );
    }
}
