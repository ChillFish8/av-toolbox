// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::depth::Element;
use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn copy_kernel<E: Int>(input: &[E], output: &mut [E], available: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    if position < available {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    } else {
        output[ABSOLUTE_POS] = E::cast_from(0u32);
    }
}

/// Copies `input` into `output` sample by sample, zeroing every sample of `output` past the end of `input`.
///
/// `output` may be larger than `input`, to pad a plane out to whole words, or smaller, to take a padded
/// plane back to its own size.
pub fn copy_plane(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    element: Element,
) {
    match element {
        Element::U8 => launch_copy::<u8>(client, dispatch, input, output),
        Element::U16 => launch_copy::<u16>(client, dispatch, input, output),
    }
}

fn launch_copy<E: Int>(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let samples = output.samples();
    let available = input.samples() as u32;
    let input_array = input.array();
    let output_array = output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the kernel reads `input` only below its own sample count, and writes `output` only below its own.
    unsafe {
        copy_kernel::launch_unchecked::<E>(client, cubes, unit, input_array, output_array, available);
    }
}
