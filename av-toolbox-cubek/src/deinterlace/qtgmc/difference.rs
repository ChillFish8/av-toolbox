// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::assert_bits;
use crate::{DevicePlane, Dispatch, assert_plane};

#[cube(launch_unchecked)]
fn difference_kernel(
    first: &Array<f32>,
    second: &Array<f32>,
    output: &mut Array<f32>,
    lowest: f32,
    highest: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let difference = first[ABSOLUTE_POS] - second[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = f32::clamp(difference, lowest, highest);
}

/// `first − second` clamped to `−(peak + 1) / 2..=(peak − 1) / 2`, as `std.MakeDiff` stores it.
///
/// `peak` is the largest code value of `bits` deep samples, and `bits` is between 8 and 16.
pub fn clamped_difference<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    first: DevicePlane<'_>,
    second: DevicePlane<'_>,
    output: DevicePlane<'_>,
    bits: u32,
) {
    assert_bits(bits);

    let (width, height) = (output.width(), output.height());
    assert_plane(&first, width, height, "first");
    assert_plane(&second, width, height, "second");
    assert_plane(&output, width, height, "output");

    let neutral = (1u32 << (bits - 1)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let output_array = output.array::<R>();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        difference_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            first_array,
            second_array,
            output_array,
            -neutral,
            peak - neutral,
        );
    }
}
