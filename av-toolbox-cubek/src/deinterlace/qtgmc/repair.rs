// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

#[cube(launch_unchecked)]
fn vertical_blur_kernel(input: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = (position / width) as i32;

    let last = height as i32 - 1;
    let above = i32::clamp(row - 1, 0, last) as u32;
    let below = i32::clamp(row + 1, 0, last) as u32;
    let top = input[(above * width + column) as usize];
    let centre = input[ABSOLUTE_POS];
    let bottom = input[(below * width + column) as usize];

    output[ABSOLUTE_POS] = (top + 2.0f32 * centre + bottom) * 0.25f32;
}

#[cube(launch_unchecked)]
fn repair_blend_kernel(
    original: &Array<f32>,
    rebuilt: &Array<f32>,
    mask: &Array<f32>,
    output: &mut Array<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let kept = original[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = kept + (rebuilt[ABSOLUTE_POS] - kept) * mask[ABSOLUTE_POS];
}

/// `(above + 2 × centre + below) / 4`, the border clamped.
///
/// This is `std.Convolution(matrix=[1, 2, 1], mode='v')` without its rounding.
pub fn vertical_blur<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let (width, height) = (output.width(), output.height());
    assert_plane(&input, width, height, "input");
    assert_plane(&output, width, height, "output");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let input_array = input.array::<R>();
    let output_array = output.array::<R>();

    // Safety: both planes hold `width` by `height` f32 samples, and every row read is clamped inside
    // the plane.
    unsafe {
        vertical_blur_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            width,
            height,
        );
    }
}

/// `original + (rebuilt − original) × mask`.
pub fn repair_blend<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    original: DevicePlane<'_>,
    rebuilt: DevicePlane<'_>,
    mask: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let (width, height) = (output.width(), output.height());
    assert_plane(&original, width, height, "original");
    assert_plane(&rebuilt, width, height, "rebuilt");
    assert_plane(&mask, width, height, "mask");
    assert_plane(&output, width, height, "output");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let original_array = original.array::<R>();
    let rebuilt_array = rebuilt.array::<R>();
    let mask_array = mask.array::<R>();
    let output_array = output.array::<R>();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        repair_blend_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            original_array,
            rebuilt_array,
            mask_array,
            output_array,
        );
    }
}
