// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::blur::{self, FilterPlanes, Storage};
use crate::{DevicePlane, Dispatch, Edge, assert_plane};

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
    let planes = FilterPlanes { input, output };

    blur::vertical_121(client, dispatch, planes, Edge::Clamp, Storage::F32);
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
