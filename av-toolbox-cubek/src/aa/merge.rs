// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

#[cube(launch_unchecked)]
fn merge_fields_kernel(first: &Array<f32>, second: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let total = first[ABSOLUTE_POS] + second[ABSOLUTE_POS] + 1.0;
    output[ABSOLUTE_POS] = f32::floor(total * 0.5);
}

/// Averages two planes of integer code values as `floor((first + second + 1) / 2)`.
///
/// The result is exact for values held in f32 up to 65535. `output` may be `first`, which merges in place.
pub fn merge_fields<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    first: DevicePlane<'_>,
    second: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let width = output.width();
    let height = output.height();

    assert!(
        first.width() == width && first.height() == height,
        "the first plane is {}x{}, expected {width}x{height}",
        first.width(),
        first.height(),
    );
    assert!(
        second.width() == width && second.height() == height,
        "the second plane is {}x{}, expected {width}x{height}",
        second.width(),
        second.height(),
    );

    let samples = output.samples();
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: all three planes hold `samples` elements, and each unit reads only its own sample
    // before writing it.
    unsafe {
        merge_fields_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            first_array,
            second_array,
            output_array,
        );
    }
}
