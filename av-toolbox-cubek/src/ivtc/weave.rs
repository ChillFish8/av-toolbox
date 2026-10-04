// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

#[cube(launch_unchecked)]
fn weave_kernel(
    matching: &Array<f32>,
    other: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    matching_parity: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let row = ABSOLUTE_POS as u32 / width;
    if row % 2 == matching_parity {
        output[ABSOLUTE_POS] = matching[ABSOLUTE_POS];
    } else {
        output[ABSOLUTE_POS] = other[ABSOLUTE_POS];
    }
}

/// The planes one weave reads and writes, f32 code values of one size.
pub struct WeavePlanes<'a> {
    pub matching: DevicePlane<'a>,
    pub other: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// Builds `planes.output` from the rows of parity `matching_parity` of `planes.matching` and the other
/// rows of `planes.other`.
pub fn weave<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: WeavePlanes<'_>,
    matching_parity: u32,
) {
    let width = planes.output.width();
    let height = planes.output.height();
    assert_plane(&planes.matching, width, height, "matching");
    assert_plane(&planes.other, width, height, "other");
    assert_plane(&planes.output, width, height, "output");

    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let matching = planes.matching.array::<R>();
    let other = planes.other.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: all three planes hold `samples` f32 values, and each unit reads and writes only its own sample.
    unsafe {
        weave_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            matching,
            other,
            output,
            width,
            matching_parity,
        );
    }
}
