// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{NeighbourPlanes, Smooth, median, smooth};
use crate::plane::assert_plane;
use crate::{DevicePlane, Dispatch};

/// The planes one MinBlur reads, passes through and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MinBlurPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub median: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The median of the source, its blur and its median, the way MinBlur picks.
pub fn median_of_three_sample(source: f32, blurred: f32, median: f32) -> f32 {
    let toward_blur = source - blurred;
    let toward_median = source - median;

    if toward_blur * toward_median < 0.0 {
        return source;
    }

    if toward_blur.abs() < toward_median.abs() {
        blurred
    } else {
        median
    }
}

#[cube(launch_unchecked)]
fn pick_kernel(input: &Array<f32>, blurred: &Array<f32>, median: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let source = input[ABSOLUTE_POS];
    let smoothed = blurred[ABSOLUTE_POS];
    let middle = median[ABSOLUTE_POS];
    let toward_blur = source - smoothed;
    let toward_median = source - middle;
    let blur_distance = f32::abs(toward_blur);
    let median_distance = f32::abs(toward_median);

    let mut picked = middle;
    if toward_blur * toward_median < 0.0 {
        picked = source;
    } else if blur_distance < median_distance {
        picked = smoothed;
    }

    output[ABSOLUTE_POS] = picked;
}

/// Runs havsfunc's MinBlur, picking the median of each sample, its blur and its median.
///
/// `radius` is 1, 2 or 3. The blur chain runs into `blurred`, using `scratch` as the plane between
/// its passes. It runs Binomial alone for radius 1, Binomial then Box for radius 2, and Binomial
/// then Box twice for radius 3. The median runs over the same radius, into `median`.
pub fn min_blur<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MinBlurPlanes<'_>,
    radius: u32,
) {
    debug_assert!((1..=3).contains(&radius));

    match radius {
        1 => {
            let blur_planes = NeighbourPlanes {
                input: planes.input,
                output: planes.blurred,
            };
            smooth(client, dispatch, blur_planes, Smooth::Binomial);
        },
        2 => {
            let binomial_planes = NeighbourPlanes {
                input: planes.input,
                output: planes.scratch,
            };
            smooth(client, dispatch, binomial_planes, Smooth::Binomial);

            let box_planes = NeighbourPlanes {
                input: planes.scratch,
                output: planes.blurred,
            };
            smooth(client, dispatch, box_planes, Smooth::Box);
        },
        3 => {
            let binomial_planes = NeighbourPlanes {
                input: planes.input,
                output: planes.blurred,
            };
            smooth(client, dispatch, binomial_planes, Smooth::Binomial);

            let first_box_planes = NeighbourPlanes {
                input: planes.blurred,
                output: planes.scratch,
            };
            smooth(client, dispatch, first_box_planes, Smooth::Box);

            let second_box_planes = NeighbourPlanes {
                input: planes.scratch,
                output: planes.blurred,
            };
            smooth(client, dispatch, second_box_planes, Smooth::Box);
        },
        _ => unreachable!("radius is 1, 2 or 3"),
    }

    let median_planes = NeighbourPlanes {
        input: planes.input,
        output: planes.median,
    };
    median(client, dispatch, median_planes, radius);

    median_of_three(
        client,
        dispatch,
        planes.input,
        planes.blurred,
        planes.median,
        planes.output,
    );
}

/// Writes the median of `first`, `second` and `third` at each sample.
pub fn median_of_three<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    first: DevicePlane<'_>,
    second: DevicePlane<'_>,
    third: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let width = output.width();
    let height = output.height();

    assert_plane(&first, width, height, "first");
    assert_plane(&second, width, height, "second");
    assert_plane(&third, width, height, "third");
    assert_plane(&output, width, height, "output");

    let samples = output.samples();
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let third_array = third.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the asserted sizes make every plane hold `samples` f32 values.
    unsafe {
        pick_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            first_array,
            second_array,
            third_array,
            output_array,
        );
    }
}
