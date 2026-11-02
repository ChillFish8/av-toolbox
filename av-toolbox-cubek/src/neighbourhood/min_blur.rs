// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{NeighbourPlanes, Smooth, median, smooth};
use crate::blur::{FilterPlanes, Separable, Storage};
use crate::plane::assert_plane;
use crate::{DevicePlane, Dispatch, Edge};

/// The 5 taps a binomial then one box pass add up to.
const RADIUS_2_WEIGHTS: [f32; 5] = [1.0, 3.0, 4.0, 3.0, 1.0];
/// The 7 taps a binomial then two box passes add up to.
const RADIUS_3_WEIGHTS: [f32; 7] = [1.0, 4.0, 8.0, 10.0, 8.0, 4.0, 1.0];

/// The planes one MinBlur reads, passes through and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MinBlurPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub median: DevicePlane<'a>,
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
fn pick_kernel(input: &[f32], blurred: &[f32], median: &[f32], output: &mut [f32]) {
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

/// A 3x3 binomial blur followed by `radius - 1` 3x3 box blurs, as a single pass.
///
/// Radius 1 runs the binomial alone. Radius 2 and 3 run the 5 and 7 tap
/// [Separable](crate::blur::Separable) the chain adds up to. Edges reflect.
#[derive(Debug)]
pub struct BinomialChain {
    radius: u32,
    separable: Option<Separable>,
}

impl BinomialChain {
    /// Builds the chain for `radius`, which is 1, 2 or 3.
    ///
    /// It panics when `radius` is outside 1..=3.
    pub fn new(client: &Client, radius: u32) -> Self {
        let separable = match radius {
            1 => None,
            2 => Some(Separable::new(client, &RADIUS_2_WEIGHTS)),
            3 => Some(Separable::new(client, &RADIUS_3_WEIGHTS)),
            _ => panic!("a binomial chain has radius 1, 2 or 3, not {radius}"),
        };

        Self { radius, separable }
    }

    pub const fn radius(&self) -> u32 {
        self.radius
    }

    /// Blurs `planes.input` into `planes.output`, both f32.
    pub fn apply(&self, client: &Client, dispatch: &impl Dispatch, planes: NeighbourPlanes<'_>) {
        let Some(separable) = &self.separable else {
            smooth(client, dispatch, planes, Smooth::Binomial);
            return;
        };

        let filter_planes = FilterPlanes {
            input: planes.input,
            output: planes.output,
        };
        separable.apply_as(client, dispatch, filter_planes, Edge::Reflect, Storage::F32);
    }
}

/// Runs havsfunc's MinBlur, picking the median of each sample, its blur and its median.
///
/// `blur` sets the radius and runs into `blurred`. The median runs over the same radius, into
/// `median`.
pub fn min_blur(client: &Client, dispatch: &impl Dispatch, planes: MinBlurPlanes<'_>, blur: &BinomialChain) {
    let blur_planes = NeighbourPlanes {
        input: planes.input,
        output: planes.blurred,
    };
    blur.apply(client, dispatch, blur_planes);

    let median_planes = NeighbourPlanes {
        input: planes.input,
        output: planes.median,
    };
    median(client, dispatch, median_planes, blur.radius());

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
pub fn median_of_three(
    client: &Client,
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
    let first_array = first.array();
    let second_array = second.array();
    let third_array = third.array();
    let output_array = output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the asserted sizes make every plane hold `samples` f32 values.
    unsafe {
        pick_kernel::launch_unchecked(
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
