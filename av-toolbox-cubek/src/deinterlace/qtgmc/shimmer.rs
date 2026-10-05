// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::neighbourhood::{ChainPlanes, Edge, Extreme, Shape, Spread, Step, chain_with_edge};
use crate::{DevicePlane, Dispatch, assert_plane};

/// The passes that erode positive differences, then grow what survives back.
pub(crate) const POSITIVE_STEPS: [Step; 5] = [
    Step::Extreme(Extreme::Minimum, Shape::Vertical),
    Step::Extreme(Extreme::Minimum, Shape::Vertical),
    Step::Spread(Spread::Deflate),
    Step::Extreme(Extreme::Maximum, Shape::Vertical),
    Step::Extreme(Extreme::Maximum, Shape::Vertical),
];

/// The passes that erode negative differences, then grow what survives back.
pub(crate) const NEGATIVE_STEPS: [Step; 5] = [
    Step::Extreme(Extreme::Maximum, Shape::Vertical),
    Step::Extreme(Extreme::Maximum, Shape::Vertical),
    Step::Spread(Spread::Inflate),
    Step::Extreme(Extreme::Minimum, Shape::Vertical),
    Step::Extreme(Extreme::Minimum, Shape::Vertical),
];

/// The scratch planes one shimmer repair passes through, all one size.
pub struct ShimmerPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub reference: DevicePlane<'a>,
    pub difference: DevicePlane<'a>,
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    pub choke: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The code value of a level given for 8 bit video, as havsfunc's `scale` rounds it.
///
/// havsfunc defines `scale(value, peak)` as `cround(value * peak / 255)` for a `peak` other than 1, and
/// `cround` rounds half away from zero.
fn scale(level: f32, peak: f32) -> f32 {
    let scaled = level * peak / 255.0;

    if scaled > 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    }
}

/// The signed differences below and above which a difference is left as it is, for `bits` deep samples.
///
/// Positive differences under the first and negative differences over the second are restored.
pub(crate) fn restore_limits(bits: u32) -> (f32, f32) {
    let neutral = (1u32 << (bits - 1)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;

    (scale(129.0, peak) - neutral, scale(127.0, peak) - neutral)
}

#[cube(launch_unchecked)]
fn difference_kernel(
    input: &Array<f32>,
    reference: &Array<f32>,
    output: &mut Array<f32>,
    lowest: f32,
    highest: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let difference = reference[ABSOLUTE_POS] - input[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = f32::clamp(difference, lowest, highest);
}

#[cube(launch_unchecked)]
fn restore_kernel(
    input: &Array<f32>,
    difference: &Array<f32>,
    positive: &Array<f32>,
    negative: &Array<f32>,
    output: &mut Array<f32>,
    above: f32,
    below: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let original = difference[ABSOLUTE_POS];

    let mut restored = original;
    if original >= above {
        restored = f32::max(positive[ABSOLUTE_POS], 0.0);
    }

    let mut result = restored;
    if restored <= below {
        result = f32::min(negative[ABSOLUTE_POS], 0.0);
    }

    output[ABSOLUTE_POS] = f32::clamp(input[ABSOLUTE_POS] + result, 0.0, peak);
}

/// Keeps only thin horizontal differences between `input` and `reference`, as QTGMC's
/// `KeepOnlyBobShimmerFixes` does at Rep 4.
///
/// Differences more than a couple of rows high are restored to `reference`. `bits` is the sample depth
/// the planes hold values for.
pub fn shimmer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ShimmerPlanes<'_>,
    bits: u32,
) {
    assert!((1..=16).contains(&bits), "{bits} bits, expected 1..=16");

    let width = planes.input.width();
    let height = planes.input.height();
    assert_plane(&planes.input, width, height, "input");
    assert_plane(&planes.reference, width, height, "reference");
    assert_plane(&planes.difference, width, height, "difference");
    assert_plane(&planes.first, width, height, "first");
    assert_plane(&planes.second, width, height, "second");
    assert_plane(&planes.choke, width, height, "choke");
    assert_plane(&planes.output, width, height, "output");

    let neutral = (1u32 << (bits - 1)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;
    let (above, below) = restore_limits(bits);
    let cubes = dispatch.cube_count(planes.output.samples());
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        difference_kernel::launch_unchecked::<R>(
            client,
            cubes.clone(),
            unit,
            planes.input.array::<R>(),
            planes.reference.array::<R>(),
            planes.difference.array::<R>(),
            -neutral,
            peak - neutral,
        );
    }

    let positive = ChainPlanes {
        input: planes.difference,
        scratch: planes.first,
        output: planes.choke,
    };
    chain_with_edge(client, dispatch, positive, &POSITIVE_STEPS, Edge::Clamp);

    let negative = ChainPlanes {
        input: planes.difference,
        scratch: planes.first,
        output: planes.second,
    };
    chain_with_edge(client, dispatch, negative, &NEGATIVE_STEPS, Edge::Clamp);

    // Safety: as above, and the choke planes are only read.
    unsafe {
        restore_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            planes.input.array::<R>(),
            planes.difference.array::<R>(),
            planes.choke.array::<R>(),
            planes.second.array::<R>(),
            planes.output.array::<R>(),
            above,
            below,
            peak,
        );
    }
}
