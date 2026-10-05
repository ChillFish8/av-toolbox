// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{assert_bits, clamped_difference, havsfunc_scale};
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

/// The planes one shimmer repair reads, passes through and writes, all one size and all separate.
pub struct ShimmerPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub reference: DevicePlane<'a>,
    pub difference: DevicePlane<'a>,
    pub first: DevicePlane<'a>,
    pub positive_choke: DevicePlane<'a>,
    pub negative_choke: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

impl ShimmerPlanes<'_> {
    /// Whether every plane is held by a separate handle.
    fn are_distinct(&self) -> bool {
        let handles = [
            self.input.handle(),
            self.reference.handle(),
            self.difference.handle(),
            self.first.handle(),
            self.positive_choke.handle(),
            self.negative_choke.handle(),
            self.output.handle(),
        ];

        handles.iter().enumerate().all(|(index, handle)| {
            handles[index + 1..]
                .iter()
                .all(|other| !std::ptr::eq(*handle, *other))
        })
    }
}

/// The signed differences below and above which a difference is left as it is, for `bits` deep samples.
///
/// Positive differences under the first and negative differences over the second are restored.
pub(crate) fn restore_limits(bits: u32) -> (f32, f32) {
    let neutral = (1u32 << (bits - 1)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;

    (
        havsfunc_scale(129.0, peak) - neutral,
        havsfunc_scale(127.0, peak) - neutral,
    )
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
/// the planes hold values for, between 8 and 16.
pub fn shimmer<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ShimmerPlanes<'_>,
    bits: u32,
) {
    assert_bits(bits);

    let width = planes.input.width();
    let height = planes.input.height();
    assert_plane(&planes.input, width, height, "input");
    assert_plane(&planes.reference, width, height, "reference");
    assert_plane(&planes.difference, width, height, "difference");
    assert_plane(&planes.first, width, height, "first");
    assert_plane(&planes.positive_choke, width, height, "positive choke");
    assert_plane(&planes.negative_choke, width, height, "negative choke");
    assert_plane(&planes.output, width, height, "output");
    debug_assert!(planes.are_distinct(), "the shimmer planes must all be separate");

    let peak = ((1u32 << bits) - 1) as f32;
    let (above, below) = restore_limits(bits);
    let cubes = dispatch.cube_count(planes.output.samples());
    let unit = dispatch.cube_dim();

    clamped_difference(
        client,
        dispatch,
        planes.reference,
        planes.input,
        planes.difference,
        bits,
    );

    let positive = ChainPlanes {
        input: planes.difference,
        scratch: planes.first,
        output: planes.positive_choke,
    };
    chain_with_edge(client, dispatch, positive, &POSITIVE_STEPS, Edge::Clamp);

    let negative = ChainPlanes {
        input: planes.difference,
        scratch: planes.first,
        output: planes.negative_choke,
    };
    chain_with_edge(client, dispatch, negative, &NEGATIVE_STEPS, Edge::Clamp);

    let input = planes.input.array::<R>();
    let difference = planes.difference.array::<R>();
    let positive_choke = planes.positive_choke.array::<R>();
    let negative_choke = planes.negative_choke.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: every plane holds `width` by `height` f32 samples, each unit reads and writes only its own
    // sample, and the choke planes are only read.
    unsafe {
        restore_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            difference,
            positive_choke,
            negative_choke,
            output,
            above,
            below,
            peak,
        );
    }
}
