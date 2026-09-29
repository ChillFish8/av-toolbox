mod mask;
mod range;

use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::neighbourhood::{
    Expansion,
    Extreme,
    NeighbourPlanes,
    Shape,
    expansion_passes,
    extreme,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

pub(crate) use self::mask::DetailMask;
pub(crate) use self::range::Range;
use crate::sizing::Sizing;

/// Runs `passes` of `which` from `input` into `target`, bouncing through `scratch`.
///
/// The passes alternate between `scratch` and `target` so a pass never reads and writes the same
/// buffer, and the last pass always lands in `target`.
fn expand_chain<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    input: DevicePlane<'_>,
    scratch: DevicePlane<'_>,
    target: DevicePlane<'_>,
    passes: &[Shape],
    which: Extreme,
) {
    let total = passes.len();
    let mut source = input;

    for (index, shape) in passes.iter().enumerate() {
        let steps_remaining = total - 1 - index;
        let destination = if steps_remaining.is_multiple_of(2) {
            target
        } else {
            scratch
        };

        let planes = NeighbourPlanes {
            input: source,
            output: destination,
        };
        extreme(client, sizing, planes, which, *shape);

        source = destination;
    }
}

#[cube(launch_unchecked)]
fn subtract_kernel(maximum: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = maximum[ABSOLUTE_POS] - output[ABSOLUTE_POS];
}

/// Writes `maximum - output` into `output`.
fn subtract<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    maximum: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let samples = output.samples();
    let read = maximum.array::<R>();
    let written = output.array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: `maximum` and `output` each hold `samples` f32 values.
    unsafe {
        subtract_kernel::launch_unchecked::<R>(client, cubes, shape, read, written);
    }
}

#[cube(launch_unchecked)]
fn join_maximum_kernel(first: &Array<f32>, second: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = f32::max(first[ABSOLUTE_POS], second[ABSOLUTE_POS]);
}

/// Writes the larger of `first` and `second` at each sample into `output`.
fn join_maximum<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    first: DevicePlane<'_>,
    second: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let samples = output.samples();
    let first_array = first.array::<R>();
    let second_array = second.array::<R>();
    let written = output.array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: `first`, `second` and `output` each hold `samples` f32 values.
    unsafe {
        join_maximum_kernel::launch_unchecked::<R>(client, cubes, shape, first_array, second_array, written);
    }
}

/// Writes the local maximum minus the local minimum of `input` over `radius` into `output`.
///
/// Overwrites `scratch`, `maximum` and `output`, and leaves `input` untouched. `radius` must be at
/// least 1, since with no passes nothing is written.
pub(crate) fn range<R: Runtime>(
    client: &ComputeClient<R>,
    sizing: &Sizing<'_>,
    input: DevicePlane<'_>,
    scratch: DevicePlane<'_>,
    maximum: DevicePlane<'_>,
    output: DevicePlane<'_>,
    radius: u32,
) {
    debug_assert!(radius >= 1, "range needs a radius of at least 1");

    let passes = expansion_passes(radius, radius, Expansion::Ellipse);

    expand_chain(client, sizing, input, scratch, maximum, &passes, Extreme::Maximum);
    expand_chain(client, sizing, input, scratch, output, &passes, Extreme::Minimum);
    subtract(client, sizing, maximum, output);
}
