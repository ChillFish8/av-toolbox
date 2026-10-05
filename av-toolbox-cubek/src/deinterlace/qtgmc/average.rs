// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// The largest number of planes one weighted sum takes.
pub const MAX_AVERAGED: usize = 5;

/// The weight `Merge(ts1, ts2, ...)` gives `ts2` at TR0 2.
const TS2_WEIGHT: f32 = 0.357;

/// The weight the bob takes when merged with `ts1` at TR0 1.
const BOB_WEIGHT_TR0_1: f32 = 0.25;
/// The weight the bob takes when merged with the `ts1` and `ts2` blend at TR0 2.
const BOB_WEIGHT_TR0_2: f32 = 0.125;

/// The TR0 weights of bobs -2..=2 for `tr0`, combining `ts1`, `ts2` and the bob as QTGMC merges them.
///
/// `ts1` averages bobs -1..=1 and `ts2` averages bobs -2..=2. A `tr0` of 0 keeps the centre bob alone, and
/// anything above 2 weighs as 2.
pub fn tr0_weights(tr0: u32) -> [f32; MAX_AVERAGED] {
    let ts1 = [0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0];
    let ts2 = [0.2; MAX_AVERAGED];
    let bob = [0.0, 0.0, 1.0, 0.0, 0.0];

    let mut weights = [0.0f32; MAX_AVERAGED];
    for (index, weight) in weights.iter_mut().enumerate() {
        *weight = match tr0 {
            0 => bob[index],
            1 => ts1[index] * (1.0 - BOB_WEIGHT_TR0_1) + bob[index] * BOB_WEIGHT_TR0_1,
            _ => {
                let smoothed = ts1[index] * (1.0 - TS2_WEIGHT) + ts2[index] * TS2_WEIGHT;

                smoothed * (1.0 - BOB_WEIGHT_TR0_2) + bob[index] * BOB_WEIGHT_TR0_2
            },
        };
    }

    weights
}

#[cube(launch_unchecked)]
fn weighted_sum_kernel(
    planes: &Sequence<Array<f32>>,
    weights: &Sequence<f32>,
    output: &mut Array<f32>,
    #[comptime] count: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let mut total = 0.0f32;

    #[unroll]
    for plane in 0..count {
        let index = plane as usize;
        total += planes.index(index)[ABSOLUTE_POS] * *weights.index(index);
    }

    output[ABSOLUTE_POS] = total;
}

/// Writes `Σ weights[i] × planes[i]` into `output`.
///
/// The same plane may appear more than once, to weigh it again.
pub fn weighted_sum<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: &[DevicePlane<'_>],
    weights: &[f32],
    output: DevicePlane<'_>,
) {
    assert!(
        (1..=MAX_AVERAGED).contains(&planes.len()),
        "{} planes, expected 1..={MAX_AVERAGED}",
        planes.len()
    );
    assert_eq!(
        planes.len(),
        weights.len(),
        "{} planes with {} weights",
        planes.len(),
        weights.len()
    );

    let width = output.width();
    let height = output.height();
    assert_plane(&output, width, height, "output");

    let mut plane_args = SequenceArg::<R, Array<f32>>::new();
    let mut weight_args = SequenceArg::<R, f32>::new();
    for (plane, weight) in planes.iter().zip(weights) {
        assert_plane(plane, width, height, "plane");

        plane_args.push(plane.array::<R>());
        weight_args.push(*weight);
    }

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let output_array = output.array::<R>();

    // Safety: every plane and the output hold `width` by `height` f32 samples, and each unit reads and
    // writes only its own sample.
    unsafe {
        weighted_sum_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            plane_args,
            weight_args,
            output_array,
            planes.len() as u32,
        );
    }
}
