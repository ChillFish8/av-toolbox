// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// The most neighbours [temporal_limits] takes.
const MAX_NEIGHBOURS: usize = 2;

#[cube(launch_unchecked)]
fn limits_kernel(
    centre: &[f32],
    neighbours: &Sequence<Box<[f32]>>,
    upper: &mut [f32],
    lower: &mut [f32],
    #[comptime] count: u32,
) {
    if ABSOLUTE_POS >= upper.len() {
        terminate!();
    }

    let mut largest = centre[ABSOLUTE_POS];
    let mut smallest = largest;

    #[unroll]
    for neighbour in 0..count {
        let sample = neighbours.index(neighbour as usize)[ABSOLUTE_POS];
        largest = f32::max(largest, sample);
        smallest = f32::min(smallest, sample);
    }

    upper[ABSOLUTE_POS] = largest;
    lower[ABSOLUTE_POS] = smallest;
}

/// The per-sample maximum and minimum of `centre` and its compensated neighbours.
///
/// There may be up to 2 neighbours. With none, both limits equal `centre`.
pub fn temporal_limits(
    client: &Client,
    dispatch: &impl Dispatch,
    centre: DevicePlane<'_>,
    neighbours: &[DevicePlane<'_>],
    upper: DevicePlane<'_>,
    lower: DevicePlane<'_>,
) {
    assert!(
        neighbours.len() <= MAX_NEIGHBOURS,
        "{} neighbours, expected at most {MAX_NEIGHBOURS}",
        neighbours.len()
    );

    let (width, height) = (centre.width(), centre.height());
    assert_plane(&centre, width, height, "centre");
    assert_plane(&upper, width, height, "upper");
    assert_plane(&lower, width, height, "lower");

    let mut neighbour_args = SequenceArg::<Box<[f32]>>::new();
    for neighbour in neighbours {
        assert_plane(neighbour, width, height, "neighbour");

        neighbour_args.push(neighbour.array());
    }

    let cubes = dispatch.cube_count(centre.samples());
    let unit = dispatch.cube_dim();
    let centre_array = centre.array();
    let upper_array = upper.array();
    let lower_array = lower.array();

    // Safety: every plane holds `width` by `height` f32 samples, and each unit reads and writes only its
    // own sample.
    unsafe {
        limits_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            centre_array,
            neighbour_args,
            upper_array,
            lower_array,
            neighbours.len() as u32,
        );
    }
}
