// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{NeighbourPlanes, read};
use crate::Dispatch;

/// Paeth's median network for 9 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_9_AT].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_9: [(usize, usize); 19] = [
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 1),
    (3, 4),
    (6, 7),
    (1, 2),
    (4, 5),
    (7, 8),
    (0, 3),
    (5, 8),
    (4, 7),
    (3, 6),
    (1, 4),
    (2, 5),
    (4, 7),
    (4, 2),
    (6, 4),
    (4, 2),
];

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_9_AT: usize = 4;

/// Devillard's `opt_med25` median network for 25 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_25_AT].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_25: [(usize, usize); 99] = [
    (0, 1),
    (3, 4),
    (2, 4),
    (2, 3),
    (6, 7),
    (5, 7),
    (5, 6),
    (9, 10),
    (8, 10),
    (8, 9),
    (12, 13),
    (11, 13),
    (11, 12),
    (15, 16),
    (14, 16),
    (14, 15),
    (18, 19),
    (17, 19),
    (17, 18),
    (21, 22),
    (20, 22),
    (20, 21),
    (23, 24),
    (2, 5),
    (3, 6),
    (0, 6),
    (0, 3),
    (4, 7),
    (1, 7),
    (1, 4),
    (11, 14),
    (8, 14),
    (8, 11),
    (12, 15),
    (9, 15),
    (9, 12),
    (13, 16),
    (10, 16),
    (10, 13),
    (20, 23),
    (17, 23),
    (17, 20),
    (21, 24),
    (18, 24),
    (18, 21),
    (19, 22),
    (8, 17),
    (9, 18),
    (0, 18),
    (0, 9),
    (10, 19),
    (1, 19),
    (1, 10),
    (11, 20),
    (2, 20),
    (2, 11),
    (12, 21),
    (3, 21),
    (3, 12),
    (13, 22),
    (4, 22),
    (4, 13),
    (14, 23),
    (5, 23),
    (5, 14),
    (15, 24),
    (6, 24),
    (6, 15),
    (7, 16),
    (7, 19),
    (13, 21),
    (15, 23),
    (7, 13),
    (7, 15),
    (1, 9),
    (3, 11),
    (5, 17),
    (11, 17),
    (9, 17),
    (4, 10),
    (6, 12),
    (7, 14),
    (4, 6),
    (4, 7),
    (12, 14),
    (10, 14),
    (6, 7),
    (10, 12),
    (6, 10),
    (6, 17),
    (12, 17),
    (7, 17),
    (7, 10),
    (12, 18),
    (7, 12),
    (10, 18),
    (12, 20),
    (10, 20),
    (10, 12),
];

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_25_AT: usize = 12;

/// Forgetful selection over 49 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_49_AT].
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_49: [(usize, usize); 624] = forgetful_selection_of_49();

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
pub(crate) const MEDIAN_OF_49_AT: usize = 48;

/// Builds forgetful selection over 49 values.
///
/// It starts with the first 26 values. Each round moves the smallest of the values in play to the front and the
/// largest to the slot after it, drops both, and brings in the next value, until one value is left.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the kernel selects through these from the next change")
)]
const fn forgetful_selection_of_49() -> [(usize, usize); 624] {
    let mut pairs = [(0usize, 0usize); 624];
    let mut written = 0;
    let mut start = 0;
    let mut end = 26;
    let mut finished = false;

    while !finished {
        let mut index = start + 1;
        while index < end {
            pairs[written] = (start, index);
            written += 1;
            index += 1;
        }

        let mut index = start + 2;
        while index < end {
            pairs[written] = (index, start + 1);
            written += 1;
            index += 1;
        }

        start += 2;
        if end == 49 {
            finished = true;
        } else {
            end += 1;
        }
    }

    assert!(written == 624 && start == MEDIAN_OF_49_AT);

    pairs
}

#[cube]
fn window_value(
    input: &Array<f32>,
    width: u32,
    height: u32,
    x: i32,
    y: i32,
    size: u32,
    radius: i32,
    slot: u32,
) -> f32 {
    let offset_x = (slot % size) as i32 - radius;
    let offset_y = (slot / size) as i32 - radius;

    read(input, width, height, x + offset_x, y + offset_y)
}

#[cube(launch_unchecked)]
fn median_kernel(input: &Array<f32>, output: &mut Array<f32>, width: u32, height: u32, radius: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let size = 2 * radius + 1;
    let count = size * size;
    let middle = count / 2;
    let reach = radius as i32;

    let mut chosen = 0.0f32;
    let mut found: u32 = 0u32;

    for candidate_slot in 0..count {
        let candidate = window_value(input, width, height, x, y, size, reach, candidate_slot);
        let mut below = 0u32;
        let mut equal = 0u32;

        for other_slot in 0..count {
            let other = window_value(input, width, height, x, y, size, reach, other_slot);

            if other < candidate {
                below += 1;
            } else if other == candidate {
                equal += 1;
            }
        }

        if found == 0u32 && below <= middle && middle < below + equal {
            chosen = candidate;
            found = 1u32;
        }
    }

    output[ABSOLUTE_POS] = chosen;
}

/// Writes the median of each sample's square window reaching `radius` samples each side.
///
/// `radius` is 1, 2 or 3, for a 3x3, 5x5 or 7x7 window.
pub fn median<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
    radius: u32,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, and every read is reflected inside the plane.
    unsafe {
        median_kernel::launch_unchecked::<R>(client, cubes, unit, input, output, width, height, radius);
    }
}
