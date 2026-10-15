// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::NeighbourPlanes;
use crate::{Dispatch, reflect};

/// Paeth's median network for 9 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_9_AT].
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

pub(crate) const MEDIAN_OF_9_AT: usize = 4;

/// Devillard's `opt_med25` median network for 25 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_25_AT].
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

pub(crate) const MEDIAN_OF_25_AT: usize = 12;

/// Forgetful selection over 49 values, as pairs whose smaller value goes to the first index.
///
/// The median ends at [MEDIAN_OF_49_AT].
pub(crate) const MEDIAN_OF_49: [(usize, usize); 624] = forgetful_selection_of_49();

pub(crate) const MEDIAN_OF_49_AT: usize = 48;

/// Builds forgetful selection over 49 values.
///
/// It starts with the first 26 values. Each round moves the smallest of the values in play to the front and the
/// largest to the slot after it, drops both, and brings in the next value, until one value is left.
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

/// Puts the smaller of two window values at `first` and the larger at `second`.
#[cube]
fn exchange(window: &mut Array<f32>, #[comptime] first: usize, #[comptime] second: usize) {
    let first_value = window[first];
    let second_value = window[second];

    window[first] = f32::min(first_value, second_value);
    window[second] = f32::max(first_value, second_value);
}

/// Runs the exchange network for a window `radius` samples each side, returning the median.
#[expect(
    clippy::needless_range_loop,
    reason = "the unrolled loops read the exchange tables when the kernel compiles"
)]
#[cube]
fn select_median(window: &mut Array<f32>, #[comptime] radius: u32) -> f32 {
    if comptime!(radius == 1) {
        #[unroll]
        for step in 0..MEDIAN_OF_9.len() {
            exchange(
                window,
                comptime!(MEDIAN_OF_9[step].0),
                comptime!(MEDIAN_OF_9[step].1),
            );
        }

        window[MEDIAN_OF_9_AT]
    } else if comptime!(radius == 2) {
        #[unroll]
        for step in 0..MEDIAN_OF_25.len() {
            exchange(
                window,
                comptime!(MEDIAN_OF_25[step].0),
                comptime!(MEDIAN_OF_25[step].1),
            );
        }

        window[MEDIAN_OF_25_AT]
    } else {
        #[unroll]
        for step in 0..MEDIAN_OF_49.len() {
            exchange(
                window,
                comptime!(MEDIAN_OF_49[step].0),
                comptime!(MEDIAN_OF_49[step].1),
            );
        }

        window[MEDIAN_OF_49_AT]
    }
}

#[cube(launch_unchecked)]
fn median_kernel(
    input: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    #[comptime] radius: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let size = comptime!(2 * radius + 1);
    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let reach = comptime!(radius as i32);

    let mut columns = Array::<u32>::new(comptime!(size as usize));
    let mut row_starts = Array::<u32>::new(comptime!(size as usize));

    #[unroll]
    for offset in 0..size {
        let shift = comptime!(offset as i32 - reach);
        let column = reflect(x + shift, width as i32);
        let row = reflect(y + shift, height as i32);

        columns[offset as usize] = column as u32;
        row_starts[offset as usize] = row as u32 * width;
    }

    let mut window = Array::<f32>::new(comptime!((size * size) as usize));

    #[unroll]
    for window_row in 0..size {
        #[unroll]
        for window_column in 0..size {
            let index = row_starts[window_row as usize] + columns[window_column as usize];
            let slot = comptime!((window_row * size + window_column) as usize);

            window[slot] = input[index as usize];
        }
    }

    output[ABSOLUTE_POS] = select_median(&mut window, radius);
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
    assert!((1..=3).contains(&radius), "radius is 1, 2 or 3");

    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.output.samples();
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` f32 values, every read is reflected inside the plane, and the window
    // holds every slot the network for `radius` touches.
    unsafe {
        median_kernel::launch_unchecked::<R>(client, cubes, unit, input, output, width, height, radius);
    }
}
