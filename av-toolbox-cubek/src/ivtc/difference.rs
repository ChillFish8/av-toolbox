// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// How many columns one partial sum covers.
///
/// A sum of 1024 values, each up to 6 × 65535 = 393 210, still fits a `u32`.
pub const SEGMENT: u32 = 1024;

/// How many partial sums cover one row `width` samples across.
pub const fn segments(width: u32) -> u32 {
    width.div_ceil(SEGMENT)
}

#[cube(launch_unchecked)]
fn difference_sums_kernel(first: &[f32], second: &[f32], sums: &mut [u32], width: u32, segments: u32) {
    if ABSOLUTE_POS >= sums.len() {
        terminate!();
    }

    let unit = ABSOLUTE_POS as u32;
    let row = unit / segments;
    let segment = unit % segments;
    let start = segment * SEGMENT;
    let end = u32::min(start + SEGMENT, width);
    let row_start = row * width;

    let mut total = 0u32;
    for column in start..end {
        let index = (row_start + column) as usize;
        let first_value = i32::cast_from(first[index]);
        let second_value = i32::cast_from(second[index]);
        let difference = first_value - second_value;
        total += u32::cast_from(i32::abs(difference));
    }

    sums[ABSOLUTE_POS] = total;
}

/// The planes one difference sum reads and the partial sums it writes.
///
/// `sums` holds `height * segments(width)` `u32` values, row by row.
pub struct DifferencePlanes<'a> {
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    pub sums: DevicePlane<'a>,
}

/// Sums `|first - second|` over every `SEGMENT` columns of every row.
pub fn difference_sums(client: &Client, dispatch: &impl Dispatch, planes: DifferencePlanes<'_>) {
    let width = planes.first.width();
    let height = planes.first.height();
    assert_plane(&planes.second, width, height, "second");

    let segment_count = segments(width);
    let units = height as usize * segment_count as usize;
    assert!(
        planes.sums.samples() == units,
        "the sums hold {} values, expected {units}",
        planes.sums.samples(),
    );

    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();
    let first = planes.first.array();
    let second = planes.second.array();
    let sums = planes.sums.array();

    // Safety: both planes hold `width * height` f32 values, and each unit reads only its own row's segment
    // and writes only its own sum.
    unsafe {
        difference_sums_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            first,
            second,
            sums,
            width,
            segment_count,
        );
    }
}
