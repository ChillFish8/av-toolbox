// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// The fewest rows a plane needs for a bwdif field pass.
pub const BWDIF_MIN_ROWS: u32 = 4;

pub(crate) const LOW_CENTRE: f32 = 4309.0 / 8192.0;
pub(crate) const LOW_OUTER: f32 = 213.0 / 8192.0;
pub(crate) const HIGH_CENTRE: f32 = 5570.0 / 8192.0;
pub(crate) const HIGH_NEAR: f32 = 3801.0 / 8192.0;
pub(crate) const HIGH_FAR: f32 = 1016.0 / 8192.0;
pub(crate) const CUBIC_CENTRE: f32 = 5077.0 / 8192.0;
pub(crate) const CUBIC_OUTER: f32 = 981.0 / 8192.0;

/// The planes one bwdif field pass reads and writes, all f32 code values of one size.
pub struct BwdifPlanes<'a> {
    pub previous: DevicePlane<'a>,
    pub current: DevicePlane<'a>,
    pub next: DevicePlane<'a>,
    /// A spatial prediction of the rebuilt rows, read in place of bwdif's own cubic.
    pub spatial: Option<DevicePlane<'a>>,
    pub output: DevicePlane<'a>,
}

/// Which rows a bwdif field pass rebuilds, and when the field it keeps was captured.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BwdifField {
    /// The parity of the rows rebuilt, 0 for the even rows and 1 for the odd rows.
    pub rebuilt: u32,
    /// Whether the kept field is the first of its frame.
    ///
    /// The rebuilt rows of a first field are judged against the previous frame and this one, and those of a
    /// second field against this frame and the next.
    pub first: bool,
}

/// The index of the sample `offset` rows from `y` in column `x`.
#[cube]
fn index_at(width: u32, x: u32, y: u32, offset: i32) -> usize {
    let row = y as i32 + offset;
    let index = row as u32 * width + x;

    index as usize
}

#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn bwdif_kernel(
    previous: &[f32],
    current: &[f32],
    next: &[f32],
    earlier: &[f32],
    later: &[f32],
    spatial: &[f32],
    output: &mut [f32],
    width: u32,
    height: u32,
    rebuilt: u32,
    #[comptime] has_spatial: bool,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let y = position / width;

    if y % 2 != rebuilt {
        output[ABSOLUTE_POS] = current[ABSOLUTE_POS];
        terminate!();
    }

    let mut above = 1i32;
    if y > 0 {
        above = -1;
    }

    let mut below = 1i32;
    if y + 1 >= height {
        below = -1;
    }

    let above_index = index_at(width, x, y, above);
    let below_index = index_at(width, x, y, below);

    let current_above = current[above_index];
    let current_below = current[below_index];
    let earlier_here = earlier[ABSOLUTE_POS];
    let later_here = later[ABSOLUTE_POS];
    let average = (earlier_here + later_here) * 0.5;

    let temporal_diff0 = f32::abs(earlier_here - later_here);
    let previous_above = f32::abs(previous[above_index] - current_above);
    let previous_below = f32::abs(previous[below_index] - current_below);
    let temporal_diff1 = (previous_above + previous_below) * 0.5;
    let next_above = f32::abs(next[above_index] - current_above);
    let next_below = f32::abs(next[below_index] - current_below);
    let temporal_diff2 = (next_above + next_below) * 0.5;

    let half_diff0 = temporal_diff0 * 0.5;
    let spread = f32::max(half_diff0, temporal_diff1);
    let mut diff = f32::max(spread, temporal_diff2);

    if diff == 0.0 {
        output[ABSOLUTE_POS] = average;
        terminate!();
    }

    let inner = y >= 4 && y + 5 <= height;
    let reaches_two = y >= 2 && y + 3 <= height;

    if reaches_two {
        let above_two = index_at(width, x, y, -2);
        let below_two = index_at(width, x, y, 2);
        let outer_above = (earlier[above_two] + later[above_two]) * 0.5 - current_above;
        let outer_below = (earlier[below_two] + later[below_two]) * 0.5 - current_below;
        let towards_above = average - current_above;
        let towards_below = average - current_below;

        let near_high = f32::max(towards_below, towards_above);
        let outer_low = f32::min(outer_above, outer_below);
        let maximum = f32::max(near_high, outer_low);
        let near_low = f32::min(towards_below, towards_above);
        let outer_high = f32::max(outer_above, outer_below);
        let minimum = f32::min(near_low, outer_high);

        let widened = f32::max(diff, minimum);
        diff = f32::max(widened, -maximum);
    }

    let pair = current_above + current_below;
    let mut interpolated = pair * 0.5;
    if has_spatial {
        interpolated = spatial[ABSOLUTE_POS];
    }

    if inner {
        let above_three = index_at(width, x, y, -3);
        let below_three = index_at(width, x, y, 3);
        let current_outer = current[above_three] + current[below_three];

        if f32::abs(current_above - current_below) > temporal_diff0 {
            let above_two = index_at(width, x, y, -2);
            let below_two = index_at(width, x, y, 2);
            let above_four = index_at(width, x, y, -4);
            let below_four = index_at(width, x, y, 4);
            let centre = earlier_here + later_here;
            let near = earlier[above_two] + later[above_two] + earlier[below_two] + later[below_two];
            let far = earlier[above_four] + later[above_four] + earlier[below_four] + later[below_four];
            let temporal = (HIGH_CENTRE * centre - HIGH_NEAR * near + HIGH_FAR * far) * 0.25;
            interpolated = temporal + LOW_CENTRE * pair - LOW_OUTER * current_outer;
        } else if !has_spatial {
            interpolated = CUBIC_CENTRE * pair - CUBIC_OUTER * current_outer;
        }
    }

    let low = average - diff;
    let high = average + diff;
    output[ABSOLUTE_POS] = f32::clamp(interpolated, low, high);
}

/// Rebuilds the rows of parity `field.rebuilt` of `planes.current` into `planes.output`, copying the others.
///
/// A rebuilt sample blends a spatial prediction with the same sample of the neighbouring fields, clamped by
/// how much the area moves. The prediction is bwdif's cubic down the column, or `planes.spatial` when given.
pub fn bwdif_field(client: &Client, dispatch: &impl Dispatch, planes: BwdifPlanes<'_>, field: BwdifField) {
    let width = planes.output.width();
    let height = planes.output.height();

    assert!(
        height >= BWDIF_MIN_ROWS,
        "bwdif needs at least {BWDIF_MIN_ROWS} rows, got {height}"
    );
    assert!(
        field.rebuilt < 2,
        "bwdif rebuilds rows of parity 0 or 1, not {}",
        field.rebuilt
    );
    assert_plane(&planes.previous, width, height, "previous");
    assert_plane(&planes.current, width, height, "current");
    assert_plane(&planes.next, width, height, "next");
    assert_plane(&planes.output, width, height, "output");
    if let Some(spatial) = &planes.spatial {
        assert_plane(spatial, width, height, "spatial");
    }

    let (earlier, later) = if field.first {
        (planes.previous, planes.current)
    } else {
        (planes.current, planes.next)
    };

    // Without a prediction the kernel never reads `spatial`, so any plane of the right size fills the
    // binding.
    let has_spatial = planes.spatial.is_some();
    let spatial = planes.spatial.unwrap_or(planes.current);

    let samples = planes.output.samples();
    let previous_array = planes.previous.array();
    let current_array = planes.current.array();
    let next_array = planes.next.array();
    let earlier_array = earlier.array();
    let later_array = later.array();
    let spatial_array = spatial.array();
    let output_array = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values. A rebuilt row reads 2 or 4 rows away only when its
    // row class has those rows, and mirrors the rows next to it at the plane's edges, so every read stays
    // inside the plane.
    unsafe {
        bwdif_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            previous_array,
            current_array,
            next_array,
            earlier_array,
            later_array,
            spatial_array,
            output_array,
            width,
            height,
            field.rebuilt,
            has_spatial,
        );
    }
}
