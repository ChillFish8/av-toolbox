// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch, assert_plane};

/// The size of one half block of a plane, half of VDecimate's `blockx` and `blocky` scaled down by the
/// plane's subsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HalfBlocks {
    pub x: u32,
    pub y: u32,
}

/// How many half blocks cover a plane `width` by `height`, across and then down.
///
/// A half block cut short by the right or bottom edge still counts.
pub fn half_block_grid(width: u32, height: u32, half: HalfBlocks) -> (u32, u32) {
    let across = width.div_ceil(half.x);
    let down = height.div_ceil(half.y);

    (across, down)
}

#[cube(launch_unchecked)]
fn half_block_differences_kernel(
    previous: &[f32],
    current: &[f32],
    cells: &mut [u32],
    width: u32,
    height: u32,
    half_x: u32,
    half_y: u32,
    across: u32,
) {
    if ABSOLUTE_POS >= cells.len() {
        terminate!();
    }

    let unit = ABSOLUTE_POS as u32;
    let row = unit / across;
    let column = unit % across;
    let start_x = u32::min(column * half_x, width);
    let end_x = u32::min(start_x + half_x, width);
    let start_y = u32::min(row * half_y, height);
    let end_y = u32::min(start_y + half_y, height);

    let mut total = 0u32;
    for y in start_y..end_y {
        let row_start = y * width;
        for x in start_x..end_x {
            let index = (row_start + x) as usize;
            let previous_value = i32::cast_from(previous[index]);
            let current_value = i32::cast_from(current[index]);
            let difference = previous_value - current_value;
            total += u32::cast_from(i32::abs(difference));
        }
    }

    cells[ABSOLUTE_POS] = total;
}

/// The two planes one half block difference compares and the cells it writes.
///
/// `cells` holds one `u32` per cell of the luma grid, row by row.
pub struct BlockPlanes<'a> {
    pub previous: DevicePlane<'a>,
    pub current: DevicePlane<'a>,
    pub cells: DevicePlane<'a>,
}

/// Sums `|previous - current|` over every half block of one plane, the way VDecimate's `calcMetric` does.
///
/// `across` is how many cells the luma grid has in each row. A cell past this plane's own grid is set to 0.
pub fn half_block_differences(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: BlockPlanes<'_>,
    half: HalfBlocks,
    across: u32,
) {
    let width = planes.previous.width();
    let height = planes.previous.height();
    assert_plane(&planes.current, width, height, "current");
    assert!(
        half.x > 0 && half.y > 0,
        "a half block is {}x{}, expected at least one sample each way",
        half.x,
        half.y,
    );

    let units = planes.cells.samples();
    assert!(
        across > 0 && units.is_multiple_of(across as usize),
        "the cells hold {units} values, which is not whole rows of {across}",
    );

    let down = (units / across as usize) as u32;
    let (plane_across, plane_down) = half_block_grid(width, height, half);
    assert!(
        plane_across <= across && plane_down <= down,
        "the plane has {plane_across}x{plane_down} half blocks, more than the {across}x{down} cells",
    );

    let cubes = dispatch.cube_count(units);
    let unit = dispatch.cube_dim();
    let previous = planes.previous.array();
    let current = planes.current.array();
    let cells = planes.cells.array();

    // Safety: both planes hold `width * height` f32 values, each unit reads only the samples of its own
    // half block clamped to the plane, and writes only its own cell.
    unsafe {
        half_block_differences_kernel::launch_unchecked(
            client, cubes, unit, previous, current, cells, width, height, half.x, half.y, across,
        );
    }
}

/// VDecimate's two measures of how much a frame differs from the one before.
///
/// `max_block` is the highest sum of a 2x2 window of half blocks, which is a whole block, and `total` the
/// sum of every half block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecimateMetric {
    pub max_block: i64,
    pub total: i64,
}

impl DecimateMetric {
    /// Folds a grid of half block sums, `across` cells by `down`, into the two measures.
    ///
    /// Windows overlap, one cell apart. `max_block` is -1 when the grid is one cell across or down, as there
    /// is no window.
    pub fn from_cells(cells: &[u64], across: usize, down: usize) -> Self {
        assert_eq!(cells.len(), across * down, "the cells should be {across}x{down}");

        let mut max_block = -1i64;
        for row in 0..down.saturating_sub(1) {
            for column in 0..across.saturating_sub(1) {
                let top = row * across + column;
                let bottom = top + across;
                let window = cells[top] + cells[top + 1] + cells[bottom] + cells[bottom + 1];
                max_block = max_block.max(window as i64);
            }
        }

        let total: u64 = cells.iter().sum();

        Self {
            max_block,
            total: total as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_one_cell_across_has_no_window() {
        let metric = DecimateMetric::from_cells(&[5, 7], 1, 2);

        assert_eq!(
            metric,
            DecimateMetric {
                max_block: -1,
                total: 12
            }
        );
    }

    #[test]
    fn the_highest_window_is_the_max_block() {
        let cells = [1, 2, 3, 4, 5, 6, 7, 8, 100];

        let metric = DecimateMetric::from_cells(&cells, 3, 3);

        assert_eq!(metric.max_block, 5 + 6 + 8 + 100);
        assert_eq!(metric.total, 136);
    }

    #[test]
    fn a_short_half_block_still_counts() {
        let half = HalfBlocks { x: 16, y: 16 };

        assert_eq!(half_block_grid(33, 16, half), (3, 1));
    }
}
