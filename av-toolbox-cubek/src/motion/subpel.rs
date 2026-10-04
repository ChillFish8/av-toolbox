use cubecl::prelude::*;

use super::search::assert_pyramids;
use super::{BlockGrid, FieldPlanes, Pyramid, assert_field, clamp_i32};
use crate::Dispatch;

/// The luma of `plane` at the half pixel position `x2`, `y2`, clamped into the frame.
///
/// Between two samples it is their mean. Between four it is their mean, summed top left, top right,
/// bottom left, then bottom right.
#[cube]
pub(crate) fn half_pel_sample(
    plane: &Array<f32>,
    x2: i32,
    y2: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
) -> f32 {
    let odd_x = x2 & 1;
    let odd_y = y2 & 1;
    let whole_x = (x2 - odd_x) / 2;
    let whole_y = (y2 - odd_y) / 2;

    let left = clamp_i32(whole_x, width as i32);
    let top = clamp_i32(whole_y, height as i32);
    let right = clamp_i32(whole_x + odd_x, width as i32);
    let bottom = clamp_i32(whole_y + odd_y, height as i32);

    let top_left = plane[(top * width as i32 + left) as usize];
    let top_right = plane[(top * width as i32 + right) as usize];
    let bottom_left = plane[(bottom * width as i32 + left) as usize];
    let bottom_right = plane[(bottom * width as i32 + right) as usize];

    let mut value = top_left;
    if odd_x == 1 && odd_y == 0 {
        value = (top_left + top_right) * 0.5f32;
    } else if odd_x == 0 && odd_y == 1 {
        value = (top_left + bottom_left) * 0.5f32;
    } else if odd_x == 1 && odd_y == 1 {
        value = (top_left + top_right + bottom_left + bottom_right) * 0.25f32;
    }

    value
}

/// The SAD of the block at `origin_x`, `origin_y` against `neighbour` shifted by `vector_x`, `vector_y`
/// half pixels, summed row by row.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel helper takes each plane and setting as its own argument"
)]
#[cube]
fn half_pel_sad(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    origin_x: i32,
    origin_y: i32,
    vector_x: i32,
    vector_y: i32,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
) -> f32 {
    let mut sad = 0.0f32;
    for iy in 0..blksize {
        for ix in 0..blksize {
            let cx = origin_x + ix as i32;
            let cy = origin_y + iy as i32;
            let clamped_x = clamp_i32(cx, width as i32);
            let clamped_y = clamp_i32(cy, height as i32);
            let centre_value = centre[(clamped_y * width as i32 + clamped_x) as usize];

            let neighbour_value =
                half_pel_sample(neighbour, 2 * cx + vector_x, 2 * cy + vector_y, width, height);
            let diff = centre_value - neighbour_value;
            let abs_diff = if diff < 0.0f32 { -diff } else { diff };
            sad += abs_diff;
        }
    }

    sad
}

/// Refines one block per unit from its whole pixel vector, doubled into half pixels.
///
/// The eight neighbours at one half pixel are tried in turn, top row first. A candidate replaces the
/// best only when its SAD is lower, so the doubled vector wins ties, then the earliest candidate.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn half_pel_kernel(
    centre: &Array<f32>,
    neighbour: &Array<f32>,
    vectors: &mut Array<i32>,
    sads: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] across: u32,
    #[comptime] down: u32,
) {
    let blocks = comptime!(across * down);
    if ABSOLUTE_POS >= blocks as usize {
        terminate!();
    }

    let block = ABSOLUTE_POS as u32;
    let origin_x = (block % across * step) as i32;
    let origin_y = (block / across * step) as i32;
    let slot = (block * 2) as usize;

    let start_x = vectors[slot] * 2;
    let start_y = vectors[slot + 1] * 2;

    let mut best_x = start_x;
    let mut best_y = start_y;
    let mut best_sad = half_pel_sad(
        centre, neighbour, origin_x, origin_y, start_x, start_y, width, height, blksize,
    );

    for candidate in 0..8u32 {
        // The ninth cell of the 3x3 window is the start itself, so cells past the fourth shift up one.
        let mut cell = candidate;
        if candidate >= 4 {
            cell = candidate + 1;
        }

        let vector_x = start_x + (cell % 3) as i32 - 1;
        let vector_y = start_y + (cell / 3) as i32 - 1;
        let sad = half_pel_sad(
            centre, neighbour, origin_x, origin_y, vector_x, vector_y, width, height, blksize,
        );
        if sad < best_sad {
            best_sad = sad;
            best_x = vector_x;
            best_y = vector_y;
        }
    }

    vectors[slot] = best_x;
    vectors[slot + 1] = best_y;
    sads[block as usize] = best_sad;
}

/// Refines the whole pixel vectors in `field` to half pixels at full resolution.
///
/// Each vector is doubled, then moved by up to one half pixel in each direction to the position that
/// matches best. `vectors` is replaced with the result in half pixels and `sads` with its SAD. The
/// confidence is left as the whole pixel search wrote it.
///
/// On a chained field, run it after [refine_seed](crate::motion::refine_seed).
pub fn refine_half_pel<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: &Pyramid<'_>,
    neighbour: &Pyramid<'_>,
    field: &FieldPlanes<'_>,
    grid: &BlockGrid,
) {
    assert_pyramids(centre, neighbour, 1);
    assert_field(field, grid);

    let cubes = dispatch.cube_count(grid.blocks());
    let unit = dispatch.cube_dim();
    let centre_plane = centre.levels[0];
    let centre_array = centre_plane.array::<R>();
    let neighbour_array = neighbour.levels[0].array::<R>();
    let vectors = field.vectors.array::<R>();
    let sads = field.sads.array::<R>();

    // Safety: both planes hold `width` by `height` samples that every read clamps into, and each unit
    // reads and writes only its own block's entries of `vectors` and `sads`.
    unsafe {
        half_pel_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            centre_array,
            neighbour_array,
            vectors,
            sads,
            centre_plane.width(),
            centre_plane.height(),
            grid.blksize,
            grid.step,
            grid.across,
            grid.down,
        );
    }
}
