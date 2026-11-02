// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{MotionField, assert_field, assert_whole_pel_field, clamp_i32};
use crate::Dispatch;

/// The block of a grid, `count` blocks long and `step` apart, that contains `position` clamped into the grid.
///
/// `span` is `count * step`. Clamping `position` into it finds the same block as clamping it into the
/// frame, since only the last block reaches past the frame's edge.
#[cube]
fn block_at(position: i32, #[comptime] span: u32, #[comptime] step: u32, #[comptime] count: u32) -> u32 {
    let clamped = clamp_i32(position, span as i32) as u32;

    u32::min(clamped / step, count - 1)
}

/// Joins `first` and `second` one block per unit, from the centre of each block of `seed`.
///
/// The walk reads `first` at the block, moves by that vector and reads `second` where it lands. The
/// seed is the sum of both. Only the block lookups clamp, the position keeps its true value.
#[cube(launch_unchecked)]
fn chain_kernel(
    first: &[i32],
    second: &[i32],
    seed: &mut [i32],
    #[comptime] step: u32,
    #[comptime] across: u32,
    #[comptime] down: u32,
) {
    let blocks = comptime!(across * down);
    if ABSOLUTE_POS >= blocks as usize {
        terminate!();
    }

    let block = ABSOLUTE_POS as u32;
    let bx = block % across;
    let by = block / across;
    let span_x = comptime!(across * step);
    let span_y = comptime!(down * step);

    let mut pos_x = (bx * step + step / 2) as i32;
    let mut pos_y = (by * step + step / 2) as i32;
    let mut acc_x = 0i32;
    let mut acc_y = 0i32;

    let first_x = block_at(pos_x, span_x, step, across);
    let first_y = block_at(pos_y, span_y, step, down);
    let first_base = (first_y * across + first_x) * 2;
    let first_dx = first[first_base as usize];
    let first_dy = first[(first_base + 1) as usize];

    acc_x += first_dx;
    acc_y += first_dy;
    pos_x += first_dx;
    pos_y += first_dy;

    let second_x = block_at(pos_x, span_x, step, across);
    let second_y = block_at(pos_y, span_y, step, down);
    let second_base = (second_y * across + second_x) * 2;
    let second_dx = second[second_base as usize];
    let second_dy = second[(second_base + 1) as usize];

    acc_x += second_dx;
    acc_y += second_dy;

    let slot = (block * 2) as usize;
    seed[slot] = acc_x;
    seed[slot + 1] = acc_y;
}

/// Joins `first`, the field from frame `a` to `b`, with `second`, from `b` to `c`, into a seed from `a`
/// to `c` in `seed`'s vectors.
///
/// Each block follows `first` from its centre, then adds the vector of the block of `second` it lands
/// in. All three fields share one grid and hold whole pixel vectors, so each has `pel` 1.
pub fn chain(
    client: &Client,
    dispatch: &impl Dispatch,
    first: &MotionField<'_>,
    second: &MotionField<'_>,
    seed: &MotionField<'_>,
) {
    assert_field(first);
    assert_field(second);
    assert_field(seed);
    assert_whole_pel_field(first, "chain");
    assert_whole_pel_field(second, "chain");
    assert_whole_pel_field(seed, "chain");
    assert_eq!(
        first.grid, seed.grid,
        "the first field's grid differs from the seed's"
    );
    assert_eq!(
        second.grid, seed.grid,
        "the second field's grid differs from the seed's"
    );

    let grid = &seed.grid;
    let cubes = dispatch.cube_count(grid.blocks());
    let unit = dispatch.cube_dim();
    let first_array = first.planes.vectors.array();
    let second_array = second.planes.vectors.array();
    let seed_array = seed.planes.vectors.array();

    // Safety: every field holds two i32 per block of `grid`, each unit reads one block of each field,
    // clamped to the grid, and writes its own block of the seed.
    unsafe {
        chain_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            first_array,
            second_array,
            seed_array,
            grid.step,
            grid.across,
            grid.down,
        );
    }
}
