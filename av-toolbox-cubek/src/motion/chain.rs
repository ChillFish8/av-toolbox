use cubecl::prelude::*;

use super::{BlockGrid, FieldPlanes, assert_field, clamp_i32};
use crate::Dispatch;

/// The block of a grid `count` blocks long and `step` apart nearest `position`, clamped to the grid.
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
    first: &Array<i32>,
    second: &Array<i32>,
    seed: &mut Array<i32>,
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
/// to `c` in `seed.vectors`.
///
/// Each block follows `first` from its centre, then adds the vector of the block of `second` it lands
/// in.
pub fn chain<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    first: &FieldPlanes<'_>,
    second: &FieldPlanes<'_>,
    seed: &FieldPlanes<'_>,
    grid: &BlockGrid,
) {
    assert_field(first, grid);
    assert_field(second, grid);
    assert_field(seed, grid);

    let cubes = dispatch.cube_count(grid.blocks());
    let unit = dispatch.cube_dim();
    let first_array = first.vectors.array::<R>();
    let second_array = second.vectors.array::<R>();
    let seed_array = seed.vectors.array::<R>();

    // Safety: every field holds two i32 per block of `grid`, each unit reads one block of each field,
    // clamped to the grid, and writes its own block of the seed.
    unsafe {
        chain_kernel::launch_unchecked::<R>(
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
