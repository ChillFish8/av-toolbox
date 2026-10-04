// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::subpel::half_pel_sample;
use super::{BlockGrid, assert_blocks};
use crate::{DevicePlane, Dispatch};

/// The planes `compensate` reads and writes.
///
/// `neighbour` and `output` hold the same plane size as f32. `vectors` holds the luma field's
/// `(x, y)` pairs of `i32`, `2 * across` wide and `down` high.
pub struct CompensatePlanes<'a> {
    pub neighbour: DevicePlane<'a>,
    pub vectors: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// How much smaller a plane is than luma, as a right shift along each axis.
///
/// It is 0 for luma and the chroma subsampling for chroma.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneScale {
    pub shift_x: u32,
    pub shift_y: u32,
}

impl PlaneScale {
    /// A plane the size of luma.
    pub const LUMA: Self = Self {
        shift_x: 0,
        shift_y: 0,
    };

    /// The scale of a plane that is `factor_x` by `factor_y` times smaller than luma.
    ///
    /// Each factor is 1 or 2, as a chroma subsampling gives them.
    pub const fn from_factors((factor_x, factor_y): (u32, u32)) -> Self {
        assert!(
            matches!(factor_x, 1 | 2) && matches!(factor_y, 1 | 2),
            "a subsampling factor is 1 or 2"
        );

        Self {
            shift_x: factor_x.trailing_zeros(),
            shift_y: factor_y.trailing_zeros(),
        }
    }
}

/// The raised-cosine weight of sample `position` along one axis of a block `length` long.
///
/// It ramps up across the first `overlap` samples, unless the block is the first along the axis, and down
/// across the last `overlap`, unless it is the last. It is 1 everywhere else.
pub fn window_weight(position: u32, length: u32, overlap: u32, first: bool, last: bool) -> f32 {
    if overlap == 0 {
        return 1.0;
    }

    let rising = !first && position < overlap;
    let falling = !last && position >= length - overlap;
    let mut weight = 1.0f32;
    if rising {
        let phase = (position as f32 + 0.5) / overlap as f32;
        weight *= 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
    }

    if falling {
        let from_end = length - 1 - position;
        let phase = (from_end as f32 + 0.5) / overlap as f32;
        weight *= 0.5 - 0.5 * (std::f32::consts::PI * phase).cos();
    }

    weight
}

/// The same weight as [window_weight], computed with the same operations on the device.
#[cube]
pub(super) fn device_window_weight(position: u32, length: u32, overlap: u32, first: bool, last: bool) -> f32 {
    let pi = comptime!(std::f32::consts::PI);
    let mut weight = 1.0f32;

    if overlap > 0 {
        if !first && position < overlap {
            let phase = (position as f32 + 0.5f32) / overlap as f32;
            weight *= 0.5f32 - 0.5f32 * f32::cos(pi * phase);
        }

        if !last && position >= length - overlap {
            let from_end = length - 1 - position;
            let phase = (from_end as f32 + 0.5f32) / overlap as f32;
            weight *= 0.5f32 - 0.5f32 * f32::cos(pi * phase);
        }
    }

    weight
}

/// The last block along one axis that can cover the sample at `position`, on a plane scaled by `shift`.
///
/// Only that block and the one before it can cover the sample, as the blocks are at least half a block apart.
#[cube]
pub(crate) fn upper_block(position: u32, step: u32, count: u32, #[comptime] shift: u32) -> u32 {
    ((((position + 1) << shift) - 1) / step).min(count - 1)
}

/// One component of a block's vector, shifted onto a plane scaled by `shift` and put in half pixels.
///
/// The vector is in units of `1 / pel` pixel, and the shift rounds towards negative infinity.
#[cube]
pub(crate) fn half_pel_component(vector: i32, #[comptime] shift: u32, #[comptime] pel: u32) -> i32 {
    let half_units = comptime!((2 / pel) as i32);

    (vector >> shift as i32) * half_units
}

/// The window weight of block `block_x`, `block_y` at the sample `x`, `y`, or 0 where the block does not
/// cover it.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel helper takes each position and setting as its own argument"
)]
#[cube]
pub(crate) fn block_window(
    x: u32,
    y: u32,
    block_x: u32,
    block_y: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] across: u32,
    #[comptime] down: u32,
    #[comptime] shift_x: u32,
    #[comptime] shift_y: u32,
) -> f32 {
    let overlap = comptime!(blksize - step);
    let length_x = comptime!(blksize >> shift_x);
    let length_y = comptime!(blksize >> shift_y);
    let overlap_x = comptime!(overlap >> shift_x);
    let overlap_y = comptime!(overlap >> shift_y);

    let start_x = (block_x * step) >> shift_x;
    let start_y = (block_y * step) >> shift_y;

    let mut window = 0.0f32;
    let covered_x = x >= start_x && x < start_x + length_x;
    let covered_y = y >= start_y && y < start_y + length_y;
    if covered_x && covered_y {
        let weight_x = device_window_weight(
            x - start_x,
            length_x,
            overlap_x,
            block_x == 0,
            block_x == across - 1,
        );
        let weight_y = device_window_weight(
            y - start_y,
            length_y,
            overlap_y,
            block_y == 0,
            block_y == down - 1,
        );
        window = weight_x * weight_y;
    }

    window
}

/// Builds each output sample from the blocks covering it, weighting their reads by their windows.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn compensate_kernel(
    neighbour: &Array<f32>,
    vectors: &Array<i32>,
    output: &mut Array<f32>,
    #[comptime] width: u32,
    #[comptime] height: u32,
    #[comptime] blksize: u32,
    #[comptime] step: u32,
    #[comptime] across: u32,
    #[comptime] down: u32,
    #[comptime] pel: u32,
    #[comptime] shift_x: u32,
    #[comptime] shift_y: u32,
) {
    let samples = comptime!(width * height);
    if ABSOLUTE_POS >= samples as usize {
        terminate!();
    }

    let x = ABSOLUTE_POS as u32 % width;
    let y = ABSOLUTE_POS as u32 / width;

    let upper_x = upper_block(x, step, across, shift_x);
    let upper_y = upper_block(y, step, down, shift_y);

    let mut weight_sum = 0.0f32;
    let mut value_sum = 0.0f32;
    for back_y in 0..2u32 {
        for back_x in 0..2u32 {
            if upper_x >= back_x && upper_y >= back_y {
                let block_x = upper_x - back_x;
                let block_y = upper_y - back_y;
                let weight = block_window(
                    x, y, block_x, block_y, blksize, step, across, down, shift_x, shift_y,
                );

                if weight > 0.0f32 {
                    let slot = ((block_y * across + block_x) * 2) as usize;
                    let vector_x = half_pel_component(vectors[slot], shift_x, pel);
                    let vector_y = half_pel_component(vectors[slot + 1], shift_y, pel);
                    let value = half_pel_sample(
                        neighbour,
                        2 * x as i32 + vector_x,
                        2 * y as i32 + vector_y,
                        width,
                        height,
                    );

                    weight_sum += weight;
                    value_sum += weight * value;
                }
            }
        }
    }

    output[ABSOLUTE_POS] = value_sum / weight_sum;
}

/// Lines `neighbour` up with the centre frame by the luma motion field in `vectors`.
///
/// Each output sample is the sum, over every block of `grid` covering it, of the block's window weight
/// times the neighbour read at the sample plus the block's vector, divided by the sum of the weights.
/// Reads between samples are bilinear, and reads past the frame take its nearest edge sample.
///
/// The vectors are in units of `1 / pel` pixel. On a plane smaller than luma, `scale` shrinks the blocks
/// and overlap by the same shifts, and each vector is shifted right by them, rounding towards negative
/// infinity.
pub fn compensate<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: CompensatePlanes<'_>,
    grid: &BlockGrid,
    pel: u32,
    scale: PlaneScale,
) {
    assert!((1..=2).contains(&pel), "pel is {pel}, expected 1 or 2");

    let width = planes.neighbour.width();
    let height = planes.neighbour.height();
    crate::assert_plane(&planes.neighbour, width, height, "neighbour");
    crate::assert_plane(&planes.output, width, height, "output");
    crate::assert_plane(&planes.vectors, 2 * grid.across, grid.down, "vectors");
    assert_blocks(grid, (width, height), scale);

    let cubes = dispatch.cube_count(planes.output.samples());
    let unit = dispatch.cube_dim();
    let neighbour = planes.neighbour.array::<R>();
    let vectors = planes.vectors.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: the planes hold `width` by `height` samples that every read clamps into, the vectors hold a
    // pair for each block of `grid`, and each unit writes only its own sample of `output`.
    unsafe {
        compensate_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            neighbour,
            vectors,
            output,
            width,
            height,
            grid.blksize,
            grid.step,
            grid.across,
            grid.down,
            pel,
            scale.shift_x,
            scale.shift_y,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_ramps_over_an_overlap_sum_to_one() {
        for position in 0..8 {
            let next_block = window_weight(position, 16, 8, false, false);
            let previous_block = window_weight(position + 8, 16, 8, false, false);

            assert!(
                (next_block + previous_block - 1.0).abs() < 1e-6,
                "ramps at {position} do not sum to 1"
            );
        }
    }

    #[test]
    fn the_outer_sides_of_the_edge_blocks_have_no_ramp() {
        assert_eq!(window_weight(0, 16, 8, true, false), 1.0);
        assert_eq!(window_weight(15, 16, 8, false, true), 1.0);
        assert_eq!(window_weight(3, 16, 0, false, false), 1.0);
    }
}
