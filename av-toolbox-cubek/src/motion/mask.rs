// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use cubecl::server::Handle;

use super::{BlockGrid, MotionField, SceneFlag, assert_field, assert_flag};
use crate::{DevicePlane, Dispatch, assert_plane};

/// QTGMC's `ProgSADMask` at "Slower".
pub const PROGRESSIVE_SAD_MASK: f32 = 10.0;

/// How many bits of fraction `SimpleResize`'s weights hold.
pub(crate) const WEIGHT_SHIFT: u32 = 14;

/// The weight of a whole sample, `simple_resize_weight_max`.
pub(crate) const WEIGHT_MAX: u32 = 1 << WEIGHT_SHIFT;

/// The rounding added before the shift, `simple_resize_weight_half`.
pub(crate) const WEIGHT_HALF: u32 = WEIGHT_MAX / 2;

/// How many mask bytes lie above 0.
const LEVELS: usize = 255;

/// MVTools' `ByteNorm` at `gamma = 1` for a block of normalised SAD `sad`.
///
/// The SAD is put on the 8-bit scale, times 255, then `255 × sad × 4 / (ml × blksize²)` is worked out in f64
/// and truncated to a byte, saturating at 255.
pub(crate) fn sad_byte(sad: f32, ml: f32, blksize: u32) -> u32 {
    let mask_norm_factor = 1.0f32 / ml;
    let samples = f64::from(blksize * blksize);
    let sad_norm_factor = 4.0 * f64::from(mask_norm_factor) / samples;
    let sad_8_bit = f64::from(sad) * 255.0;

    let level = 255.0 * (sad_8_bit * sad_norm_factor);
    let saturated = level.min(255.0);

    saturated as u32
}

/// The smallest SAD giving each mask byte from 1 to 255.
///
/// [sad_byte] never falls as the SAD grows, so a block's byte is how many of these it reaches. Each is found
/// by bisecting the bit patterns of positive f32s, which sort as the values do.
fn byte_thresholds(ml: f32, blksize: u32) -> Vec<f32> {
    let highest = f32::INFINITY.to_bits();

    let mut thresholds = Vec::with_capacity(LEVELS);
    for level in 1..=LEVELS as u32 {
        let mut low = 0u32;
        let mut high = highest;
        while low < high {
            let middle = low + (high - low) / 2;
            let sad = f32::from_bits(middle);

            if sad_byte(sad, ml, blksize) >= level {
                high = middle;
            } else {
                low = middle + 1;
            }
        }

        thresholds.push(f32::from_bits(low));
    }

    thresholds
}

/// MVTools' `InitTables` from `SimpleResize.cpp`, resizing `input` samples to `output`.
///
/// Each output reads the pair of inputs from its offset, the second weighed by its weight out of
/// [WEIGHT_MAX]. Positions are worked out in f32 as MVTools does. With one input, the last outputs have an
/// offset of -1 and a weight of [WEIGHT_MAX], so they read only the input.
pub(crate) fn simple_resize_tables(output: u32, input: u32) -> (Vec<i32>, Vec<u32>) {
    let leftmost = 0.5f32;
    let rightmost = input as f32 - 0.5;
    let leftmost_index = (leftmost as i32).max(0);
    let rightmost_index = (rightmost as i32).min(input as i32 - 1);

    let mut offsets = Vec::with_capacity(output as usize);
    let mut weights = Vec::with_capacity(output as usize);
    for index in 0..output {
        let position = (index as f32 + 0.5) * input as f32 / output as f32;

        let (offset, weight) = if position <= leftmost {
            (leftmost_index, 0.0f32)
        } else if position >= rightmost {
            (rightmost_index - 1, 1.0f32)
        } else {
            let offset = (position - leftmost) as i32;
            (offset, position - leftmost - offset as f32)
        };

        offsets.push(offset);
        weights.push((weight * WEIGHT_MAX as f32) as u32);
    }

    (offsets, weights)
}

/// The resize of `blocks` blocks to `covered` samples as `(first block, weight)` pairs for a plane `length`
/// long.
///
/// Samples past the covered area take the last covered sample's pair. The first block is clamped into the
/// grid, which only moves reads that weigh 0.
fn axis_table(covered: u32, blocks: u32, length: u32) -> Vec<u32> {
    let (offsets, weights) = simple_resize_tables(covered, blocks);
    let last = covered as usize - 1;

    let mut table = Vec::with_capacity(2 * length as usize);
    for index in 0..length as usize {
        let source = index.min(last);
        let first = offsets[source].clamp(0, blocks as i32 - 1);

        table.push(first as u32);
        table.push(weights[source]);
    }

    table
}

/// The width and height the blocks of `grid` cover, MVTools' `nWidthB` and `nHeightB`.
pub(crate) fn covered_area(grid: &BlockGrid) -> (u32, u32) {
    let width = (grid.across - 1) * grid.step + grid.blksize;
    let height = (grid.down - 1) * grid.step + grid.blksize;

    (width, height)
}

/// `(first × (max − weight) + second × weight + half) >> shift`, one of `SimpleResize`'s blends.
#[cube]
fn blend(first: u32, second: u32, weight: u32) -> u32 {
    let first_weight = WEIGHT_MAX - weight;

    (first * first_weight + second * weight + WEIGHT_HALF) >> WEIGHT_SHIFT
}

/// The mask byte of block `block`, how many of the thresholds its SAD reaches.
#[cube]
fn block_byte(sads: &Array<f32>, thresholds: &Array<f32>, block: u32) -> u32 {
    let sad = sads[block as usize];

    let mut count = 0u32;
    #[unroll]
    for halving in 0..8u32 {
        let probe = count + (128u32 >> halving);
        if probe <= LEVELS as u32 && thresholds[(probe - 1) as usize] <= sad {
            count = probe;
        }
    }

    count
}

/// One sample of the mask, resized from the bytes of the four blocks around it.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn sad_mask_kernel(
    sads: &Array<f32>,
    thresholds: &Array<f32>,
    across_table: &Array<u32>,
    down_table: &Array<u32>,
    flags: &Array<u32>,
    output: &mut Array<f32>,
    flag_index: u32,
    width: u32,
    across: u32,
    down: u32,
    inverse_peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    if flags[flag_index as usize] != 0u32 {
        output[ABSOLUTE_POS] = 0.0f32;
        terminate!();
    }

    let x = ABSOLUTE_POS as u32 % width;
    let y = ABSOLUTE_POS as u32 / width;

    let left = across_table[(2 * x) as usize];
    let right_weight = across_table[(2 * x + 1) as usize];
    let right = u32::min(left + 1, across - 1);
    let top = down_table[(2 * y) as usize];
    let bottom_weight = down_table[(2 * y + 1) as usize];
    let bottom = u32::min(top + 1, down - 1);

    let top_left = block_byte(sads, thresholds, top * across + left);
    let bottom_left = block_byte(sads, thresholds, bottom * across + left);
    let top_right = block_byte(sads, thresholds, top * across + right);
    let bottom_right = block_byte(sads, thresholds, bottom * across + right);

    let left_column = blend(top_left, bottom_left, bottom_weight);
    let right_column = blend(top_right, bottom_right, bottom_weight);
    let byte = blend(left_column, right_column, right_weight);

    output[ABSOLUTE_POS] = byte as f32 * inverse_peak;
}

/// The tables [sad_mask] reads for one grid, threshold and output size, uploaded once.
///
/// Each SAD byte's threshold comes from `ml` and the grid's block size. The resize tables map each column
/// and row of the output to the blocks around it, as `SimpleResize` does.
pub struct SadMaskTables {
    thresholds: Handle,
    across: Handle,
    down: Handle,
    grid: BlockGrid,
    width: u32,
    height: u32,
}

impl SadMaskTables {
    /// The tables for masks of `width` by `height` from fields on `grid`, at `ml`.
    pub fn new<R: Runtime>(
        client: &ComputeClient<R>,
        grid: &BlockGrid,
        ml: f32,
        (width, height): (u32, u32),
    ) -> Self {
        assert!(
            ml > 0.0 && ml.is_finite(),
            "ml is {ml}, expected a positive value"
        );
        assert!(grid.blocks() > 0, "the grid holds no blocks");

        let (covered_width, covered_height) = covered_area(grid);
        let thresholds = byte_thresholds(ml, grid.blksize);
        let across_table = axis_table(covered_width, grid.across, width);
        let down_table = axis_table(covered_height, grid.down, height);

        Self {
            thresholds: client.create_from_slice(f32::as_bytes(&thresholds)),
            across: client.create_from_slice(u32::as_bytes(&across_table)),
            down: client.create_from_slice(u32::as_bytes(&down_table)),
            grid: *grid,
            width,
            height,
        }
    }

    /// The thresholds and both resize tables as planes.
    fn planes(&self) -> [DevicePlane<'_>; 3] {
        // Safety: `new` uploads 255 thresholds, and a pair of 4 byte values for each column and each row.
        unsafe {
            [
                DevicePlane::new(&self.thresholds, LEVELS as u32, 1),
                DevicePlane::new(&self.across, 2 * self.width, 1),
                DevicePlane::new(&self.down, 2 * self.height, 1),
            ]
        }
    }
}

/// Writes MVTools' `Mask(kind=1)` of `field` as a plane between 0 and 1, zero where `flag` is set.
///
/// This follows `Mask` at `time = 100`, `gamma = 1` and `ysc = 0`. Each block's SAD becomes a byte as
/// `ByteNorm` makes it with the `ml` of `tables`, and the bytes are resized to the area the blocks cover as
/// `SimpleResize` does, in fixed point. Columns and rows past the covered area repeat its last column and
/// row, and a byte is stored as its value times `1 / 255`.
pub fn sad_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    field: &MotionField<'_>,
    flag: SceneFlag<'_>,
    tables: &SadMaskTables,
    output: DevicePlane<'_>,
) {
    assert_field(field);
    assert_flag(&flag);
    assert_eq!(
        field.grid, tables.grid,
        "the field and the tables are on other grids"
    );

    let grid = &field.grid;
    let (width, height) = (tables.width, tables.height);
    assert_plane(&output, width, height, "output");

    let [threshold_plane, across_plane, down_plane] = tables.planes();
    assert_plane(&threshold_plane, LEVELS as u32, 1, "thresholds");
    assert_plane(&across_plane, 2 * width, 1, "across table");
    assert_plane(&down_plane, 2 * height, 1, "down table");

    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let sads = field.planes.sads.array::<R>();
    let threshold_array = threshold_plane.array::<R>();
    let across_array = across_plane.array::<R>();
    let down_array = down_plane.array::<R>();
    let flags = flag.flags.array::<R>();
    let output_array = output.array::<R>();

    // Safety: the SADs hold a SAD for each block of `grid`, the tables hold a pair for each column and row
    // of `output` whose blocks lie inside the grid, there are 255 thresholds, the flag index is inside its
    // plane, and each unit writes only its own sample.
    unsafe {
        sad_mask_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            sads,
            threshold_array,
            across_array,
            down_array,
            flags,
            output_array,
            flag.index,
            width,
            grid.across,
            grid.down,
            1.0f32 / 255.0,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_threshold_is_the_first_sad_reaching_its_byte() {
        for (ml, blksize) in [(10.0, 16), (10.0, 8), (20.0, 4), (3.0, 32)] {
            let thresholds = byte_thresholds(ml, blksize);

            for (index, threshold) in thresholds.iter().enumerate() {
                let level = index as u32 + 1;
                let below = f32::from_bits(threshold.to_bits() - 1);

                assert!(
                    sad_byte(*threshold, ml, blksize) >= level,
                    "ml {ml} level {level}"
                );
                assert!(sad_byte(below, ml, blksize) < level, "ml {ml} level {level}");
            }
        }
    }

    #[test]
    fn three_blocks_to_eight_samples_match_init_tables() {
        let (offsets, weights) = simple_resize_tables(8, 3);

        assert_eq!(offsets, [0, 0, 0, 0, 1, 1, 1, 1]);
        assert_eq!(weights, [0, 1024, 7168, 13312, 3072, 9216, 15360, 16384]);
    }

    #[test]
    fn one_block_reads_only_itself() {
        let (offsets, weights) = simple_resize_tables(4, 1);

        assert_eq!(offsets, [0, 0, -1, -1]);
        assert_eq!(weights, [0, 0, WEIGHT_MAX, WEIGHT_MAX]);
        assert_eq!(
            axis_table(4, 1, 6),
            [0, 0, 0, 0, 0, 16384, 0, 16384, 0, 16384, 0, 16384]
        );
    }
}
