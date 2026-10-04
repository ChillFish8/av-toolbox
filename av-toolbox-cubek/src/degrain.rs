use cubecl::prelude::*;

use crate::motion::{BlockGrid, PlaneScale, assert_blocks, block_window, half_pel_sample, upper_block};
use crate::{DevicePlane, Dispatch};

/// The most neighbours on each side of the centre frame.
pub const MAX_TEMPORAL_RADIUS: u32 = 6;

/// How degrain weighs and picks its neighbours.
///
/// `thsad`, `thsad_chroma` and `thscd1` are per 8x8 block on the 8-bit scale. The defaults are MVTools'.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegrainSettings {
    /// How many frames on each side of the centre are averaged, between 1 and [MAX_TEMPORAL_RADIUS].
    pub temporal_radius: u32,
    /// The SAD at which a luma block's neighbour stops counting.
    pub thsad: f32,
    /// The SAD at which a chroma block's neighbour stops counting.
    pub thsad_chroma: f32,
    /// The SAD above which a block counts as changed between scenes.
    pub thscd1: f32,
    /// How many of every 255 blocks may change before the whole neighbour is a scene change.
    pub thscd2: u32,
}

impl Default for DegrainSettings {
    fn default() -> Self {
        Self {
            temporal_radius: 1,
            thsad: 400.0,
            thsad_chroma: 400.0,
            thscd1: 400.0,
            thscd2: 130,
        }
    }
}

impl DegrainSettings {
    /// The luma or chroma threshold as a normalised block SAD.
    pub fn sad_threshold(&self, blksize: u32, chroma: bool) -> f32 {
        let value = if chroma { self.thsad_chroma } else { self.thsad };

        normalised_threshold(value, blksize)
    }
}

/// An 8-bit, 8x8 block threshold as a normalised SAD of a block `blksize` across.
fn normalised_threshold(value: f32, blksize: u32) -> f32 {
    let samples = (blksize * blksize) as f32;

    value * samples / 64.0 / 255.0
}

/// One neighbour frame and the field finding the centre's blocks in it.
///
/// `vectors` and `sads` are laid out as in [FieldPlanes](crate::motion::FieldPlanes). The field runs from the centre
/// to the neighbour, for frames before and after the centre alike.
pub struct DegrainNeighbour<'a> {
    pub plane: DevicePlane<'a>,
    pub vectors: DevicePlane<'a>,
    pub sads: DevicePlane<'a>,
}

/// Whether a neighbour is across a scene change from the centre, given its blocks' normalised `sads`.
///
/// It is when more than `thscd2` of every 255 blocks have a SAD above `thscd1`.
pub fn scene_changed(sads: &[f32], grid: &BlockGrid, settings: &DegrainSettings) -> bool {
    let threshold = normalised_threshold(settings.thscd1, grid.blksize);
    let changed = sads.iter().filter(|sad| **sad > threshold).count() as u64;
    let allowed = u64::from(settings.thscd2) * grid.blocks() as u64;

    changed * 255 > allowed
}

/// The weight of a block with `sad` against `threshold`, on a scale where the centre weighs 256.
///
/// It falls from 256 at a SAD of 0 to 0 at `threshold`, and stays 0 beyond.
pub fn degrain_weight(sad: f32, threshold: f32) -> f32 {
    if sad >= threshold {
        return 0.0;
    }

    let threshold_squared = threshold * threshold;
    let sad_squared = sad * sad;

    256.0 * (threshold_squared - sad_squared) / (threshold_squared + sad_squared)
}

/// The same weight as [degrain_weight], computed with the same operations on the device.
#[cube]
fn device_degrain_weight(sad: f32, threshold: f32) -> f32 {
    let mut weight = 0.0f32;

    if sad < threshold {
        let threshold_squared = threshold * threshold;
        let sad_squared = sad * sad;
        weight = 256.0f32 * (threshold_squared - sad_squared) / (threshold_squared + sad_squared);
    }

    weight
}

/// Averages the centre sample with each neighbour's compensated sample, over the blocks covering it.
///
/// Every term is weighted by its block's window, and a neighbour's also by its block's SAD.
#[expect(
    clippy::too_many_arguments,
    reason = "a kernel takes each plane and setting as its own argument"
)]
#[cube(launch_unchecked)]
fn degrain_kernel(
    centre: &Array<f32>,
    planes: &Sequence<Array<f32>>,
    vectors: &Sequence<Array<i32>>,
    sads: &Sequence<Array<f32>>,
    output: &mut Array<f32>,
    threshold: f32,
    #[comptime] count: u32,
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
    let half_units = comptime!((2 / pel) as i32);

    let upper_x = upper_block(x, step, across, shift_x);
    let upper_y = upper_block(y, step, down, shift_y);

    let centre_value = centre[ABSOLUTE_POS];
    let mut weight_sum = 0.0f32;
    let mut value_sum = 0.0f32;
    for back_y in 0..2u32 {
        for back_x in 0..2u32 {
            if upper_x >= back_x && upper_y >= back_y {
                let block_x = upper_x - back_x;
                let block_y = upper_y - back_y;
                let window = block_window(
                    x, y, block_x, block_y, blksize, step, across, down, shift_x, shift_y,
                );

                if window > 0.0f32 {
                    let block = (block_y * across + block_x) as usize;
                    let centre_weight = 256.0f32 * window;
                    weight_sum += centre_weight;
                    value_sum += centre_weight * centre_value;

                    #[unroll]
                    for neighbour in 0..count {
                        let index = neighbour as usize;
                        let sad = sads.index(index)[block];
                        let weight = window * device_degrain_weight(sad, threshold);

                        let vector_x = (vectors.index(index)[block * 2] >> shift_x as i32) * half_units;
                        let vector_y = (vectors.index(index)[block * 2 + 1] >> shift_y as i32) * half_units;
                        let value = half_pel_sample(
                            planes.index(index),
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
    }

    output[ABSOLUTE_POS] = value_sum / weight_sum;
}

/// Averages `centre` with its compensated `neighbours` into `output`.
///
/// Each sample takes the blocks of `grid` covering it. A block
/// weighs its window times 256 for the centre, and its window times its neighbour's
/// [degrain_weight] for each neighbour. The sample is the weighted mean of the centre and each neighbour
/// read at the block's vector. `weights_on` says which neighbours count, and a neighbour that is off weighs
/// nothing.
///
/// `chroma` picks `thsad_chroma` over `thsad`. The vectors and SADs are the luma field's, with `scale`
/// shrinking the blocks and vectors onto a smaller plane. All the planes read and written share one scale
/// of values.
#[expect(
    clippy::too_many_arguments,
    reason = "degrain takes each plane and setting as its own argument"
)]
pub fn degrain<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: DevicePlane<'_>,
    neighbours: &[DegrainNeighbour<'_>],
    weights_on: &[bool],
    output: DevicePlane<'_>,
    grid: &BlockGrid,
    settings: &DegrainSettings,
    pel: u32,
    scale: PlaneScale,
    chroma: bool,
) {
    assert!((1..=2).contains(&pel), "pel is {pel}, expected 1 or 2");

    let radius = settings.temporal_radius;
    assert!(
        (1..=MAX_TEMPORAL_RADIUS).contains(&radius),
        "temporal_radius is {radius}, expected 1..={MAX_TEMPORAL_RADIUS}"
    );
    assert_eq!(
        neighbours.len(),
        weights_on.len(),
        "{} neighbours with {} weights_on",
        neighbours.len(),
        weights_on.len()
    );
    assert!(
        neighbours.len() <= 2 * radius as usize,
        "{} neighbours, expected at most {}",
        neighbours.len(),
        2 * radius
    );

    let width = centre.width();
    let height = centre.height();
    crate::assert_plane(&centre, width, height, "centre");
    crate::assert_plane(&output, width, height, "output");
    assert_blocks(grid, (width, height), scale);

    let mut planes = SequenceArg::<R, Array<f32>>::new();
    let mut vectors = SequenceArg::<R, Array<i32>>::new();
    let mut sads = SequenceArg::<R, Array<f32>>::new();
    let mut counted = 0u32;
    for (neighbour, on) in neighbours.iter().zip(weights_on) {
        crate::assert_plane(&neighbour.plane, width, height, "neighbour");
        crate::assert_plane(&neighbour.vectors, 2 * grid.across, grid.down, "vectors");
        crate::assert_plane(&neighbour.sads, grid.across, grid.down, "sads");

        if *on {
            planes.push(neighbour.plane.array::<R>());
            vectors.push(neighbour.vectors.array::<R>());
            sads.push(neighbour.sads.array::<R>());
            counted += 1;
        }
    }

    let threshold = settings.sad_threshold(grid.blksize, chroma);
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let centre_array = centre.array::<R>();
    let output_array = output.array::<R>();

    // Safety: every plane holds `width` by `height` samples that each read clamps into, the fields hold a
    // vector pair and a SAD for each block of `grid`, and each unit writes only its own output sample.
    unsafe {
        degrain_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            centre_array,
            planes,
            vectors,
            sads,
            output_array,
            threshold,
            counted,
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

    fn grid_of_blocks(blocks: u32) -> BlockGrid {
        BlockGrid {
            blksize: 8,
            step: 8,
            across: blocks,
            down: 1,
        }
    }

    #[test]
    fn a_perfect_match_weighs_256_and_the_threshold_weighs_nothing() {
        assert_eq!(degrain_weight(0.0, 4.0), 256.0);
        assert_eq!(degrain_weight(4.0, 4.0), 0.0);
        assert_eq!(degrain_weight(9.0, 4.0), 0.0);
    }

    #[test]
    fn half_the_threshold_weighs_153_6() {
        assert_eq!(degrain_weight(2.0, 4.0), 256.0 * 12.0 / 20.0);
    }

    #[test]
    fn thresholds_scale_with_the_block_area() {
        let settings = DegrainSettings::default();

        assert_eq!(settings.sad_threshold(8, false), 400.0 / 255.0);
        assert_eq!(settings.sad_threshold(16, false), 4.0 * 400.0 / 255.0);
    }

    #[test]
    fn a_scene_change_needs_more_than_thscd2_of_255_blocks() {
        let settings = DegrainSettings {
            thscd2: 255,
            ..DegrainSettings::default()
        };
        let grid = grid_of_blocks(4);
        let high = 100.0;

        assert!(!scene_changed(&[high; 4], &grid, &settings));

        let strict = DegrainSettings {
            thscd2: 127,
            ..DegrainSettings::default()
        };
        assert!(scene_changed(&[high, high, 0.0, 0.0], &grid, &strict));
        assert!(!scene_changed(&[high, 0.0, 0.0, 0.0], &grid, &strict));
    }
}
