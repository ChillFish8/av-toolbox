// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use thiserror::Error;

use crate::motion::{
    MotionField,
    PlaneScale,
    SceneChange,
    SceneFlag,
    assert_blocks,
    assert_field,
    assert_flag,
    block_window,
    half_pel_component,
    half_pel_sample,
    normalised_threshold,
    upper_block,
};
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

/// A degrain setting outside the range it supports.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum DegrainError {
    #[error("temporal_radius is {0}, expected 1..={MAX_TEMPORAL_RADIUS}")]
    TemporalRadius(u32),
    #[error("thsad is {0}, expected a finite value above 0")]
    Thsad(f32),
    #[error("thsad_chroma is {0}, expected a finite value above 0")]
    ThsadChroma(f32),
    #[error("thscd1 is {0}, expected a finite value above 0")]
    Thscd1(f32),
    #[error("thscd2 is {0}, expected at most 255")]
    Thscd2(u32),
}

/// Whether a threshold is finite and above 0.
fn is_positive_threshold(value: f32) -> bool {
    value.is_finite() && value > 0.0
}

impl DegrainSettings {
    /// Refuses the first setting outside its range.
    pub fn validate(&self) -> Result<(), DegrainError> {
        if !(1..=MAX_TEMPORAL_RADIUS).contains(&self.temporal_radius) {
            return Err(DegrainError::TemporalRadius(self.temporal_radius));
        }

        if !is_positive_threshold(self.thsad) {
            return Err(DegrainError::Thsad(self.thsad));
        }

        if !is_positive_threshold(self.thsad_chroma) {
            return Err(DegrainError::ThsadChroma(self.thsad_chroma));
        }

        if !is_positive_threshold(self.thscd1) {
            return Err(DegrainError::Thscd1(self.thscd1));
        }

        if self.thscd2 > 255 {
            return Err(DegrainError::Thscd2(self.thscd2));
        }

        Ok(())
    }

    /// The luma or chroma threshold as a normalised block SAD.
    pub fn sad_threshold(&self, blksize: u32, chroma: bool) -> f32 {
        let value = if chroma { self.thsad_chroma } else { self.thsad };

        normalised_threshold(value, blksize)
    }

    /// The rule that marks a neighbour as across a scene change.
    pub const fn scene_change(&self) -> SceneChange {
        SceneChange {
            thscd1: self.thscd1,
            thscd2: self.thscd2,
        }
    }
}

/// One neighbour frame, the field finding the centre's blocks in it and the field's scene-change flag.
///
/// The field runs from the centre to the neighbour, for frames before and after the centre alike.
pub struct DegrainNeighbour<'a> {
    pub plane: DevicePlane<'a>,
    pub field: MotionField<'a>,
    pub flag: SceneFlag<'a>,
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
/// Every term is weighted by its block's window, and a neighbour's also by its block's SAD. A flagged
/// neighbour weighs nothing.
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
    flags: &Sequence<Array<u32>>,
    flag_indices: &Sequence<u32>,
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
                        let flagged = flags.index(index)[*flag_indices.index(index) as usize] != 0u32;
                        let mut weight = 0.0f32;
                        if !flagged {
                            let sad = sads.index(index)[block];
                            weight = window * device_degrain_weight(sad, threshold);
                        }

                        let raw_x = vectors.index(index)[block * 2];
                        let raw_y = vectors.index(index)[block * 2 + 1];
                        let vector_x = half_pel_component(raw_x, shift_x, pel);
                        let vector_y = half_pel_component(raw_y, shift_y, pel);
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

#[cube(launch_unchecked)]
fn copy_kernel(input: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

/// Writes `centre` into `output`, which both hold at `centre`'s size.
fn copy_centre<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let centre_array = centre.array::<R>();
    let output_array = output.array::<R>();

    // Safety: both planes were checked to hold the same number of f32 samples, and each unit reads and
    // writes only its own sample.
    unsafe {
        copy_kernel::launch_unchecked::<R>(client, cubes, unit, centre_array, output_array);
    }
}

/// Averages `centre` with its compensated `neighbours` into `output`.
///
/// Each sample takes the blocks of the neighbours' shared grid covering it. A block weighs its window
/// times 256 for the centre, and its window times its neighbour's [degrain_weight] for each neighbour. The
/// sample is the weighted mean of the centre and each neighbour read at the block's vector. A neighbour
/// whose flag is set weighs nothing.
///
/// `chroma` picks `thsad_chroma` over `thsad`. The vectors and SADs are the luma fields', with `scale`
/// shrinking the blocks and vectors onto a smaller plane. All the planes read and written share one scale
/// of values. Every neighbour's field shares one grid and one `pel`. With no neighbours, `output`
/// is `centre`.
#[expect(
    clippy::too_many_arguments,
    reason = "degrain takes each plane and setting as its own argument"
)]
pub fn degrain<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    centre: DevicePlane<'_>,
    neighbours: &[DegrainNeighbour<'_>],
    output: DevicePlane<'_>,
    settings: &DegrainSettings,
    scale: PlaneScale,
    chroma: bool,
) {
    debug_assert_eq!(settings.validate(), Ok(()));

    let radius = settings.temporal_radius;
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

    let Some(first) = neighbours.first() else {
        copy_centre::<R>(client, dispatch, centre, output);
        return;
    };
    let grid = &first.field.grid;
    let pel = first.field.pel;
    assert_blocks(grid, (width, height), scale);

    let mut planes = SequenceArg::<R, Array<f32>>::new();
    let mut vectors = SequenceArg::<R, Array<i32>>::new();
    let mut sads = SequenceArg::<R, Array<f32>>::new();
    let mut flags = SequenceArg::<R, Array<u32>>::new();
    let mut flag_indices = SequenceArg::<R, u32>::new();
    for neighbour in neighbours {
        crate::assert_plane(&neighbour.plane, width, height, "neighbour");
        assert_field(&neighbour.field);
        assert_flag(&neighbour.flag);
        assert_eq!(neighbour.field.grid, *grid, "the neighbours' grids differ");
        assert_eq!(neighbour.field.pel, pel, "the neighbours' pels differ");

        planes.push(neighbour.plane.array::<R>());
        vectors.push(neighbour.field.planes.vectors.array::<R>());
        sads.push(neighbour.field.planes.sads.array::<R>());
        flags.push(neighbour.flag.flags.array::<R>());
        flag_indices.push(neighbour.flag.index);
    }

    let threshold = settings.sad_threshold(grid.blksize, chroma);
    let cubes = dispatch.cube_count(output.samples());
    let unit = dispatch.cube_dim();
    let centre_array = centre.array::<R>();
    let output_array = output.array::<R>();

    // Safety: every plane holds `width` by `height` samples that each read clamps into, the fields hold a
    // vector pair and a SAD for each block of `grid`, each flag index is inside its plane, and each
    // unit writes only its own output sample.
    unsafe {
        degrain_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            centre_array,
            planes,
            vectors,
            sads,
            flags,
            flag_indices,
            output_array,
            threshold,
            neighbours.len() as u32,
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

    fn refused(settings: DegrainSettings) -> String {
        let error = settings.validate().expect_err("the settings should be refused");

        error.to_string()
    }

    #[test]
    fn the_default_settings_are_valid() {
        DegrainSettings::default()
            .validate()
            .expect("the defaults should be valid");
    }

    #[test]
    fn invalid_settings_are_refused_naming_the_setting() {
        let defaults = DegrainSettings::default();
        let cases = [
            (
                DegrainSettings {
                    temporal_radius: 0,
                    ..defaults
                },
                "temporal_radius",
            ),
            (
                DegrainSettings {
                    temporal_radius: MAX_TEMPORAL_RADIUS + 1,
                    ..defaults
                },
                "temporal_radius",
            ),
            (
                DegrainSettings {
                    thsad: 0.0,
                    ..defaults
                },
                "thsad is",
            ),
            (
                DegrainSettings {
                    thsad: -1.0,
                    ..defaults
                },
                "thsad is",
            ),
            (
                DegrainSettings {
                    thsad: f32::NAN,
                    ..defaults
                },
                "thsad is",
            ),
            (
                DegrainSettings {
                    thsad: f32::INFINITY,
                    ..defaults
                },
                "thsad is",
            ),
            (
                DegrainSettings {
                    thsad_chroma: 0.0,
                    ..defaults
                },
                "thsad_chroma",
            ),
            (
                DegrainSettings {
                    thsad_chroma: f32::NAN,
                    ..defaults
                },
                "thsad_chroma",
            ),
            (
                DegrainSettings {
                    thscd1: 0.0,
                    ..defaults
                },
                "thscd1",
            ),
            (
                DegrainSettings {
                    thscd1: f32::INFINITY,
                    ..defaults
                },
                "thscd1",
            ),
            (
                DegrainSettings {
                    thscd2: 256,
                    ..defaults
                },
                "thscd2",
            ),
        ];

        for (settings, setting) in cases {
            let message = refused(settings);

            assert!(
                message.starts_with(setting),
                "{settings:?} was refused with `{message}`"
            );
        }
    }

    #[test]
    fn the_largest_scene_change_share_is_valid() {
        let settings = DegrainSettings {
            thscd2: 255,
            temporal_radius: MAX_TEMPORAL_RADIUS,
            ..DegrainSettings::default()
        };

        settings.validate().expect("the largest values should be valid");
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
}
