//! Motion search between frames
//!
//! A search measures where each block of a centre frame is found in a neighbouring frame, by the sum of
//! absolute differences over luma normalised to between 0 and 1:
//!
//! - [build_pyramid] halves the luma into smaller levels.
//! - [search] runs a coarse search on the smallest level, seeding a fine search at full resolution.
//! - [chain] joins two adjacent fields into a seed for a neighbour two frames away.
//! - [refine_seed] searches a small window around a seed.
//! - [refine_half_pel] refines whole pixel vectors to half pixels.
//! - [scene_change_flags] marks the fields that are scene changes.
//! - [compensate] lines a neighbour up with the centre frame, blending overlapping blocks.
//! - [sad_mask] turns a field's SADs into MVTools' SAD mask.
//! - [normalise] stretches luma code values to between 0 and 1 for searching.
//!
//! Each search also writes every block's best SAD and, when asked, its confidence.

mod buffers;
mod chain;
mod compensate;
mod confidence;
mod mask;
mod normalise;
mod pyramid;
mod scene;
mod search;
mod subpel;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;
use thiserror::Error;

pub use self::buffers::{FieldBuffers, SceneFlags};
pub use self::chain::chain;
pub use self::compensate::{CompensatePlanes, PlaneScale, compensate, window_weight};
pub(crate) use self::compensate::{block_window, half_pel_component, upper_block};
pub use self::confidence::ConfidenceScores;
pub use self::mask::{PROGRESSIVE_SAD_MASK, sad_mask};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::mask::{
    WEIGHT_HALF,
    WEIGHT_MAX,
    WEIGHT_SHIFT,
    covered_area,
    sad_byte,
    simple_resize_tables,
};
pub use self::normalise::normalise;
pub use self::pyramid::{Pyramid, build_pyramid, level_dims};
pub use self::scene::{SceneChange, SceneFlag, scene_change_flags};
pub(crate) use self::scene::{assert_flag, normalised_threshold};
pub use self::search::{refine_seed, search};
pub(crate) use self::subpel::half_pel_sample;
pub use self::subpel::refine_half_pel;
use crate::DevicePlane;

/// The largest `blksize`.
pub const MAX_BLKSIZE: u32 = 32;

/// The largest `search_radius` and `refine_radius`.
pub const MAX_SEARCH_RADIUS: u32 = 8;

/// The largest `pyramid_levels`.
pub const MAX_PYRAMID_LEVELS: u32 = 3;

/// How a motion search is tuned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MotionSettings {
    /// The side of each block, in pixels at full resolution.
    pub blksize: u32,
    /// How many pixels neighbouring blocks share.
    pub overlap: u32,
    /// How far the search reaches either way, in pixels of the level searched.
    pub search_radius: u32,
    /// How many levels the pyramid has, full resolution included.
    pub pyramid_levels: u32,
    /// The vector precision, 1 for whole pixels and 2 for half pixels.
    pub pel: u32,
    pub estimation: MotionEstimation,
}

impl Default for MotionSettings {
    fn default() -> Self {
        Self {
            blksize: 16,
            overlap: 8,
            search_radius: 4,
            pyramid_levels: 2,
            pel: 2,
            estimation: MotionEstimation::Direct,
        }
    }
}

/// How the motion towards each neighbour is estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MotionEstimation {
    /// Searches every neighbour against the centre frame.
    Direct,
    /// Joins adjacent fields into a seed, then searches `refine_radius` pixels around it.
    Chained { refine_radius: u32 },
}

/// A motion setting outside the range the search supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum MotionError {
    #[error("blksize is {0}, expected an even size between 4 and {MAX_BLKSIZE}")]
    Blksize(u32),
    #[error("overlap is {overlap}, expected at most half of blksize {blksize}")]
    Overlap { overlap: u32, blksize: u32 },
    #[error("search_radius is {0}, expected 1..={MAX_SEARCH_RADIUS}")]
    SearchRadius(u32),
    #[error("pyramid_levels is {0}, expected 1..={MAX_PYRAMID_LEVELS}")]
    PyramidLevels(u32),
    #[error("pel is {0}, expected 1 or 2")]
    Pel(u32),
    #[error("refine_radius is {0}, expected 1..={MAX_SEARCH_RADIUS}")]
    RefineRadius(u32),
}

impl MotionSettings {
    /// Refuses the first setting outside its range.
    pub fn validate(&self) -> Result<(), MotionError> {
        let blksize = self.blksize;
        if !(4..=MAX_BLKSIZE).contains(&blksize) || !blksize.is_multiple_of(2) {
            return Err(MotionError::Blksize(blksize));
        }

        if self.overlap > blksize / 2 {
            let overlap = self.overlap;
            return Err(MotionError::Overlap { overlap, blksize });
        }

        if !(1..=MAX_SEARCH_RADIUS).contains(&self.search_radius) {
            return Err(MotionError::SearchRadius(self.search_radius));
        }

        if !(1..=MAX_PYRAMID_LEVELS).contains(&self.pyramid_levels) {
            return Err(MotionError::PyramidLevels(self.pyramid_levels));
        }

        if !(1..=2).contains(&self.pel) {
            return Err(MotionError::Pel(self.pel));
        }

        if let MotionEstimation::Chained { refine_radius } = self.estimation
            && !(1..=MAX_SEARCH_RADIUS).contains(&refine_radius)
        {
            return Err(MotionError::RefineRadius(refine_radius));
        }

        Ok(())
    }
}

/// The blocks a field holds, `across` by `down`, `blksize` pixels each and `step` pixels apart.
///
/// The last blocks of a row or column run past the frame's edge. A frame smaller than one step still
/// has one block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGrid {
    pub blksize: u32,
    pub step: u32,
    pub across: u32,
    pub down: u32,
}

impl BlockGrid {
    pub fn new(settings: &MotionSettings, width: u32, height: u32) -> Self {
        let step = settings.blksize - settings.overlap;
        let across = width.div_ceil(step).max(1);
        let down = height.div_ceil(step).max(1);

        Self {
            blksize: settings.blksize,
            step,
            across,
            down,
        }
    }

    pub const fn blocks(&self) -> usize {
        self.across as usize * self.down as usize
    }

    /// The width and height of a vector plane, two `i32` per block.
    pub const fn vector_dims(&self) -> (u32, u32) {
        (2 * self.across, self.down)
    }

    /// The width and height of a SAD or confidence plane, one `f32` per block.
    pub const fn sad_dims(&self) -> (u32, u32) {
        (self.across, self.down)
    }
}

/// One field's planes on the device, telling where each block of the centre frame is found in the
/// neighbour.
///
/// `vectors` holds an `(x, y)` pair of `i32` per block, `2 * across` wide and `down` high. `sads` holds
/// each block's best SAD as `f32`, and `confidence` its score, `across` by `down`.
#[derive(Debug, Clone, Copy)]
pub struct FieldPlanes<'a> {
    pub vectors: DevicePlane<'a>,
    pub sads: DevicePlane<'a>,
    pub confidence: Option<DevicePlane<'a>>,
}

/// One centre-to-neighbour field on the device, with its block grid and its vector unit.
#[derive(Debug, Clone, Copy)]
pub struct MotionField<'a> {
    pub planes: FieldPlanes<'a>,
    pub grid: BlockGrid,
    /// The vectors are in units of `1 / pel` pixel.
    pub pel: u32,
}

/// Asserts that `grid` has a step between half its block size and its block size.
fn assert_grid(grid: &BlockGrid) {
    let least = grid.blksize.div_ceil(2);
    assert!(
        (least..=grid.blksize).contains(&grid.step),
        "step is {}, expected {least}..={}",
        grid.step,
        grid.blksize
    );
}

/// Asserts that `field` has a valid pel and its planes hold its grid's blocks.
pub(crate) fn assert_field(field: &MotionField<'_>) {
    let grid = &field.grid;
    let (vector_width, vector_height) = grid.vector_dims();
    let (sad_width, sad_height) = grid.sad_dims();

    assert!(
        (1..=2).contains(&field.pel),
        "pel is {}, expected 1 or 2",
        field.pel
    );
    assert_grid(grid);
    crate::assert_plane(&field.planes.vectors, vector_width, vector_height, "vectors");
    crate::assert_plane(&field.planes.sads, sad_width, sad_height, "sads");

    if let Some(confidence) = &field.planes.confidence {
        crate::assert_plane(confidence, sad_width, sad_height, "confidence");
    }
}

/// Asserts that a field read by `operation` holds whole pixel vectors.
fn assert_whole_pel(pel: u32, operation: &str) {
    assert!(
        pel == 1,
        "{operation} needs a field with pel 1, but its pel is {pel}"
    );
}

/// Asserts that `field` holds whole pixel vectors, which `operation` reads or seeds.
pub(crate) fn assert_whole_pel_field(field: &MotionField<'_>, operation: &str) {
    assert_whole_pel(field.pel, operation);
}

/// Asserts that scores are given exactly when the field has a confidence plane to write them to.
fn assert_scoring(has_confidence: bool, has_scores: bool) {
    assert!(
        has_confidence == has_scores,
        "the field has a confidence plane: {has_confidence}, but scores were given: {has_scores}"
    );
}

/// Asserts that `scores` are given exactly when `field` has a confidence plane.
pub(crate) fn assert_scores_match(field: &MotionField<'_>, scores: Option<ConfidenceScores>) {
    assert_scoring(field.planes.confidence.is_some(), scores.is_some());
}

/// Asserts that the blocks of `grid` cover a plane `width` by `height` scaled by `scale`.
pub(crate) fn assert_blocks(grid: &BlockGrid, (width, height): (u32, u32), scale: PlaneScale) {
    assert_grid(grid);
    assert!(
        scale.shift_x <= 1 && scale.shift_y <= 1,
        "the shifts are {} and {}, expected at most 1 each",
        scale.shift_x,
        scale.shift_y
    );

    let end_x = (((grid.across - 1) * grid.step) >> scale.shift_x) + (grid.blksize >> scale.shift_x);
    let end_y = (((grid.down - 1) * grid.step) >> scale.shift_y) + (grid.blksize >> scale.shift_y);

    assert!(
        end_x >= width && end_y >= height,
        "the grid covers {end_x}x{end_y}, short of the {width}x{height} plane"
    );
}

/// `value` clamped into `0..limit`.
#[cube]
fn clamp_i32(value: i32, limit: i32) -> i32 {
    let mut result = value;
    if value < 0 {
        result = 0;
    } else if value >= limit {
        result = limit - 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refused(settings: MotionSettings) -> String {
        let error = settings.validate().expect_err("the settings should be refused");

        error.to_string()
    }

    #[test]
    fn the_default_settings_are_valid() {
        MotionSettings::default()
            .validate()
            .expect("the defaults should be valid");
    }

    #[test]
    fn invalid_settings_are_refused_naming_the_setting() {
        let defaults = MotionSettings::default();
        let cases = [
            (
                MotionSettings {
                    blksize: 2,
                    ..defaults
                },
                "blksize",
            ),
            (
                MotionSettings {
                    blksize: 34,
                    ..defaults
                },
                "blksize",
            ),
            (
                MotionSettings {
                    blksize: 9,
                    overlap: 0,
                    ..defaults
                },
                "blksize",
            ),
            (
                MotionSettings {
                    overlap: 9,
                    ..defaults
                },
                "overlap",
            ),
            (
                MotionSettings {
                    search_radius: 0,
                    ..defaults
                },
                "search_radius",
            ),
            (
                MotionSettings {
                    search_radius: 9,
                    ..defaults
                },
                "search_radius",
            ),
            (
                MotionSettings {
                    pyramid_levels: 0,
                    ..defaults
                },
                "pyramid_levels",
            ),
            (
                MotionSettings {
                    pyramid_levels: 4,
                    ..defaults
                },
                "pyramid_levels",
            ),
            (MotionSettings { pel: 0, ..defaults }, "pel"),
            (MotionSettings { pel: 4, ..defaults }, "pel"),
            (
                MotionSettings {
                    estimation: MotionEstimation::Chained { refine_radius: 0 },
                    ..defaults
                },
                "refine_radius",
            ),
            (
                MotionSettings {
                    estimation: MotionEstimation::Chained { refine_radius: 9 },
                    ..defaults
                },
                "refine_radius",
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
    #[should_panic(expected = "the shifts are 2 and 0")]
    fn a_plane_scale_shift_of_2_is_refused() {
        let grid = BlockGrid::new(&MotionSettings::default(), 64, 64);
        let scale = PlaneScale {
            shift_x: 2,
            shift_y: 0,
        };

        assert_blocks(&grid, (16, 64), scale);
    }

    #[test]
    #[should_panic(expected = "step is 0")]
    fn a_grid_with_no_step_is_refused() {
        let grid = BlockGrid {
            blksize: 16,
            step: 0,
            across: 1,
            down: 1,
        };

        assert_blocks(&grid, (16, 16), PlaneScale::LUMA);
    }

    #[test]
    fn subsampling_factors_give_the_shifts() {
        let scale = PlaneScale::from_factors((2, 1));

        assert_eq!((scale.shift_x, scale.shift_y), (1, 0));
        assert_eq!(PlaneScale::from_factors((1, 1)), PlaneScale::LUMA);
    }

    #[test]
    #[should_panic(expected = "factor")]
    fn a_subsampling_factor_of_4_is_refused() {
        PlaneScale::from_factors((4, 1));
    }

    #[test]
    fn half_a_block_of_overlap_is_valid() {
        let settings = MotionSettings {
            blksize: 8,
            overlap: 4,
            ..MotionSettings::default()
        };

        settings.validate().expect("half a block should be valid");
    }

    #[test]
    fn vector_and_sad_dims_follow_the_grid() {
        let grid = BlockGrid {
            blksize: 16,
            step: 8,
            across: 16,
            down: 8,
        };

        assert_eq!(grid.vector_dims(), (32, 8));
        assert_eq!(grid.sad_dims(), (16, 8));
    }

    #[test]
    #[should_panic(expected = "refine_half_pel needs a field with pel 1, but its pel is 2")]
    fn refining_a_half_pel_field_again_is_refused() {
        assert_whole_pel(2, "refine_half_pel");
    }

    #[test]
    #[should_panic(expected = "confidence plane: false, but scores were given: true")]
    fn scores_without_a_confidence_plane_are_refused() {
        assert_scoring(false, true);
    }

    #[test]
    #[should_panic(expected = "confidence plane: true, but scores were given: false")]
    fn a_confidence_plane_without_scores_is_refused() {
        assert_scoring(true, false);
    }

    #[test]
    fn a_frame_smaller_than_a_step_has_one_block() {
        let grid = BlockGrid::new(&MotionSettings::default(), 5, 3);

        assert_eq!((grid.across, grid.down), (1, 1));
    }

    #[test]
    fn the_last_blocks_run_past_the_edge() {
        let grid = BlockGrid::new(&MotionSettings::default(), 129, 64);

        assert_eq!((grid.across, grid.down, grid.step), (17, 8, 8));
    }
}
