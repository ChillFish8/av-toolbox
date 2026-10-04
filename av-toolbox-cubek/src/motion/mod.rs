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
//!
//! Each search also writes every block's best SAD and, when asked, its confidence.

mod chain;
mod confidence;
mod pyramid;
mod search;
mod subpel;

use cubecl::prelude::*;
use thiserror::Error;

pub use self::chain::chain;
pub use self::confidence::ConfidenceScores;
pub use self::pyramid::{Pyramid, build_pyramid, level_dims};
pub use self::search::{refine_seed, search};
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
}

/// One field's planes on the device, telling where each block of the centre frame is found in the
/// neighbour.
///
/// `vectors` holds an `(x, y)` pair of `i32` per block, in whole pixels or, after half pixel refinement,
/// half pixels, so it is `2 * across` wide and
/// `down` high. `sads` holds each block's best SAD as `f32`, and `confidence` its score, `across` by
/// `down`.
pub struct FieldPlanes<'a> {
    pub vectors: DevicePlane<'a>,
    pub sads: DevicePlane<'a>,
    pub confidence: Option<DevicePlane<'a>>,
}

/// Asserts that `field` holds `grid`'s blocks.
fn assert_field(field: &FieldPlanes<'_>, grid: &BlockGrid) {
    crate::assert_plane(&field.vectors, 2 * grid.across, grid.down, "vectors");
    crate::assert_plane(&field.sads, grid.across, grid.down, "sads");

    if let Some(confidence) = &field.confidence {
        crate::assert_plane(confidence, grid.across, grid.down, "confidence");
    }
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
    fn half_a_block_of_overlap_is_valid() {
        let settings = MotionSettings {
            blksize: 8,
            overlap: 4,
            ..MotionSettings::default()
        };

        settings.validate().expect("half a block should be valid");
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
