use av_toolbox_core::StageError;
use av_toolbox_cubek::degrain::{DegrainError, DegrainSettings};
use av_toolbox_cubek::motion::{MotionError, MotionSettings, SceneChange, level_dims};
use av_toolbox_gpu::FrameFormat;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::checks::{at_least, between_u32};
use crate::deinterlace::bwdif::{Order, Rate};
use crate::deinterlace::qtgmc::STAGE;
use crate::deinterlace::spatial::{Eedi3Spatial, Spatial, SpatialStage};

/// The SAD above which a block counts as changed between scenes, on the 8-bit scale.
const SCENE_CHANGE_SAD: f32 = 180.0;

/// How many of every 255 blocks may change before a neighbour counts as a scene change.
const SCENE_CHANGE_BLOCKS: u32 = 98;

/// Names `deinterlace/qtgmc` in a refusal of its `spatial`.
#[derive(Debug)]
pub(crate) struct QtgmcSpatial;

impl SpatialStage for QtgmcSpatial {
    const STAGE: &'static str = STAGE;
    const DEFAULT_MDIS: u32 = 8;
}

/// Arguments for `deinterlace/qtgmc`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct QtgmcArgs {
    /// The field shown first, the stream's own field order when not given.
    #[serde(default)]
    pub(crate) order: Option<Order>,
    /// One frame for each frame in, or one for each field. One for each field when not given.
    #[serde(default)]
    pub(crate) rate: Rate,
    /// Where each rebuilt row's spatial prediction comes from, EEDI3 with a `mdis` of 8 when not given.
    ///
    /// `cubic` is a plain bicubic resize of each field, QTGMC's `bob`.
    #[serde(default = "default_spatial")]
    pub(crate) spatial: Spatial<QtgmcSpatial>,
    /// The temporal radius of the smoothing that makes the motion search clip, between 0 and 2.
    #[serde(default = "default_tr0")]
    pub(crate) tr0: u32,
    /// The temporal radius of the motion compensated smoothing of the interpolated fields, between 0 and 2.
    #[serde(default = "default_tr1")]
    pub(crate) tr1: u32,
    /// The temporal radius of the final smoothing, between 0 and 3.
    #[serde(default = "default_tr2")]
    pub(crate) tr2: u32,
    /// How strongly the result is sharpened, 0 or more. 0 turns the sharpening off.
    #[serde(default = "default_sharpness")]
    pub(crate) sharpness: f32,
    /// The SAD up to which a block is still averaged in the `tr1` smoothing, above 0.
    ///
    /// It is on the 8-bit scale of an 8x8 block.
    #[serde(default = "default_thsad1")]
    pub(crate) thsad1: f32,
    /// The SAD up to which a block is still averaged in the `tr2` smoothing, above 0.
    ///
    /// It is on the 8-bit scale of an 8x8 block.
    #[serde(default = "default_thsad2")]
    pub(crate) thsad2: f32,
}

fn default_spatial() -> Spatial<QtgmcSpatial> {
    Spatial::Eedi3(Eedi3Spatial::defaults())
}

fn default_tr0() -> u32 {
    2
}

fn default_tr1() -> u32 {
    2
}

fn default_tr2() -> u32 {
    1
}

fn default_sharpness() -> f32 {
    1.0
}

fn default_thsad1() -> f32 {
    640.0
}

fn default_thsad2() -> f32 {
    256.0
}

fn bad_threshold(name: &str, value: f32) -> StageError {
    let message = format!("{STAGE} needs a finite {name} above 0, got {value}");

    StageError::msg(message)
}

fn refused_motion(error: MotionError) -> StageError {
    let message = format!("{STAGE} was given motion settings it cannot search with: {error}");

    StageError::msg(message)
}

fn refused_degrain(error: DegrainError) -> StageError {
    let message = format!("{STAGE} was given smoothing settings it cannot run with: {error}");

    StageError::msg(message)
}

fn field_too_small(blksize: u32, width: u32, height: u32) -> StageError {
    let message =
        format!("{STAGE} needs fields of at least one {blksize} pixel block, got fields of {width}x{height}");

    StageError::msg(message)
}

fn pyramid_too_small(levels: u32, width: u32, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs fields large enough for a {levels} level motion pyramid, got {width}x{height}"
    );

    StageError::msg(message)
}

/// The scene-change rule of the compensation and both degrains.
pub(crate) const fn scene_change() -> SceneChange {
    SceneChange {
        thscd1: SCENE_CHANGE_SAD,
        thscd2: SCENE_CHANGE_BLOCKS,
    }
}

/// The motion search settings every QTGMC search runs with.
pub(crate) fn motion_settings() -> MotionSettings {
    MotionSettings::default()
}

/// The settings of a degrain averaging `temporal_radius` fields either side, below `thsad`.
pub(crate) fn degrain_settings(temporal_radius: u32, thsad: f32) -> DegrainSettings {
    DegrainSettings {
        temporal_radius,
        thsad,
        thsad_chroma: thsad,
        thscd1: SCENE_CHANGE_SAD,
        thscd2: SCENE_CHANGE_BLOCKS,
    }
}

impl QtgmcArgs {
    /// How far apart the furthest motion neighbours are, in fields.
    pub(crate) fn max_tr(&self) -> u32 {
        self.tr1.max(self.tr2).max(1)
    }

    /// How many fields the output trails the input by.
    ///
    /// A field's output waits on the limits of the fields `tr2` after it, and they wait on their motion
    /// search, which reaches `max_tr` fields further, on search clips that average `tr0` fields further.
    pub(crate) fn lag(&self) -> u32 {
        self.tr0 + self.tr2 + self.max_tr()
    }

    /// How many input frames the stage holds.
    pub(crate) fn held_frames(&self) -> usize {
        self.lag() as usize / 2 + 2
    }

    /// Refuses the first argument outside its range, and the first motion or degrain setting built from
    /// them that fails its own checks.
    pub(crate) fn checked_values(&self) -> Result<(), StageError> {
        between_u32(STAGE, "tr0", self.tr0, 0, 2)?;
        between_u32(STAGE, "tr1", self.tr1, 0, 2)?;
        between_u32(STAGE, "tr2", self.tr2, 0, 3)?;
        at_least(STAGE, "sharpness", self.sharpness, 0.0)?;

        if !self.thsad1.is_finite() || self.thsad1 <= 0.0 {
            return Err(bad_threshold("thsad1", self.thsad1));
        }

        if !self.thsad2.is_finite() || self.thsad2 <= 0.0 {
            return Err(bad_threshold("thsad2", self.thsad2));
        }

        motion_settings().validate().map_err(refused_motion)?;

        let first = degrain_settings(1, self.thsad1);
        first.validate().map_err(refused_degrain)?;

        let last = degrain_settings(self.tr2.max(1), self.thsad2);
        last.validate().map_err(refused_degrain)
    }
}

/// Refuses an `input` whose fields are smaller than a motion block, or than the pyramid can search.
pub(crate) fn checked_field_size(input: &FrameFormat) -> Result<(), StageError> {
    let settings = motion_settings();
    let blksize = settings.blksize;
    let width = input.width;
    let height = input.height / 2;
    if width < blksize || height < blksize {
        return Err(field_too_small(blksize, width, height));
    }

    let coarse_level = settings.pyramid_levels - 1;
    let coarse_block = (blksize >> coarse_level).max(2);
    let (coarse_width, coarse_height) = level_dims(width, height, coarse_level);
    if coarse_width < coarse_block || coarse_height < coarse_block {
        return Err(pyramid_too_small(settings.pyramid_levels, width, height));
    }

    Ok(())
}
