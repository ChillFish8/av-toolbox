mod metrics;
mod reference;
mod synthetic;

pub use self::metrics::{psnr, psnr_within, vector_accuracy, write_luma_y4m};
pub use self::reference::{
    ReferenceField,
    half_pel_sample,
    reference_chain_seed,
    reference_clip_pyramids,
    reference_compensate,
    reference_confidence,
    reference_covering_blocks,
    reference_downscale,
    reference_nlm_mc_warp,
    reference_pyramid,
    reference_refine_half_pel,
    reference_refine_seed,
    reference_search,
};
pub(crate) use self::reference::{covering_windows, scaled_vector};
pub(crate) use self::synthetic::render_at;
pub use self::synthetic::{MotionClip, motion_clips, normalised, object_mask, write_motion_inputs};
use crate::motion::{ConfidenceScores, MotionEstimation, MotionSettings};

/// The fields av-denoise wrote for one clip at one setting, each as little-endian bytes.
///
/// Each `.i32` file holds one field after another, two `i32` per block. Each `.f32` file holds one
/// confidence per block. The fixtures' README gives every file's frames.
pub struct SettingFixture {
    pub label: &'static str,
    pub settings: MotionSettings,
    pub refine_radius: u32,
    pub scores: ConfidenceScores,
    pub direct_1: &'static [u8],
    pub direct_2: &'static [u8],
    pub chained_2: &'static [u8],
    pub chained_back_2: &'static [u8],
    pub confidence_1: &'static [u8],
    pub confidence_chained_2: &'static [u8],
}

/// Everything av-denoise wrote for one of the [motion_clips], at commit `dcb9ad9`.
///
/// `pyramid_1` and `pyramid_2` hold levels 1 and 2 of every frame's pyramid, frame after frame.
pub struct MotionFixture {
    pub clip: &'static str,
    pub pyramid_1: &'static [u8],
    pub pyramid_2: &'static [u8],
    pub settings: [SettingFixture; 2],
}

/// The refine radius every chained fixture was written with.
const FIXTURE_REFINE_RADIUS: u32 = 2;

const BLK16: MotionSettings = MotionSettings {
    blksize: 16,
    overlap: 8,
    search_radius: 4,
    pyramid_levels: 2,
    pel: 1,
    estimation: MotionEstimation::Chained {
        refine_radius: FIXTURE_REFINE_RADIUS,
    },
};

const BLK8: MotionSettings = MotionSettings {
    blksize: 8,
    overlap: 4,
    search_radius: 2,
    pyramid_levels: 3,
    pel: 1,
    estimation: MotionEstimation::Chained {
        refine_radius: FIXTURE_REFINE_RADIUS,
    },
};

/// `sad_noise_floor(16, 3 / 255)` and `thsad(16, 1)` as av-denoise worked them out.
const BLK16_SCORES: ConfidenceScores = ConfidenceScores {
    noise_floor: f32::from_bits(0x40597f97),
    thsad: f32::from_bits(0x40a3d70a),
};

/// `sad_noise_floor(8, 3 / 255)` and `thsad(8, 1)` as av-denoise worked them out.
const BLK8_SCORES: ConfidenceScores = ConfidenceScores {
    noise_floor: f32::from_bits(0x3f597f97),
    thsad: f32::from_bits(0x3fa3d70a),
};

macro_rules! setting_fixture {
    ($clip:literal, $label:literal, $settings:expr, $scores:expr) => {
        SettingFixture {
            label: $label,
            settings: $settings,
            refine_radius: FIXTURE_REFINE_RADIUS,
            scores: $scores,
            direct_1: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/direct_1.i32"
            )),
            direct_2: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/direct_2.i32"
            )),
            chained_2: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/chained_2.i32"
            )),
            chained_back_2: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/chained_back_2.i32"
            )),
            confidence_1: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/confidence_1.f32"
            )),
            confidence_chained_2: include_bytes!(concat!(
                "../fixtures/motion/outputs/",
                $clip,
                "/",
                $label,
                "/confidence_chained_2.f32"
            )),
        }
    };
}

macro_rules! motion_fixture {
    ($clip:literal) => {
        MotionFixture {
            clip: $clip,
            pyramid_1: include_bytes!(concat!("../fixtures/motion/outputs/", $clip, "/pyramid_1.f32")),
            pyramid_2: include_bytes!(concat!("../fixtures/motion/outputs/", $clip, "/pyramid_2.f32")),
            settings: [
                setting_fixture!($clip, "blk16", BLK16, BLK16_SCORES),
                setting_fixture!($clip, "blk8", BLK8, BLK8_SCORES),
            ],
        }
    };
}

/// The fixtures of every one of the [motion_clips], in the same order.
pub const MOTION_FIXTURES: [MotionFixture; 6] = [
    motion_fixture!("pan_8"),
    motion_fixture!("pan_10"),
    motion_fixture!("objects_8"),
    motion_fixture!("noisy_8"),
    motion_fixture!("noisy_10"),
    motion_fixture!("cut_8"),
];

/// The little-endian `i32` values in `bytes`.
pub fn fixture_i32s(bytes: &[u8]) -> Vec<i32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "a fixture of i32 values has {} stray bytes",
        remainder.len()
    );

    words.iter().map(|word| i32::from_le_bytes(*word)).collect()
}

/// The little-endian `f32` values in `bytes`.
pub fn fixture_f32s(bytes: &[u8]) -> Vec<f32> {
    let (words, remainder) = bytes.as_chunks::<4>();
    assert!(
        remainder.is_empty(),
        "a fixture of f32 values has {} stray bytes",
        remainder.len()
    );

    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}
