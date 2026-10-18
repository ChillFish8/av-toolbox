//! The kernels QTGMC is built from.
//!
//! - [bob] resizes a field to a full-height plane.
//! - [weighted_sum] averages bobs as QTGMC's TR0 does.
//! - [clamped_difference] stores a difference as `std.MakeDiff` does.
//! - [temporal_limits] bounds a plane by its compensated neighbours.
//! - [shimmer] keeps only thin horizontal differences from a reference.
//! - [search_clip] builds the clip motion is searched on.
//! - [resharpen], [subtract], [clamp_between] and [merge] are QTGMC's other per-pixel steps.
//! - [search_clip_from_repaired], [smode2_resharpen] and [back_blend] compose those into QTGMC's steps.
//! - [QtgmcTables] holds the bob and gaussian tables those steps read.
//! - [blur_and_spread] blurs with the 3x3 binomial and then the gaussian of a [GaussTables].
//! - [vertical_midpoint] builds the plane SMode 2 blurs before resharpening.
//! - [vertical_blur] and [repair_blend] are the steps repair mode adds.
//! - [sharpness_strength] gives the strength [resharpen] is run with.
//! - [fmtc_gauss_taps] builds the gaussian QTGMC blurs with.
//! - [havsfunc_scale] converts a level given for 8 bit video to a code value.

mod average;
mod bob;
mod difference;
mod limits;
mod pointwise;
mod repair;
mod shimmer;
mod spread;
mod steps;
mod tables;

pub use self::average::{MAX_AVERAGED, tr0_taps, tr0_weights, weighted_sum};
pub use self::bob::{BobPlanes, Parity, bob, bob_taps};
pub use self::difference::clamped_difference;
pub use self::limits::temporal_limits;
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::pointwise::search_levels;
pub use self::pointwise::{
    clamp_between,
    fmtc_gauss_taps,
    merge,
    resharpen,
    search_clip,
    sharpness_strength,
    subtract,
    vertical_midpoint,
};
pub use self::repair::{repair_blend, vertical_blur};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::shimmer::{NEGATIVE_STEPS, POSITIVE_STEPS, restore_limits};
pub use self::shimmer::{ShimmerPlanes, shimmer};
pub use self::spread::{GaussTables, blur_and_spread};
pub use self::steps::{
    BackBlendPlanes,
    SearchPlanes,
    StepScratch,
    back_blend,
    search_clip_from_repaired,
    smode2_resharpen,
};
pub use self::tables::{BobWeights, QtgmcTables};

/// The code value of a level given for 8 bit video, as havsfunc's `scale` rounds it.
///
/// havsfunc (r33, line 5906) defines `scale(value, peak)` as `cround(value * peak / 255)` for a `peak`
/// other than 1, and `cround` rounds half away from zero.
pub fn havsfunc_scale(level: f32, peak: f32) -> f32 {
    let scaled = level * peak / 255.0;

    if scaled > 0.0 {
        (scaled + 0.5).floor()
    } else {
        (scaled - 0.5).ceil()
    }
}

/// Asserts that `bits` is a sample depth the QTGMC kernels take.
pub(crate) fn assert_bits(bits: u32) {
    assert!((8..=16).contains(&bits), "{bits} bits, expected 8..=16");
}
