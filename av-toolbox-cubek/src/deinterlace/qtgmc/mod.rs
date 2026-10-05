//! The kernels QTGMC is built from.
//!
//! - [bob] resizes a field to a full-height plane.
//! - [weighted_sum] averages bobs as QTGMC's TR0 does.
//! - [shimmer] keeps only thin horizontal differences from a reference.
//! - [search_clip], [resharpen], [subtract], [clamp_between] and [merge] are QTGMC's per-pixel steps.
//! - [fmtc_gauss_taps] builds the gaussian QTGMC blurs with.

mod average;
mod bob;
mod pointwise;
mod shimmer;

pub use self::average::{MAX_AVERAGED, tr0_weights, weighted_sum};
pub use self::bob::{BobPlanes, Parity, bob, bob_taps};
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
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::shimmer::{NEGATIVE_STEPS, POSITIVE_STEPS, restore_limits};
pub use self::shimmer::{ShimmerPlanes, shimmer};
