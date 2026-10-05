//! The kernels QTGMC is built from.
//!
//! - [bob] resizes a field to a full-height plane.
//! - [weighted_sum] averages bobs as QTGMC's TR0 does.
//! - [shimmer] keeps only thin horizontal differences from a reference.

mod average;
mod bob;
mod shimmer;

pub use self::average::{MAX_AVERAGED, tr0_weights, weighted_sum};
pub use self::bob::{BobPlanes, Parity, bob, bob_taps};
#[cfg(any(test, feature = "test-support"))]
pub(crate) use self::shimmer::{NEGATIVE_STEPS, POSITIVE_STEPS, restore_limits};
pub use self::shimmer::{ShimmerPlanes, shimmer};
