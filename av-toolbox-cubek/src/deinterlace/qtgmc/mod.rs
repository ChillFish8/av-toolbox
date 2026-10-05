//! The kernels QTGMC is built from.
//!
//! - [bob] resizes a field to a full-height plane.
//! - [weighted_sum] averages bobs as QTGMC's TR0 does.

mod average;
mod bob;

pub use self::average::{MAX_AVERAGED, tr0_weights, weighted_sum};
pub use self::bob::{BobPlanes, Parity, bob, bob_taps};
