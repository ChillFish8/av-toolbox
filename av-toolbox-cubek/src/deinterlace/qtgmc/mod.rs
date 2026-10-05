//! The kernels QTGMC is built from.
//!
//! - [bob] resizes a field to a full-height plane.

mod bob;

pub use self::bob::{BobPlanes, Parity, bob, bob_taps};
