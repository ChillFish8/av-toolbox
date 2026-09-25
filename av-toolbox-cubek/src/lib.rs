//! Shared CubeCL kernels for av-toolbox stages
//!
//! Every module launches its kernels over planes the caller already holds on the device:
//!
//! - [blur] for separable gaussian blurs.
//! - [reduce] for the extremes and sums of a plane.
//! - [histogram] for percentiles.
//! - [sobel] for image gradients.
//! - [labels] for connected components.

pub mod blur;
mod border;
mod dispatch;
pub mod histogram;
pub mod labels;
mod plane;
pub mod reduce;
pub mod sobel;

pub use self::border::reflect;
pub use self::dispatch::Dispatch;
pub use self::plane::DevicePlane;
