//! Shared CubeCL kernels for av-toolbox stages
//!
//! Every module launches its kernels over planes the caller already holds on the device:
//!
//! - [blur] for separable gaussian blurs.
//! - [deband] for smoothing banding the way libplacebo does.
//! - [depth] for changing a plane's bit depth, with dithering.
//! - [reduce] for the extremes and sums of a plane.
//! - [histogram] for percentiles.
//! - [sobel] for image gradients.
//! - [values] for moving integer planes into f32 code values and back.
//! - [labels] for connected components.
//! - [resize] for separable resizing through per axis weight tables.
//! - [warp] for aWarpSharp2's edge mask, mask blurs and warp.

pub mod blur;
mod border;
pub mod deband;
pub mod depth;
mod dispatch;
pub mod histogram;
pub mod labels;
mod plane;
pub mod reduce;
pub mod resize;
pub mod sobel;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
pub mod values;
pub mod warp;

pub use self::border::reflect;
pub use self::dispatch::Dispatch;
pub use self::plane::DevicePlane;
