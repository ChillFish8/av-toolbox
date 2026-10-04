//! Shared CubeCL kernels for av-toolbox stages
//!
//! Every module launches its kernels over planes the caller already holds on the device:
//!
//! - [aa] for the steps of the anti-aliasing filters.
//! - [artcnn] for running ArtCNN luma doublers.
//! - [bilateral] for vszip's truncated bilateral.
//! - [blur] for separable gaussian blurs.
//! - [deband] for smoothing banding the way libplacebo does.
//! - [degrain] for averaging a frame with its motion-compensated neighbours, the way MVTools' MDegrain does.
//! - [dehalo] for the fused per-pixel steps of the dehalo filters.
//! - [depth] for changing a plane's bit depth, with dithering.
//! - [deinterlace] for rebuilding one field of a frame from its neighbours, the way bwdif does.
//! - [edges] for Prewitt, Sobel and Kirsch edge strength, and binarizing.
//! - [reduce] for the extremes and sums of a plane.
//! - [histogram] for percentiles.
//! - [hysteresis] for keeping the connected regions of a mask that hold a seed.
//! - [ivtc] for the weaves and metrics of inverse telecine, the way VIVTC's VFM and VDecimate compute them.
//! - [lineart] for the per-pixel steps of the lineart filters.
//! - [motion] for searching how each block of a frame moved between frames.
//! - [sobel] for image gradients.
//! - [values] for moving integer planes into f32 code values and back.
//! - [labels] for connected components.
//! - [neighbourhood] for 3x3 maximum, minimum, blurs, medians, Prewitt, MinBlur, inflate and deflate.
//! - [padding] for growing a plane by repeating its edges, and cropping it back.
//! - [resize] for separable resizing through per axis weight tables.
//! - [retinex] for the recursive Gaussian and the multi-scale retinex.
//! - [rgvs] for RGVS's Repair and RemoveGrain modes.
//! - [warp] for aWarpSharp2's edge mask, mask blurs and warp.

pub mod aa;
pub mod artcnn;
pub mod bilateral;
pub mod blur;
mod border;
pub mod deband;
pub mod degrain;
pub mod dehalo;
pub mod deinterlace;
pub mod depth;
mod dispatch;
pub mod edges;
pub mod histogram;
pub mod hysteresis;
pub mod ivtc;
pub mod labels;
pub mod lineart;
pub mod motion;
pub mod neighbourhood;
pub mod padding;
mod plane;
pub mod reduce;
pub mod resize;
pub mod retinex;
pub mod rgvs;
pub mod sobel;
#[cfg(any(test, feature = "test-support"))]
pub mod testing;
pub mod values;
pub mod warp;

pub use self::border::reflect;
pub use self::dispatch::Dispatch;
pub use self::plane::{DevicePlane, assert_plane};
