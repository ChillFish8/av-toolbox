//! The pixel model a pipeline moves through.
//!
//! A [FrameFormat] describes what a frame is: its dimensions, which channels it carries, how they
//! are subsampled, and what each sample is stored as. A [Frame] is that description plus one
//! [Plane] per channel. A [Mask] is the same machinery with a single plane.
//!
//! Planes are flat, with one sample per pixel per row and no padding between rows.

mod format;
pub(crate) mod gpu;
mod host;
mod layout;
mod sample;

pub use self::format::{
    ColourInfo,
    ColourRange,
    FrameFormat,
    MaskFormat,
    Matrix,
    Primaries,
    StreamFormat,
    Transfer,
};
pub use self::gpu::{Frame, FrameMeta, FrameMut, Mask, Plane, PlaneBuffer};
pub use self::host::HostPlanes;
pub use self::layout::{Channel, FrameLayout, Subsampling};
pub use self::sample::{Depth, DepthError, SampleFormat};
