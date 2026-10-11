//! Frame accurate video decoding through a statically linked BestSource
//!
//! [VideoSource] opens a video track and decodes frames by number, indexing the file on open.

mod error;
mod format;
mod frame;
mod sys;
#[cfg(test)]
mod tests;
mod video;

pub use self::error::BestSourceError;
pub use self::format::{ColorFamily, VideoFormat};
pub use self::frame::{Frame, PlanarTarget, PlaneTarget};
pub use self::video::{IndexCache, OpenOptions, VideoProperties, VideoSource};
