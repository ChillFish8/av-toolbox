//! Sources that bring video into the pipeline
//!
//! - `y4m/stdin` and `y4m/pipe` read a y4m stream.
//! - `ffms2/file` decodes a media file.
//! - `solid/fill` and `image/file` repeat one frame of a colour or an image.

mod constant;
#[cfg(feature = "ffms2")]
mod ffms2;
mod pipe;
mod shared;
mod stdin;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every source this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_source::<stdin::Stdin>()?;
    registry.register_source::<pipe::Pipe>()?;
    registry.register_source::<constant::Fill>()?;
    registry.register_source::<constant::File>()?;

    #[cfg(feature = "ffms2")]
    registry.register_source::<ffms2::File>()?;

    Ok(())
}
