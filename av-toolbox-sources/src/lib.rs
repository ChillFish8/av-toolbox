//! Sources that read video from outside the pipeline as host frames.

mod pipe;
mod shared;
mod stdin;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every source this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_source::<stdin::Stdin>()?;
    registry.register_source::<pipe::Pipe>()?;

    Ok(())
}
