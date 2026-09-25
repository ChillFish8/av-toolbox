//! Sinks that write the finished stream out of the pipeline.

mod metadata;
mod pipe;
mod shared;
mod stdout;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every sink this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_sink::<stdout::Stdout>()?;
    registry.register_sink::<pipe::Pipe>()?;
    registry.register_meta_sink::<metadata::Ndjson>()?;

    Ok(())
}
