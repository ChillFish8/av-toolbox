//! Sinks that write the finished stream out of the pipeline from host frames.

mod shared;
mod stdout;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every sink this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_sink::<stdout::Stdout>()?;

    Ok(())
}
