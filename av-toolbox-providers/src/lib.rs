//! Providers that supply what stages demand
//!
//! - The scene provider, which finds scene cuts and breaks the stream into scenes for every
//!   stage demanding [Scenes].

mod provider;
mod scene;
mod split;
#[cfg(test)]
mod tests;

use av_toolbox_core::{PipelineRegistry, RegistryError};

pub use self::provider::{SceneBreaks, SceneChunk, SceneRequest, SceneSpeed, Scenes};

/// Registers every provider this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_provider::<provider::SceneProvider>()?;

    Ok(())
}
