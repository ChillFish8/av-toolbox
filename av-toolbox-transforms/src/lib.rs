//! Transforms that change how a stream is divided into chunks, and the scene provider.

mod provider;
mod scene;
mod split;
#[cfg(test)]
mod tests;

use av_toolbox_core::{PipelineRegistry, RegistryError};

pub use self::provider::{SceneBreaks, SceneRequest, SceneSpeed, Scenes};

/// Registers every transform and provider this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_transform::<scene::SceneDetect>()?;
    registry.register_meta_transform::<split::SceneSplit>()?;
    registry.register_provider::<provider::SceneProvider>()?;

    Ok(())
}
