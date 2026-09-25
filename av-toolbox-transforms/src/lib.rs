//! Transforms that change how a stream is divided into chunks.

mod scene;
mod split;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every transform this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_transform::<scene::SceneDetect>()?;
    registry.register_meta_transform::<split::SceneSplit>()?;

    Ok(())
}
