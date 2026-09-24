//! Transforms that change how a stream is divided into chunks.

#[cfg(feature = "scene")]
mod scene;

#[cfg_attr(
    not(test),
    expect(dead_code, reason = "nothing outside the tests splits scenes yet")
)]
mod split;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every transform this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    #[cfg(feature = "scene")]
    registry.register_transform::<scene::SceneDetect>()?;

    Ok(())
}
