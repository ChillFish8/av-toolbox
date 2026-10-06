//! Providers that supply what stages demand
//!
//! - `scene-score`, which scores every frame as a scene cut for every stage demanding
//!   [SceneScores].
//! - `scene-split`, which breaks the stream into scenes from those scores for every stage
//!   demanding [Scenes].

mod scene;
mod split;
#[cfg(test)]
mod tests;

use av_toolbox_core::{PipelineRegistry, RegistryError};

pub use self::scene::{
    SceneBreaks,
    SceneChunk,
    SceneCut,
    SceneRequest,
    SceneScore,
    SceneScores,
    SceneSpeed,
    Scenes,
    ScoreRequest,
};

/// Registers every provider this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_provider::<scene::SceneScoreProvider>()?;
    registry.register_provider::<split::SceneSplitProvider>()?;

    Ok(())
}
