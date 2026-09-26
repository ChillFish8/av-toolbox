//! Sources that bring video into the pipeline
//!
//! - `y4m/stdin` and `y4m/pipe` read a y4m stream.
//! - `ffms2/file` decodes a media file.
//! - `solid/fill` and `image/file` repeat one frame of a colour or an image.
//! - `solid/like` and `image/like` are filters that stand in for a source, repeating one frame in the size, format
//!   and length of a reference clip.

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
    registry.register_filter::<constant::SolidLike>()?;
    registry.register_filter::<constant::ImageLike>()?;

    #[cfg(feature = "ffms2")]
    registry.register_source::<ffms2::File>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    fn registered(kind: StageKind, identifier: &str) -> bool {
        let mut registry = PipelineRegistry::new();
        register(&mut registry).expect("the sources register");

        registry
            .stages()
            .any(|info| info.kind() == kind && info.full_identifier() == identifier)
    }

    #[test]
    fn every_constant_stage_is_registered() {
        assert!(registered(StageKind::Source, "solid/fill"));
        assert!(registered(StageKind::Source, "image/file"));
        assert!(registered(StageKind::Filter, "solid/like"));
        assert!(registered(StageKind::Filter, "image/like"));
    }
}
