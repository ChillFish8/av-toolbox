//! Masks that select pixels from a stream
//!
//! Every mask is a single U8 plane, where 0 is background and 255 is fully selected.
//!
//! - `lines/adaptive` finds dark strokes against their neighbourhood, and can
//!   even out lighting first to find them in deep shadow.
//! - `lines/canny` finds edges with Canny's method.
//! - `lines/xdog` finds line art with an extended difference of gaussians.
//! - `mask/threshold` hardens a mask to 0 or 255 at a cutoff.
//! - `mask/union` keeps the largest sample of every input.
//! - `mask/vote` selects the pixels enough of its inputs agree on.

mod lines;
mod luma;
mod mask;
mod sizing;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every mask this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_mask::<lines::Adaptive>()?;
    registry.register_mask::<lines::Canny>()?;
    registry.register_mask::<lines::Xdog>()?;
    registry.register_mask::<mask::Threshold>()?;
    registry.register_mask::<mask::Union>()?;
    registry.register_mask::<mask::Vote>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    fn registered(identifier: &str) -> bool {
        let mut registry = PipelineRegistry::new();
        register(&mut registry).expect("the masks register");

        registry
            .stages()
            .any(|info| info.kind() == StageKind::Mask && info.full_identifier() == identifier)
    }

    #[test]
    fn every_mask_is_registered() {
        assert!(registered("lines/adaptive"));
        assert!(registered("lines/canny"));
        assert!(registered("lines/xdog"));
        assert!(registered("mask/threshold"));
        assert!(registered("mask/union"));
        assert!(registered("mask/vote"));
    }
}
