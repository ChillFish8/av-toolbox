//! Masks that select pixels from a stream
//!
//! Every mask is a single U8 plane, where 0 is background and 255 is fully selected.
//!
//! - `mask/threshold` hardens a mask to 0 or 255 at a cutoff.

mod mask;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every mask this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_mask::<mask::Threshold>()?;

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
        assert!(registered("mask/threshold"));
    }
}
