//! Filters that change the frames of a stream
//!
//! - `blend/masked` picks each pixel from one of two clips by a mask.

mod blend;
mod resize;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every filter this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_filter::<blend::Masked>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    fn registered(identifier: &str) -> bool {
        let mut registry = PipelineRegistry::new();
        register(&mut registry).expect("the filters register");

        registry
            .stages()
            .any(|info| info.kind() == StageKind::Filter && info.full_identifier() == identifier)
    }

    #[test]
    fn every_filter_is_registered() {
        assert!(registered("blend/masked"));
    }
}
