#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]

//! Sinks that write the finished stream and its artefacts out of the pipeline.

mod artefacts;
mod metadata;
mod null;
mod pipe;
mod shared;
mod stdout;

use av_toolbox_core::{PipelineRegistry, RegistryError};

/// Registers every sink this crate provides.
pub fn register(registry: &mut PipelineRegistry) -> Result<(), RegistryError> {
    registry.register_sink::<stdout::Stdout>()?;
    registry.register_sink::<pipe::Pipe>()?;
    registry.register_sink::<null::Discard>()?;
    registry.register_meta_sink::<metadata::Ndjson>()?;
    registry.register_artefact_sink::<artefacts::Files>()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y4m_sinks_want_the_output_depth() {
        let mut registry = PipelineRegistry::new();
        register(&mut registry).expect("should register");

        let wants = |identifier: &str| {
            registry
                .stages()
                .find(|info| info.full_identifier() == identifier)
                .expect("should be registered")
                .wants_output_depth()
        };

        assert!(wants("y4m/pipe"));
        assert!(wants("y4m/stdout"));
        assert!(!wants("null/discard"));
    }
}
