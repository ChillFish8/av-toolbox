use av_toolbox_core::{PipelineRegistry, RegistryError};

/// A registry holding every stage the toolbox ships.
///
/// A caller may register its own stages on the result before building a pipeline.
pub fn registry() -> Result<PipelineRegistry, RegistryError> {
    let mut registry = PipelineRegistry::new();

    av_toolbox_sources::register(&mut registry)?;
    av_toolbox_transforms::register(&mut registry)?;
    av_toolbox_sinks::register(&mut registry)?;

    Ok(registry)
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    fn registered(kind: StageKind, identifier: &str) -> bool {
        let registry = registry().expect("the built-in stages register");
        registry
            .stages()
            .any(|info| info.kind() == kind && info.full_identifier() == identifier)
    }

    #[test]
    fn every_built_in_stage_is_registered() {
        assert!(registered(StageKind::Source, "y4m/stdin"));
        assert!(registered(StageKind::Source, "y4m/pipe"));
        assert!(registered(StageKind::Sink, "y4m/stdout"));
        assert!(registered(StageKind::Sink, "y4m/pipe"));
        assert!(registered(StageKind::Transform, "scene/detect"));
        assert!(registered(StageKind::Transform, "scene/split"));
    }

    #[cfg(feature = "ffms2")]
    #[test]
    fn the_ffms2_source_is_registered() {
        assert!(registered(StageKind::Source, "ffms2/file"));
    }
}
