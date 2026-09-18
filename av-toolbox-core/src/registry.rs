use std::collections::HashMap;

use serde_json::Value;

use crate::factory::Definition;
use crate::{
    BuildStageError,
    Filter,
    InputArgument,
    Mask,
    RegistryError,
    Sink,
    Source,
    StageDefinition,
    StageFactory,
    StageInfo,
    StageKind,
    Transform,
};

type Factories<K> = HashMap<String, Box<dyn StageFactory<K>>>;

/// The registry tracks all stages available for use within the pipeline.
///
/// Stages are keyed by their fully qualified `package/stage` identifier.
#[derive(Default)]
pub struct PipelineRegistry {
    sources: Factories<dyn Source>,
    filters: Factories<dyn Filter>,
    masks: Factories<dyn Mask>,
    transforms: Factories<dyn Transform>,
    sinks: Factories<dyn Sink>,
}

impl PipelineRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a source, rejecting a definition whose `inputs()` is non-empty.
    pub fn register_source<D>(&mut self) -> Result<(), RegistryError>
    where
        D: StageDefinition,
        D::Instance: Source,
    {
        let definition = Definition::<D>::new(StageKind::Source)?;
        if !D::inputs().is_empty() {
            return Err(RegistryError::SourceTakesInputs {
                identifier: definition.full_identifier().to_string(),
            });
        }

        insert(&mut self.sources, definition)
    }

    pub fn register_filter<D>(&mut self) -> Result<(), RegistryError>
    where
        D: StageDefinition,
        D::Instance: Filter,
    {
        let definition = Definition::<D>::new(StageKind::Filter)?;
        insert(&mut self.filters, definition)
    }

    pub fn register_mask<D>(&mut self) -> Result<(), RegistryError>
    where
        D: StageDefinition,
        D::Instance: Mask,
    {
        let definition = Definition::<D>::new(StageKind::Mask)?;
        insert(&mut self.masks, definition)
    }

    pub fn register_transform<D>(&mut self) -> Result<(), RegistryError>
    where
        D: StageDefinition,
        D::Instance: Transform,
    {
        let definition = Definition::<D>::new(StageKind::Transform)?;
        insert(&mut self.transforms, definition)
    }

    pub fn register_sink<D>(&mut self) -> Result<(), RegistryError>
    where
        D: StageDefinition,
        D::Instance: Sink,
    {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        insert(&mut self.sinks, definition)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_source(
        &self,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<Box<dyn Source>, BuildStageError> {
        build(&self.sources, StageKind::Source, identifier, args, inputs)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_filter(
        &self,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<Box<dyn Filter>, BuildStageError> {
        build(&self.filters, StageKind::Filter, identifier, args, inputs)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_mask(
        &self,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<Box<dyn Mask>, BuildStageError> {
        build(&self.masks, StageKind::Mask, identifier, args, inputs)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_transform(
        &self,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<Box<dyn Transform>, BuildStageError> {
        build(&self.transforms, StageKind::Transform, identifier, args, inputs)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_sink(
        &self,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<Box<dyn Sink>, BuildStageError> {
        build(&self.sinks, StageKind::Sink, identifier, args, inputs)
    }

    /// Describes every registered stage.
    pub fn stages(&self) -> impl Iterator<Item = StageInfo<'_>> {
        let sources = self.sources.values().map(|factory| factory.info());
        let filters = self.filters.values().map(|factory| factory.info());
        let masks = self.masks.values().map(|factory| factory.info());
        let transforms = self.transforms.values().map(|factory| factory.info());
        let sinks = self.sinks.values().map(|factory| factory.info());

        sources.chain(filters).chain(masks).chain(transforms).chain(sinks)
    }
}

fn insert<D, K>(factories: &mut Factories<K>, definition: Definition<D>) -> Result<(), RegistryError>
where
    D: StageDefinition,
    K: ?Sized,
    Definition<D>: StageFactory<K> + 'static,
{
    let identifier = definition.full_identifier().to_string();
    if factories.contains_key(&identifier) {
        return Err(RegistryError::DuplicateStage {
            kind: definition.kind(),
            identifier,
        });
    }

    factories.insert(identifier, Box::new(definition));

    Ok(())
}

fn build<K: ?Sized>(
    factories: &Factories<K>,
    kind: StageKind,
    identifier: &str,
    args: &Value,
    inputs: &[InputArgument],
) -> Result<Box<K>, BuildStageError> {
    let factory = factories
        .get(identifier)
        .ok_or_else(|| BuildStageError::UnknownStage {
            kind,
            identifier: identifier.to_string(),
        })?;

    factory.build(args, inputs)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{
        BoundInputs,
        InputParam,
        InputSignature,
        StageError,
        StageHandle,
        StageInstance,
        StreamType,
    };

    #[derive(Debug, Deserialize, JsonSchema)]
    struct DenoiseArgs {
        sigma_scale: f32,
    }

    struct DenoiseFilter {
        sigma_scale: f32,
    }

    impl StageInstance for DenoiseFilter {}
    impl Filter for DenoiseFilter {}

    static RECORDED_SIGMA_SCALE: AtomicU32 = AtomicU32::new(0);

    struct Denoise;

    impl StageDefinition for Denoise {
        type Args = DenoiseArgs;
        type Instance = DenoiseFilter;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "nl4d"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }

        fn build(args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
            let instance = DenoiseFilter {
                sigma_scale: args.sigma_scale,
            };
            RECORDED_SIGMA_SCALE.store(instance.sigma_scale.to_bits(), Ordering::SeqCst);

            Ok(instance)
        }
    }

    #[expect(dead_code, reason = "the fixture only deserializes its arguments")]
    #[derive(Debug, Deserialize, JsonSchema)]
    struct FileArgs {
        input: String,
    }

    struct FileSource;

    impl StageInstance for FileSource {}
    impl Source for FileSource {}

    struct FileSourceDefinition;

    impl StageDefinition for FileSourceDefinition {
        type Args = FileArgs;
        type Instance = FileSource;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "file"
        }

        fn inputs() -> &'static InputSignature {
            &InputSignature::EMPTY
        }

        fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
            Ok(FileSource)
        }
    }

    /// A source that wrongly declares inputs, used to check the registry rejects it.
    struct GreedySource;

    impl StageDefinition for GreedySource {
        type Args = FileArgs;
        type Instance = FileSource;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "greedy"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }

        fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
            Ok(FileSource)
        }
    }

    fn registry_with_denoise() -> PipelineRegistry {
        let mut registry = PipelineRegistry::new();
        registry.register_filter::<Denoise>().expect("should register");
        registry
    }

    fn reference_input() -> Vec<InputArgument> {
        vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
    }

    #[test]
    fn a_registered_filter_builds_from_its_identifier() {
        let registry = registry_with_denoise();
        let args = serde_json::json!({ "sigma_scale": 0.8 });

        let filter = registry
            .build_filter("av-denoise/nl4d", &args, &reference_input())
            .expect("should build");

        let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
        assert_eq!(recorded_sigma_scale, 0.8);

        drop(filter);
    }

    #[test]
    fn registering_the_same_filter_twice_is_rejected() {
        let mut registry = registry_with_denoise();

        let error = registry
            .register_filter::<Denoise>()
            .expect_err("should not register");

        assert!(matches!(error, RegistryError::DuplicateStage { .. }));
    }

    #[test]
    fn a_source_declaring_inputs_is_rejected() {
        let mut registry = PipelineRegistry::new();

        let error = registry
            .register_source::<GreedySource>()
            .expect_err("should not register");

        assert!(matches!(error, RegistryError::SourceTakesInputs { .. }));
    }

    #[test]
    fn an_unregistered_identifier_reports_the_kind_looked_in() {
        let registry = registry_with_denoise();
        let args = serde_json::json!({});

        let result = registry.build_sink("av-denoise/nl4d", &args, &[]);
        let Err(error) = result else {
            panic!("the stage should not have been built");
        };

        assert!(matches!(
            error,
            BuildStageError::UnknownStage {
                kind: StageKind::Sink,
                ..
            },
        ));
    }

    #[test]
    fn stages_lists_every_registered_stage_across_kinds() {
        let mut registry = registry_with_denoise();
        registry
            .register_source::<FileSourceDefinition>()
            .expect("should register");

        let mut identifiers: Vec<String> = registry
            .stages()
            .map(|info| info.full_identifier().to_string())
            .collect();
        identifiers.sort();

        assert_eq!(identifiers, vec!["av-denoise/nl4d", "builtin/file"]);
    }
}
