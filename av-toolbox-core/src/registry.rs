use std::collections::HashMap;

use av_toolbox_gpu::{Backend, ResolvedFormats};
use serde_json::Value;

use crate::factory::Definition;
use crate::{
    BuildStageError,
    Filter,
    FilterDefinition,
    InputArgument,
    Mask,
    MaskDefinition,
    RegistryError,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageFactory,
    StageInfo,
    StageKind,
    Transform,
    TransformDefinition,
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
    pub fn register_source<D: SourceDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Source)?;
        if !D::inputs().is_empty() {
            return Err(RegistryError::SourceTakesInputs {
                identifier: definition.full_identifier().to_string(),
            });
        }

        insert(&mut self.sources, definition)
    }

    pub fn register_filter<D: FilterDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Filter)?;

        insert(&mut self.filters, definition)
    }

    pub fn register_mask<D: MaskDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Mask)?;

        insert(&mut self.masks, definition)
    }

    pub fn register_transform<D: TransformDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Transform)?;

        insert(&mut self.transforms, definition)
    }

    pub fn register_sink<D: SinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;

        insert(&mut self.sinks, definition)
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_source(
        &self,
        identifier: &str,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, BuildStageError> {
        build(
            &self.sources,
            StageKind::Source,
            identifier,
            backend,
            args,
            inputs,
            formats,
        )
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_filter(
        &self,
        identifier: &str,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, BuildStageError> {
        build(
            &self.filters,
            StageKind::Filter,
            identifier,
            backend,
            args,
            inputs,
            formats,
        )
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_mask(
        &self,
        identifier: &str,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, BuildStageError> {
        build(
            &self.masks,
            StageKind::Mask,
            identifier,
            backend,
            args,
            inputs,
            formats,
        )
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_transform(
        &self,
        identifier: &str,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Transform>, BuildStageError> {
        build(
            &self.transforms,
            StageKind::Transform,
            identifier,
            backend,
            args,
            inputs,
            formats,
        )
    }

    /// Validates the arguments, deserializes them, binds the inputs, then builds the stage.
    pub fn build_sink(
        &self,
        identifier: &str,
        backend: &Backend,
        args: &Value,
        inputs: &[InputArgument],
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, BuildStageError> {
        build(
            &self.sinks,
            StageKind::Sink,
            identifier,
            backend,
            args,
            inputs,
            formats,
        )
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
    backend: &Backend,
    args: &Value,
    inputs: &[InputArgument],
    formats: &ResolvedFormats,
) -> Result<Box<K>, BuildStageError> {
    let factory = factories
        .get(identifier)
        .ok_or_else(|| BuildStageError::UnknownStage {
            kind,
            identifier: identifier.to_string(),
        })?;

    factory.build(backend, args, inputs, formats)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{
        BuildContext,
        Capabilities,
        FormatSet,
        OutputFormats,
        Runtime,
        StreamFormat,
        StreamTemplate,
    };
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{BoundInputs, InputParam, InputSignature, StageError, StageInstance};
    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    use crate::{StageHandle, StreamType};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct DenoiseArgs {
        sigma_scale: f32,
    }

    struct DenoiseFilter;

    static RECORDED_SIGMA_SCALE: AtomicU32 = AtomicU32::new(0);

    struct Denoise;

    impl StageDefinition for Denoise {
        type Args = DenoiseArgs;

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
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    #[expect(dead_code, reason = "the fixture only deserializes its arguments")]
    struct FileArgs {
        input: String,
    }

    struct FileSource;

    struct FileSourceDefinition;

    impl StageDefinition for FileSourceDefinition {
        type Args = FileArgs;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "file"
        }

        fn inputs() -> &'static InputSignature {
            &InputSignature::EMPTY
        }
    }

    impl SourceDefinition for FileSourceDefinition {
        fn produces(_capabilities: &Capabilities, _args: &Self::Args) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Source>, StageError> {
            Ok(Box::new(FileSource))
        }
    }

    /// A source that wrongly declares inputs, used to check the registry rejects it.
    struct GreedySource;

    impl StageDefinition for GreedySource {
        type Args = FileArgs;

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
    }

    impl SourceDefinition for GreedySource {
        fn produces(_capabilities: &Capabilities, _args: &Self::Args) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Source>, StageError> {
            Ok(Box::new(FileSource))
        }
    }

    impl FilterDefinition for Denoise {
        fn accepts(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            offered: &[StreamFormat],
        ) -> Result<Vec<FormatSet>, StageError> {
            Ok(offered
                .iter()
                .map(|format| FormatSet::of([format.sample()]))
                .collect())
        }

        fn produces(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            chosen: &[StreamFormat],
        ) -> Result<OutputFormats, StageError> {
            let format = chosen[0];
            Ok(OutputFormats {
                template: StreamTemplate::from_format(&format),
                samples: FormatSet::of([format.sample()]),
            })
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Filter>, StageError> {
            RECORDED_SIGMA_SCALE.store(args.sigma_scale.to_bits(), Ordering::SeqCst);

            Ok(Box::new(DenoiseFilter))
        }
    }

    impl Source for FileSource {}
    impl StageInstance for FileSource {}
    impl Filter for DenoiseFilter {}
    impl StageInstance for DenoiseFilter {}

    fn registry_with_denoise() -> PipelineRegistry {
        let mut registry = PipelineRegistry::new();
        registry.register_filter::<Denoise>().expect("should register");
        registry
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    fn reference_input() -> Vec<InputArgument> {
        vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)]
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

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    mod device_tests {
        use av_toolbox_gpu::Device;

        use super::*;

        fn backend() -> Backend {
            Backend::open_default(Device::Default).expect("should open")
        }

        fn formats() -> ResolvedFormats {
            ResolvedFormats::new(vec![], None)
        }

        #[test]
        fn a_registered_filter_builds_from_its_identifier() {
            let registry = registry_with_denoise();
            let args = serde_json::json!({ "sigma_scale": 0.8 });
            let backend = backend();
            let formats = formats();

            let filter = registry
                .build_filter("av-denoise/nl4d", &backend, &args, &reference_input(), &formats)
                .expect("should build");
            drop(filter);

            let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
            assert_eq!(recorded_sigma_scale, 0.8);
        }

        #[test]
        fn an_unregistered_identifier_reports_the_kind_looked_in() {
            let registry = registry_with_denoise();
            let args = serde_json::json!({});
            let backend = backend();
            let formats = formats();

            let result = registry.build_sink("av-denoise/nl4d", &backend, &args, &[], &formats);
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
    }
}
