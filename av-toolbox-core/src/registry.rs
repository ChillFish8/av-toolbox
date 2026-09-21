use std::collections::HashMap;

use serde_json::Value;

use crate::factory::Definition;
use crate::{
    BuildStageError,
    FilterDefinition,
    InputArgument,
    MaskDefinition,
    PreparedFilter,
    PreparedMask,
    PreparedSink,
    PreparedSource,
    PreparedStage,
    PreparedTransform,
    RegistryError,
    SinkDefinition,
    SourceDefinition,
    StageDefinition,
    StageFactory,
    StageInfo,
    StageKind,
    TransformDefinition,
};

type Factories<P> = HashMap<String, Box<dyn StageFactory<P>>>;

/// The registry tracks all stages available for use within the pipeline.
///
/// Stages are keyed by their fully qualified `package/stage` identifier.
#[derive(Default)]
pub struct PipelineRegistry {
    sources: Factories<dyn PreparedSource>,
    filters: Factories<dyn PreparedFilter>,
    masks: Factories<dyn PreparedMask>,
    transforms: Factories<dyn PreparedTransform>,
    sinks: Factories<dyn PreparedSink>,
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

    /// Deserializes the arguments and binds the inputs, without needing a device.
    pub fn prepare(
        &self,
        kind: StageKind,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<PreparedStage, BuildStageError> {
        match kind {
            StageKind::Source => {
                prepare(&self.sources, kind, identifier, args, inputs).map(PreparedStage::Source)
            },
            StageKind::Filter => {
                prepare(&self.filters, kind, identifier, args, inputs).map(PreparedStage::Filter)
            },
            StageKind::Mask => prepare(&self.masks, kind, identifier, args, inputs).map(PreparedStage::Mask),
            StageKind::Transform => {
                prepare(&self.transforms, kind, identifier, args, inputs).map(PreparedStage::Transform)
            },
            StageKind::Sink => prepare(&self.sinks, kind, identifier, args, inputs).map(PreparedStage::Sink),
        }
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

fn insert<D, P>(factories: &mut Factories<P>, definition: Definition<D>) -> Result<(), RegistryError>
where
    D: StageDefinition,
    P: ?Sized,
    Definition<D>: StageFactory<P> + 'static,
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

fn prepare<P: ?Sized>(
    factories: &Factories<P>,
    kind: StageKind,
    identifier: &str,
    args: &Value,
    inputs: &[InputArgument],
) -> Result<Box<P>, BuildStageError> {
    let factory = factories
        .get(identifier)
        .ok_or_else(|| BuildStageError::UnknownStage {
            kind,
            identifier: identifier.to_string(),
        })?;

    factory.prepare(args, inputs)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{
        BuildContext,
        Capabilities,
        FormatSet,
        Offered,
        OutputFormats,
        ResolvedFormats,
        Runtime,
        StreamFormat,
        StreamTemplate,
    };
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{BoundInputs, Filter, InputParam, InputSignature, Source, StageError, StageInstance};
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
        fn accepts_input(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            _index: usize,
            offered: StreamFormat,
            _inputs: &[Offered],
        ) -> Result<FormatSet, StageError> {
            Ok(FormatSet::of([offered.sample()]))
        }

        fn produces(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            chosen: &[Option<StreamFormat>],
        ) -> Result<OutputFormats, StageError> {
            let format = chosen.iter().flatten().next().copied().expect("an input");
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

    #[test]
    fn an_unregistered_identifier_reports_the_kind_looked_in() {
        let registry = registry_with_denoise();
        let args = serde_json::json!({});

        let result = registry.prepare(StageKind::Sink, "av-denoise/nl4d", &args, &[]);
        let Err(error) = result else {
            panic!("the stage should not have been prepared");
        };

        assert!(matches!(
            error,
            BuildStageError::UnknownStage {
                kind: StageKind::Sink,
                ..
            },
        ));
    }

    #[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
    mod device_tests {
        use av_toolbox_gpu::{Backend, Device};

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

            let prepared = registry
                .prepare(StageKind::Filter, "av-denoise/nl4d", &args, &reference_input())
                .expect("should prepare");
            let PreparedStage::Filter(prepared) = prepared else {
                panic!("expected a filter");
            };

            let filter = prepared.build(&backend, &formats).expect("should build");
            drop(filter);

            let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
            assert_eq!(recorded_sigma_scale, 0.8);
        }
    }
}
