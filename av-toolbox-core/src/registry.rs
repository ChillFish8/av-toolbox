use std::collections::HashMap;

use serde_json::Value;

use crate::factory::Definition;
use crate::{
    BuildStageError,
    Crossing,
    FilterDefinition,
    InputArgument,
    MaskDefinition,
    MetaSinkDefinition,
    MetaTransformDefinition,
    OpenSlot,
    PreparedFilter,
    PreparedMask,
    PreparedMetaSink,
    PreparedMetaTransform,
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
/// Stages are keyed by their fully qualified `package/stage` identifier. `builtin/upload` and
/// `builtin/unload` name the crossings, so no stage may take either.
#[derive(Default)]
pub struct PipelineRegistry {
    sources: Factories<dyn PreparedSource>,
    filters: Factories<dyn PreparedFilter>,
    masks: Factories<dyn PreparedMask>,
    transforms: Factories<dyn PreparedTransform>,
    meta_transforms: Factories<dyn PreparedMetaTransform>,
    sinks: Factories<dyn PreparedSink>,
    meta_sinks: Factories<dyn PreparedMetaSink>,
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
        let identifier = definition.full_identifier();
        if self.meta_transforms.contains_key(identifier) {
            return Err(RegistryError::DuplicateStage {
                kind: StageKind::Transform,
                identifier: identifier.to_string(),
            });
        }

        insert(&mut self.transforms, definition)
    }

    /// Registers a metadata-only transform, which shares the transform identifiers.
    pub fn register_meta_transform<D: MetaTransformDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Transform)?;
        let identifier = definition.full_identifier();
        if self.transforms.contains_key(identifier) {
            return Err(RegistryError::DuplicateStage {
                kind: StageKind::Transform,
                identifier: identifier.to_string(),
            });
        }

        insert(&mut self.meta_transforms, definition)
    }

    pub fn register_sink<D: SinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        let identifier = definition.full_identifier();
        if self.meta_sinks.contains_key(identifier) {
            return Err(RegistryError::DuplicateStage {
                kind: StageKind::Sink,
                identifier: identifier.to_string(),
            });
        }

        insert(&mut self.sinks, definition)
    }

    /// Registers a metadata-only sink, which shares the sink identifiers.
    pub fn register_meta_sink<D: MetaSinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        let identifier = definition.full_identifier();
        if self.sinks.contains_key(identifier) {
            return Err(RegistryError::DuplicateStage {
                kind: StageKind::Sink,
                identifier: identifier.to_string(),
            });
        }

        insert(&mut self.meta_sinks, definition)
    }

    /// Deserializes the arguments and binds the inputs, without needing a device.
    pub fn prepare(
        &self,
        kind: StageKind,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
    ) -> Result<PreparedStage, BuildStageError> {
        let slot = OpenSlot::new();

        self.prepare_with(kind, identifier, args, inputs, &slot)
    }

    /// The same, opening a source into `slot` so every pass sharing it opens the input once.
    pub fn prepare_with(
        &self,
        kind: StageKind,
        identifier: &str,
        args: &Value,
        inputs: &[InputArgument],
        slot: &OpenSlot,
    ) -> Result<PreparedStage, BuildStageError> {
        match kind {
            StageKind::Source => {
                prepare(&self.sources, kind, identifier, args, inputs, slot).map(PreparedStage::Source)
            },
            StageKind::Filter => {
                prepare(&self.filters, kind, identifier, args, inputs, slot).map(PreparedStage::Filter)
            },
            StageKind::Mask => {
                prepare(&self.masks, kind, identifier, args, inputs, slot).map(PreparedStage::Mask)
            },
            StageKind::Transform if self.meta_transforms.contains_key(identifier) => {
                prepare(&self.meta_transforms, kind, identifier, args, inputs, slot)
                    .map(PreparedStage::MetaTransform)
            },
            StageKind::Transform => {
                prepare(&self.transforms, kind, identifier, args, inputs, slot).map(PreparedStage::Transform)
            },
            StageKind::Sink if self.meta_sinks.contains_key(identifier) => {
                prepare(&self.meta_sinks, kind, identifier, args, inputs, slot).map(PreparedStage::MetaSink)
            },
            StageKind::Sink => {
                prepare(&self.sinks, kind, identifier, args, inputs, slot).map(PreparedStage::Sink)
            },
        }
    }

    /// Whether opening a registered source's input, given `args`, would wait on a live stream.
    pub fn reads_stream(&self, identifier: &str, args: &Value) -> Result<bool, BuildStageError> {
        let factory = lookup(&self.sources, StageKind::Source, identifier)?;

        factory.reads_stream(args)
    }

    /// Describes every registered stage.
    pub fn stages(&self) -> impl Iterator<Item = StageInfo<'_>> {
        let sources = self.sources.values().map(|factory| factory.info());
        let filters = self.filters.values().map(|factory| factory.info());
        let masks = self.masks.values().map(|factory| factory.info());
        let transforms = self.transforms.values().map(|factory| factory.info());
        let meta_transforms = self.meta_transforms.values().map(|factory| factory.info());
        let sinks = self.sinks.values().map(|factory| factory.info());
        let meta_sinks = self.meta_sinks.values().map(|factory| factory.info());

        sources
            .chain(filters)
            .chain(masks)
            .chain(transforms)
            .chain(meta_transforms)
            .chain(sinks)
            .chain(meta_sinks)
    }
}

fn insert<D, P>(factories: &mut Factories<P>, definition: Definition<D>) -> Result<(), RegistryError>
where
    D: StageDefinition,
    P: ?Sized,
    Definition<D>: StageFactory<P> + 'static,
{
    let identifier = definition.full_identifier().to_string();
    let reserved = [Crossing::Upload, Crossing::Unload]
        .iter()
        .any(|crossing| crossing.identifier() == identifier);
    if reserved {
        return Err(RegistryError::ReservedIdentifier { identifier });
    }

    if factories.contains_key(&identifier) {
        return Err(RegistryError::DuplicateStage {
            kind: definition.kind(),
            identifier,
        });
    }

    factories.insert(identifier, Box::new(definition));

    Ok(())
}

/// Looks `identifier` up among `factories`, or reports it unknown for `kind`.
fn lookup<'f, P: ?Sized>(
    factories: &'f Factories<P>,
    kind: StageKind,
    identifier: &str,
) -> Result<&'f dyn StageFactory<P>, BuildStageError> {
    factories
        .get(identifier)
        .map(|factory| factory.as_ref())
        .ok_or_else(|| BuildStageError::UnknownStage {
            kind,
            identifier: identifier.to_string(),
        })
}

fn prepare<P: ?Sized>(
    factories: &Factories<P>,
    kind: StageKind,
    identifier: &str,
    args: &Value,
    inputs: &[InputArgument],
    slot: &OpenSlot,
) -> Result<Box<P>, BuildStageError> {
    let factory = lookup(factories, kind, identifier)?;

    factory.prepare(args, inputs, slot)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};

    use av_toolbox_gpu::{
        BuildContext,
        Capabilities,
        DeviceSide,
        FormatSet,
        Frame,
        FrameContext,
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
    use crate::{
        BoundInputs,
        Filter,
        InputParam,
        InputSignature,
        Inputs,
        MetaInputs,
        MetaItem,
        MetaSink,
        MetaSinkDefinition,
        MetaTransform,
        Residency,
        Sink,
        Source,
        StageError,
        StageHandle,
        StageInstance,
        StreamType,
        Transform,
    };

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
        type Side = DeviceSide;
        type Opened = ();

        fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
            Ok(())
        }

        fn produces(
            _capabilities: &Capabilities,
            _opened: &Self::Opened,
        ) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _opened: Self::Opened,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Source>, StageError> {
            Ok(Box::new(FileSource))
        }
    }

    struct LiveSourceDefinition;

    impl StageDefinition for LiveSourceDefinition {
        type Args = FileArgs;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "live"
        }

        fn inputs() -> &'static InputSignature {
            &InputSignature::EMPTY
        }
    }

    impl SourceDefinition for LiveSourceDefinition {
        type Side = DeviceSide;
        type Opened = ();

        fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
            Ok(())
        }

        fn reads_stream(_args: &Self::Args) -> bool {
            true
        }

        fn produces(
            _capabilities: &Capabilities,
            _opened: &Self::Opened,
        ) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _opened: Self::Opened,
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
        type Side = DeviceSide;
        type Opened = ();

        fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
            Ok(())
        }

        fn produces(
            _capabilities: &Capabilities,
            _opened: &Self::Opened,
        ) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _opened: Self::Opened,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Source>, StageError> {
            Ok(Box::new(FileSource))
        }
    }

    impl FilterDefinition for Denoise {
        type Side = DeviceSide;

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

    impl Source for FileSource {
        fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
            Ok(None)
        }
    }

    impl StageInstance for FileSource {}

    impl Filter for DenoiseFilter {
        fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
            inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
        }
    }

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

    struct Restamp;

    impl StageInstance for Restamp {}

    impl MetaTransform for Restamp {
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: MetaInputs,
        ) -> Result<Vec<MetaItem>, StageError> {
            Ok(inputs.take(0).into_iter().collect())
        }
    }

    #[derive(Debug, Deserialize, JsonSchema)]
    struct NoArgs {}

    struct RestampDefinition;

    impl StageDefinition for RestampDefinition {
        type Args = NoArgs;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "restamp"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
            SIGNATURE
        }
    }

    impl MetaTransformDefinition for RestampDefinition {
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
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn MetaTransform>, StageError> {
            Ok(Box::new(Restamp))
        }
    }

    #[test]
    fn a_meta_transform_prepares_as_a_transform_that_runs_on_either_side() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_meta_transform::<RestampDefinition>()
            .expect("should register");
        let input = vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)];

        let prepared = registry
            .prepare(
                StageKind::Transform,
                "builtin/restamp",
                &serde_json::json!({}),
                &input,
            )
            .expect("should prepare");

        assert_eq!(prepared.kind(), StageKind::Transform);
        assert_eq!(prepared.residency(), Residency::Either);
        assert!(matches!(prepared, PreparedStage::MetaTransform(_)));
    }

    struct RestampTransformDefinition;

    impl StageDefinition for RestampTransformDefinition {
        type Args = NoArgs;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            "restamp"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
            SIGNATURE
        }
    }

    impl TransformDefinition for RestampTransformDefinition {
        type Side = DeviceSide;

        fn accepts_input(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            _index: usize,
            _offered: StreamFormat,
            _inputs: &[Offered],
        ) -> Result<FormatSet, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn produces(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            _chosen: &[Option<StreamFormat>],
        ) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Transform>, StageError> {
            unreachable!("not exercised by these tests")
        }
    }

    #[test]
    fn a_meta_transform_cannot_share_an_identifier_with_a_transform() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_transform::<RestampTransformDefinition>()
            .expect("should register");

        let error = registry
            .register_meta_transform::<RestampDefinition>()
            .expect_err("should not register beside the transform");

        assert!(matches!(error, RegistryError::DuplicateStage { .. }));
    }

    #[test]
    fn a_transform_cannot_share_an_identifier_with_a_meta_transform() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_meta_transform::<RestampDefinition>()
            .expect("should register");

        let error = registry
            .register_transform::<RestampTransformDefinition>()
            .expect_err("should not register beside the meta transform");

        assert!(matches!(error, RegistryError::DuplicateStage { .. }));
    }

    struct LogStage;

    impl StageInstance for LogStage {}

    impl MetaSink for LogStage {
        fn consume(&mut self, _context: &mut FrameContext, _inputs: MetaInputs) -> Result<(), StageError> {
            Ok(())
        }
    }

    struct LogDefinition;

    impl StageDefinition for LogDefinition {
        type Args = NoArgs;

        fn package() -> &'static str {
            "test"
        }

        fn identifier() -> &'static str {
            "log"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature =
                &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
            SIGNATURE
        }
    }

    impl MetaSinkDefinition for LogDefinition {
        fn accepts_input(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            _index: usize,
            offered: StreamFormat,
            _inputs: &[Offered],
        ) -> Result<FormatSet, StageError> {
            Ok(FormatSet::of([offered.sample()]))
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn MetaSink>, StageError> {
            Ok(Box::new(LogStage))
        }
    }

    /// A sink registered under the meta sink's identifier, to test the two cannot share one.
    struct ClashingSinkDefinition;

    impl StageDefinition for ClashingSinkDefinition {
        type Args = NoArgs;

        fn package() -> &'static str {
            "test"
        }

        fn identifier() -> &'static str {
            "log"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature =
                &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
            SIGNATURE
        }
    }

    impl SinkDefinition for ClashingSinkDefinition {
        type Side = DeviceSide;

        fn accepts_input(
            _capabilities: &Capabilities,
            _args: &Self::Args,
            _index: usize,
            offered: StreamFormat,
            _inputs: &[Offered],
        ) -> Result<FormatSet, StageError> {
            Ok(FormatSet::of([offered.sample()]))
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _args: Self::Args,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Sink>, StageError> {
            unreachable!("not exercised by these tests")
        }
    }

    #[test]
    fn a_meta_sink_prepares_as_a_sink_that_runs_on_either_side() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_meta_sink::<LogDefinition>()
            .expect("should register");
        let inputs = vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)];

        let prepared = registry
            .prepare(StageKind::Sink, "test/log", &serde_json::json!({}), &inputs)
            .expect("should prepare");

        assert!(matches!(prepared, PreparedStage::MetaSink(_)));
        assert_eq!(prepared.kind(), StageKind::Sink);
        assert_eq!(prepared.residency(), Residency::Either);
    }

    #[test]
    fn a_meta_sink_cannot_share_an_identifier_with_a_sink() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_sink::<ClashingSinkDefinition>()
            .expect("should register");

        let result = registry.register_meta_sink::<LogDefinition>();

        assert!(matches!(result, Err(RegistryError::DuplicateStage { .. })));
    }

    #[test]
    fn a_sink_cannot_share_an_identifier_with_a_meta_sink() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_meta_sink::<LogDefinition>()
            .expect("should register");

        let result = registry.register_sink::<ClashingSinkDefinition>();

        assert!(matches!(result, Err(RegistryError::DuplicateStage { .. })));
    }

    #[test]
    fn a_meta_sink_is_listed_with_the_stages() {
        let mut registry = PipelineRegistry::new();
        registry
            .register_meta_sink::<LogDefinition>()
            .expect("should register");

        let identifiers: Vec<&str> = registry.stages().map(|stage| stage.full_identifier()).collect();

        assert_eq!(identifiers, vec!["test/log"]);
    }

    /// A source named after a crossing, `builtin/unload` when `UNLOAD` is set and `builtin/upload`
    /// otherwise.
    struct CrossingImpostor<const UNLOAD: bool>;

    impl<const UNLOAD: bool> StageDefinition for CrossingImpostor<UNLOAD> {
        type Args = FileArgs;

        fn package() -> &'static str {
            "builtin"
        }

        fn identifier() -> &'static str {
            if UNLOAD { "unload" } else { "upload" }
        }

        fn inputs() -> &'static InputSignature {
            &InputSignature::EMPTY
        }
    }

    impl<const UNLOAD: bool> SourceDefinition for CrossingImpostor<UNLOAD> {
        type Side = DeviceSide;
        type Opened = ();

        fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
            Ok(())
        }

        fn produces(
            _capabilities: &Capabilities,
            _opened: &Self::Opened,
        ) -> Result<OutputFormats, StageError> {
            unreachable!("not exercised by these tests")
        }

        fn build<R: Runtime>(
            _context: &BuildContext<R>,
            _opened: Self::Opened,
            _inputs: BoundInputs,
            _formats: &ResolvedFormats,
        ) -> Result<Box<dyn Source>, StageError> {
            Ok(Box::new(FileSource))
        }
    }

    #[test]
    fn a_stage_named_after_a_crossing_is_rejected() {
        let mut registry = PipelineRegistry::new();

        let upload = registry
            .register_source::<CrossingImpostor<false>>()
            .expect_err("builtin/upload is reserved");
        let unload = registry
            .register_source::<CrossingImpostor<true>>()
            .expect_err("builtin/unload is reserved");

        assert!(matches!(upload, RegistryError::ReservedIdentifier { .. }));
        assert!(matches!(unload, RegistryError::ReservedIdentifier { .. }));
    }

    #[test]
    fn another_builtin_stage_still_registers() {
        let mut registry = PipelineRegistry::new();

        registry
            .register_source::<FileSourceDefinition>()
            .expect("only the crossing identifiers are reserved");
    }

    fn registry_with_file_and_live_sources() -> PipelineRegistry {
        let mut registry = PipelineRegistry::new();
        registry
            .register_source::<FileSourceDefinition>()
            .expect("should register");
        registry
            .register_source::<LiveSourceDefinition>()
            .expect("should register");

        registry
    }

    #[test]
    fn reads_stream_answers_true_for_a_source_that_overrides_it() {
        let registry = registry_with_file_and_live_sources();
        let args = serde_json::json!({ "input": "clip.y4m" });

        let reads_stream = registry
            .reads_stream("builtin/live", &args)
            .expect("should answer");

        assert!(reads_stream);
    }

    #[test]
    fn reads_stream_defaults_to_false() {
        let registry = registry_with_file_and_live_sources();
        let args = serde_json::json!({ "input": "clip.y4m" });

        let reads_stream = registry
            .reads_stream("builtin/file", &args)
            .expect("should answer");

        assert!(!reads_stream);
    }

    #[test]
    fn reads_stream_rejects_invalid_args() {
        let registry = registry_with_file_and_live_sources();
        let args = serde_json::json!({});

        let error = registry
            .reads_stream("builtin/file", &args)
            .expect_err("input is required");

        assert!(matches!(error, BuildStageError::InvalidArgs { .. }));
    }

    #[test]
    fn reads_stream_rejects_an_unknown_identifier() {
        let registry = registry_with_file_and_live_sources();
        let args = serde_json::json!({ "input": "clip.y4m" });

        let error = registry
            .reads_stream("builtin/missing", &args)
            .expect_err("the identifier is not registered");

        assert!(matches!(
            error,
            BuildStageError::UnknownStage {
                kind: StageKind::Source,
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
            let filter = filter.into_device().expect("a device filter");
            drop(filter);

            let recorded_sigma_scale = f32::from_bits(RECORDED_SIGMA_SCALE.load(Ordering::SeqCst));
            assert_eq!(recorded_sigma_scale, 0.8);
        }
    }
}
