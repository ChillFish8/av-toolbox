use std::collections::HashMap;

use schemars::Schema;
use serde_json::Value;

use crate::factory::Definition;
use crate::{
    AnyDemand,
    AnyProbe,
    ArtefactSinkDefinition,
    BuildStageError,
    CombineProbeFn,
    Crossing,
    DemandDefinition,
    DemandKind,
    DemandsFn,
    FilterDefinition,
    InputArgument,
    InputDemand,
    LayerDemand,
    MaskDefinition,
    MetaSinkDefinition,
    OpenSlot,
    PathEdge,
    PathView,
    PlaceError,
    PlaceFn,
    PreparedArtefactSink,
    PreparedFilter,
    PreparedMask,
    PreparedMetaSink,
    PreparedProvider,
    PreparedSink,
    PreparedSource,
    PreparedStage,
    ProbeDefinition,
    ProbeKind,
    ProbedValue,
    ProviderDefinition,
    ProviderSide,
    RegistryError,
    RequestError,
    Residency,
    RunProbeFn,
    SinkDefinition,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageFactory,
    StageInfo,
    StageKind,
    probe,
};

type Factories<P> = HashMap<String, Box<dyn StageFactory<P>>>;

/// A registered demand kind's request schema and how to describe one of its requests.
struct RegisteredDemand {
    kind: DemandKind,
    schema: Schema,
    describe: fn(&AnyDemand) -> String,
}

/// A registered probe kind and the erased functions that run it.
struct RegisteredProbe {
    kind: ProbeKind,
    describe: fn(&AnyProbe) -> String,
    show: fn(&ProbedValue) -> String,
    run: RunProbeFn,
    combine: CombineProbeFn,
}

/// The provider registered for one demand kind.
struct RegisteredProvider {
    full_identifier: String,
    place: PlaceFn,
    demands: DemandsFn,
    residency: Residency,
}

/// The registry tracks all stages available for use within the pipeline.
///
/// Stages are keyed by their fully qualified `package/stage` identifier. `builtin/upload` and
/// `builtin/unload` name the crossings, so no stage may take either.
#[derive(Default)]
pub struct PipelineRegistry {
    sources: Factories<dyn PreparedSource>,
    filters: Factories<dyn PreparedFilter>,
    masks: Factories<dyn PreparedMask>,
    providers: Factories<dyn PreparedProvider>,
    sinks: Factories<dyn PreparedSink>,
    meta_sinks: Factories<dyn PreparedMetaSink>,
    artefact_sinks: Factories<dyn PreparedArtefactSink>,
    demands: HashMap<&'static str, RegisteredDemand>,
    served: HashMap<DemandKind, RegisteredProvider>,
    probes: HashMap<&'static str, RegisteredProbe>,
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

    /// Registers the provider of `P::Demand`, registering that demand kind too if it is new.
    ///
    /// Each demand kind has at most one provider.
    pub fn register_provider<P: ProviderDefinition>(&mut self) -> Result<(), RegistryError> {
        let kind = DemandKind::of::<P::Demand>();
        if self.served.contains_key(&kind) {
            return Err(RegistryError::DuplicateProvider { kind: kind.name() });
        }

        let definition = Definition::<P>::new(StageKind::Provider)?;
        let full_identifier = definition.full_identifier().to_string();
        insert(&mut self.providers, definition)?;

        if !self.demands.contains_key(kind.name()) {
            self.register_demand::<P::Demand>()?;
        }

        let registered = RegisteredProvider {
            full_identifier,
            place: place_erased::<P>,
            demands: demands_erased::<P>,
            residency: <P::Side as ProviderSide>::RESIDENCY,
        };
        self.served.insert(kind, registered);

        Ok(())
    }

    /// The full identifier of the provider serving `kind`, and how it places itself.
    pub fn provider_for(&self, kind: DemandKind) -> Option<(&str, PlaceFn)> {
        let registered = self.served.get(&kind)?;

        Some((&registered.full_identifier, registered.place))
    }

    /// How the provider serving `kind` reports its own demands from its args.
    pub fn provider_demands(&self, kind: DemandKind) -> Option<DemandsFn> {
        let registered = self.served.get(&kind)?;

        Some(registered.demands)
    }

    /// The side the provider serving `kind` was registered for.
    pub fn provider_side(&self, kind: DemandKind) -> Option<Residency> {
        self.served.get(&kind).map(|registered| registered.residency)
    }

    /// Every registered provider as its full identifier, the demand kind it serves and its side.
    pub fn providers(&self) -> impl Iterator<Item = (&str, DemandKind, Residency)> {
        self.served.iter().map(|(kind, registered)| {
            let identifier = registered.full_identifier.as_str();

            (identifier, *kind, registered.residency)
        })
    }

    pub fn register_sink<D: SinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        self.check_sink_identifier(definition.full_identifier())?;

        insert(&mut self.sinks, definition)
    }

    /// Registers a metadata-only sink, which shares the sink identifiers.
    pub fn register_meta_sink<D: MetaSinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        self.check_sink_identifier(definition.full_identifier())?;

        insert(&mut self.meta_sinks, definition)
    }

    /// Registers an artefact sink, which shares the sink identifiers and takes no inputs.
    pub fn register_artefact_sink<D: ArtefactSinkDefinition>(&mut self) -> Result<(), RegistryError> {
        let definition = Definition::<D>::new(StageKind::Sink)?;
        let identifier = definition.full_identifier();
        if !D::inputs().is_empty() {
            return Err(RegistryError::ArtefactSinkTakesInputs {
                identifier: identifier.to_string(),
            });
        }

        self.check_sink_identifier(identifier)?;

        insert(&mut self.artefact_sinks, definition)
    }

    /// Whether `identifier` names a registered artefact sink.
    pub fn is_artefact_sink(&self, identifier: &str) -> bool {
        self.artefact_sinks.contains_key(identifier)
    }

    /// Deserializes an artefact sink's arguments, without needing a device.
    pub fn prepare_artefact_sink(
        &self,
        identifier: &str,
        args: &Value,
    ) -> Result<Box<dyn PreparedArtefactSink>, BuildStageError> {
        let slot = OpenSlot::new();

        prepare(
            &self.artefact_sinks,
            StageKind::Sink,
            identifier,
            args,
            &[],
            &slot,
        )
    }

    /// Refuses an identifier any kind of sink already holds.
    fn check_sink_identifier(&self, identifier: &str) -> Result<(), RegistryError> {
        let taken = self.sinks.contains_key(identifier)
            || self.meta_sinks.contains_key(identifier)
            || self.artefact_sinks.contains_key(identifier);
        if !taken {
            return Ok(());
        }

        Err(RegistryError::DuplicateStage {
            kind: StageKind::Sink,
            identifier: identifier.to_string(),
        })
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
            StageKind::Provider => {
                prepare(&self.providers, kind, identifier, args, inputs, slot).map(PreparedStage::Provider)
            },
            // The graph places crossings itself, so none is ever registered.
            StageKind::Crossing => Err(BuildStageError::UnknownStage {
                kind,
                identifier: identifier.to_string(),
            }),
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

    /// Registers a demand kind, rejecting a second kind of the same name.
    pub fn register_demand<D: DemandDefinition>(&mut self) -> Result<(), RegistryError> {
        let kind = DemandKind::of::<D>();
        if self.demands.contains_key(kind.name()) {
            return Err(RegistryError::DuplicateDemand { kind: kind.name() });
        }

        let registered = RegisteredDemand {
            kind,
            schema: schemars::schema_for!(D::Request),
            describe: describe_as::<D>,
        };
        self.demands.insert(kind.name(), registered);

        Ok(())
    }

    /// Registers a probe kind, rejecting a second kind of the same name.
    pub fn register_probe<P: ProbeDefinition>(&mut self) -> Result<(), RegistryError> {
        let kind = ProbeKind::of::<P>();
        if self.probes.contains_key(kind.name()) {
            return Err(RegistryError::DuplicateProbe { kind: kind.name() });
        }

        let registered = RegisteredProbe {
            kind,
            describe: probe::describe_as::<P>,
            show: probe::show_as::<P>,
            run: probe::run_as::<P>,
            combine: probe::combine_as::<P>,
        };
        self.probes.insert(kind.name(), registered);

        Ok(())
    }

    /// The registered probe kind named `name`.
    pub fn probe_kind(&self, name: &str) -> Option<ProbeKind> {
        self.probes.get(name).map(|registered| registered.kind)
    }

    /// A short description of `probe`, or its kind's name when the kind is not registered.
    pub fn describe_probe(&self, probe: &AnyProbe) -> String {
        let kind = probe.kind();

        match self.probes.get(kind.name()) {
            Some(registered) => (registered.describe)(probe),
            None => kind.name().to_string(),
        }
    }

    pub fn probe_runner(&self, kind: ProbeKind) -> Option<RunProbeFn> {
        self.probes.get(kind.name()).map(|registered| registered.run)
    }

    pub fn probe_combiner(&self, kind: ProbeKind) -> Option<CombineProbeFn> {
        self.probes.get(kind.name()).map(|registered| registered.combine)
    }

    /// What a probe of `kind` found, written out for a log line.
    pub fn show_probed(&self, kind: ProbeKind, value: &ProbedValue) -> String {
        match self.probes.get(kind.name()) {
            Some(registered) => (registered.show)(value),
            None => kind.name().to_string(),
        }
    }

    /// The registered demand kind named `name`.
    pub fn demand_kind(&self, name: &str) -> Option<DemandKind> {
        self.demands.get(name).map(|registered| registered.kind)
    }

    /// The schema of a registered demand kind's requests.
    pub fn demand_schema(&self, kind: DemandKind) -> Option<&Schema> {
        self.demands.get(kind.name()).map(|registered| &registered.schema)
    }

    /// A short description of `demand`, or its kind's name when the kind is not registered.
    pub fn describe(&self, demand: &AnyDemand) -> String {
        let kind = demand.kind();

        match self.demands.get(kind.name()) {
            Some(registered) => (registered.describe)(demand),
            None => kind.name().to_string(),
        }
    }

    /// Describes every registered stage.
    pub fn stages(&self) -> impl Iterator<Item = StageInfo<'_>> {
        let sources = self.sources.values().map(|factory| factory.info());
        let filters = self.filters.values().map(|factory| factory.info());
        let masks = self.masks.values().map(|factory| factory.info());
        let providers = self.providers.values().map(|factory| factory.info());
        let sinks = self.sinks.values().map(|factory| factory.info());
        let meta_sinks = self.meta_sinks.values().map(|factory| factory.info());
        let artefact_sinks = self.artefact_sinks.values().map(|factory| factory.info());

        sources
            .chain(filters)
            .chain(masks)
            .chain(providers)
            .chain(sinks)
            .chain(meta_sinks)
            .chain(artefact_sinks)
    }
}

/// Describes `demand` as kind `D` would, or by the kind's name when its request does not read as `D`'s.
fn describe_as<D: DemandDefinition>(demand: &AnyDemand) -> String {
    match demand.request::<D>() {
        Some(request) => D::describe(&request),
        None => D::kind().to_string(),
    }
}

/// Places `demand` with `P`'s `place`, once its request reads as `P`'s demand kind.
fn place_erased<P: ProviderDefinition>(
    demand: &AnyDemand,
    path: &PathView<'_>,
) -> Result<PathEdge, PlaceError> {
    let request = demand.try_request::<P::Demand>().map_err(unreadable_request)?;

    P::place(&request, path)
}

/// Reads `args` as `P`'s layer demands and reports what `P` demands for them.
fn demands_erased<P: ProviderDefinition>(args: &Value) -> Result<Vec<InputDemand>, StageError> {
    let layers: Vec<LayerDemand> = serde_json::from_value(args.clone()).map_err(unreadable_args)?;

    Ok(P::demands(&layers))
}

fn unreadable_args(error: serde_json::Error) -> StageError {
    StageError::with_source("the provider's layer demands cannot be read", error)
}

fn unreadable_request(error: RequestError) -> PlaceError {
    let message = format!("its request cannot be read because {error}");

    PlaceError::msg(message)
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
    use crate::tests::demand::{Breaks, Cuts};
    use crate::{
        AnyDemand,
        BoundInputs,
        DemandKind,
        Filter,
        InputParam,
        InputSignature,
        Inputs,
        MetaInputs,
        MetaSink,
        MetaSinkDefinition,
        Residency,
        Sink,
        Source,
        StageError,
        StageHandle,
        StageInstance,
        StreamType,
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

        fn description() -> &'static str {
            "A stage used by tests."
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

        fn description() -> &'static str {
            "A stage used by tests."
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

        fn description() -> &'static str {
            "A stage used by tests."
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

        fn description() -> &'static str {
            "A stage used by tests."
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
        fn process(
            &mut self,
            _context: &mut FrameContext,
            mut inputs: Inputs,
        ) -> Result<Vec<Frame>, StageError> {
            let frame = inputs
                .take_frame(0)
                .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

            Ok(vec![frame])
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

    #[derive(Debug, Deserialize, JsonSchema)]
    struct NoArgs {}

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

        fn description() -> &'static str {
            "A stage used by tests."
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

        fn description() -> &'static str {
            "A stage used by tests."
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

        fn description() -> &'static str {
            "A stage used by tests."
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
    fn a_demand_kind_registers_once() {
        let mut registry = PipelineRegistry::new();

        registry.register_demand::<Cuts>().expect("should register");
        let error = registry
            .register_demand::<Cuts>()
            .expect_err("a second registration is refused");

        assert!(matches!(error, RegistryError::DuplicateDemand { kind: "cuts" }));
    }

    #[test]
    fn a_registered_demand_kind_is_found_and_described_by_name() {
        let mut registry = PipelineRegistry::new();
        registry.register_demand::<Cuts>().expect("should register");

        let demand = AnyDemand::new::<Cuts>(Breaks { every: 4 });

        assert_eq!(registry.demand_kind("cuts"), Some(DemandKind::of::<Cuts>()));
        assert_eq!(registry.demand_kind("other"), None);
        assert_eq!(registry.describe(&demand), "cuts");
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
        use av_toolbox_gpu::{Backend, test_device};

        use super::*;

        fn backend() -> Backend {
            Backend::open_default(test_device()).expect("should open")
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
