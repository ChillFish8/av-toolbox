use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use serde_json::json;

use crate::{
    ArtefactData,
    ArtefactDecl,
    ArtefactKind,
    ArtefactPiece,
    ArtefactSink,
    ArtefactSinkDefinition,
    BoundInputs,
    BuildStageError,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    KeyFilter,
    MergedArtefact,
    MetaInputs,
    MetaSink,
    MetaSinkDefinition,
    Pattern,
    PipelineRegistry,
    RegistryError,
    Sink,
    SinkDefinition,
    StageArtefacts,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
    StreamType,
    WantedArtefacts,
    merge_single,
};

fn piece(chunk: Option<u64>, value: serde_json::Value) -> ArtefactPiece {
    ArtefactPiece {
        chunk,
        data: ArtefactData::Structured(value),
    }
}

#[test]
fn the_default_merge_returns_a_lone_piece() {
    let pieces = vec![piece(None, json!([1, 2]))];

    let merged = merge_single("cuts", pieces).expect("one piece merges");

    assert_eq!(merged, ArtefactData::Structured(json!([1, 2])));
}

#[test]
fn the_default_merge_refuses_two_pieces() {
    let pieces = vec![piece(Some(0), json!(1)), piece(Some(1), json!(2))];

    let error = merge_single("cuts", pieces).expect_err("two pieces are refused");

    assert!(error.to_string().contains("cuts"));
}

#[test]
fn data_matches_only_its_own_kind() {
    let structured = ArtefactData::Structured(json!({}));
    let opaque = ArtefactData::Opaque(vec![1, 2, 3]);
    let table = ArtefactKind::Opaque { extension: "tbl" };

    assert!(structured.is_kind(ArtefactKind::Structured));
    assert!(!structured.is_kind(table));
    assert!(opaque.is_kind(table));
    assert!(!opaque.is_kind(ArtefactKind::Structured));
}

#[test]
fn a_structured_artefact_is_written_as_json() {
    assert_eq!(ArtefactKind::Structured.extension(), "json");
    assert_eq!(ArtefactKind::Opaque { extension: "tbl" }.extension(), "tbl");
}

#[test]
fn wanted_artefacts_hold_only_their_names() {
    let wanted = WantedArtefacts::new(vec!["table"]);

    assert!(wanted.contains("table"));
    assert!(!wanted.contains("cuts"));
    assert!(!wanted.is_empty());
    assert!(WantedArtefacts::default().is_empty());
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct NoArgs {}

/// A filter declaring the artefacts `ARTEFACTS` picks, registered to check the declarations.
struct Declaring<const ARTEFACTS: usize>;

const GOOD: &[ArtefactDecl] = &[
    ArtefactDecl::opaque("table", "tbl", "A grain table."),
    ArtefactDecl::structured("cuts", "Where each scene starts."),
];
const REPEATED: &[ArtefactDecl] = &[
    ArtefactDecl::structured("cuts", "Where each scene starts."),
    ArtefactDecl::structured("cuts", "Where each scene starts."),
];
const DOTTED: &[ArtefactDecl] = &[ArtefactDecl::structured("grain.table", "A grain table.")];
const EMPTY_NAME: &[ArtefactDecl] = &[ArtefactDecl::structured("", "Nothing in particular.")];
const SLASHED: &[ArtefactDecl] = &[ArtefactDecl::structured("grain/table", "A grain table.")];

impl<const ARTEFACTS: usize> StageDefinition for Declaring<ARTEFACTS> {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "declaring"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        match ARTEFACTS {
            0 => GOOD,
            1 => REPEATED,
            2 => DOTTED,
            3 => EMPTY_NAME,
            _ => SLASHED,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl<const ARTEFACTS: usize> FilterDefinition for Declaring<ARTEFACTS> {
    type Side = av_toolbox_gpu::DeviceSide;

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
        _chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        unreachable!("not exercised by these tests")
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        unreachable!("not exercised by these tests")
    }
}

#[test]
fn a_registered_stage_lists_its_artefacts() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_filter::<Declaring<0>>()
        .expect("should register");

    let info = registry.stages().next().expect("one stage");
    let artefacts = info.artefacts();

    assert_eq!(artefacts.declared(), GOOD);
    assert_eq!(artefacts.find("cuts"), Some(&GOOD[1]));
    assert_eq!(artefacts.find("missing"), None);
}

#[test]
fn a_stage_declaring_one_artefact_twice_is_refused() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_filter::<Declaring<1>>();

    assert!(matches!(
        result,
        Err(RegistryError::DuplicateArtefact { name: "cuts", .. })
    ));
}

#[test]
fn an_artefact_name_holding_a_dot_is_refused() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_filter::<Declaring<2>>();

    assert!(matches!(result, Err(RegistryError::InvalidArtefactName { .. })));
}

#[test]
fn an_empty_artefact_name_is_refused() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_filter::<Declaring<3>>();

    assert!(matches!(result, Err(RegistryError::InvalidArtefactName { .. })));
}

#[test]
fn an_artefact_name_holding_a_slash_is_refused() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_filter::<Declaring<4>>();

    assert!(matches!(result, Err(RegistryError::InvalidArtefactName { .. })));
}

#[test]
fn a_stage_declaring_nothing_merges_with_the_default() {
    let artefacts = StageArtefacts::of::<Declaring<0>>();
    let pieces = vec![piece(None, json!(7))];

    let merged = artefacts.merge("cuts", pieces).expect("one piece merges");

    assert_eq!(merged, ArtefactData::Structured(json!(7)));
    assert!(StageArtefacts::NONE.declared().is_empty());
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
struct FilesArgs {
    include: Option<Vec<Pattern>>,
}

struct FilesSink;

impl ArtefactSink for FilesSink {
    fn write(&mut self, artefacts: &[&MergedArtefact]) -> Result<usize, StageError> {
        Ok(artefacts.len())
    }
}

struct Files;

impl StageDefinition for Files {
    type Args = FilesArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "files"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl ArtefactSinkDefinition for Files {
    fn selection(args: &Self::Args) -> KeyFilter {
        KeyFilter::new(args.include.clone(), None)
    }

    fn build(_args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        Ok(Box::new(FilesSink))
    }
}

/// An artefact sink that declares an input, which artefact sinks cannot take.
struct ReadingFiles;

impl StageDefinition for ReadingFiles {
    type Args = FilesArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "reading-files"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl ArtefactSinkDefinition for ReadingFiles {
    fn selection(args: &Self::Args) -> KeyFilter {
        KeyFilter::new(args.include.clone(), None)
    }

    fn build(_args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        Ok(Box::new(FilesSink))
    }
}

#[test]
fn an_artefact_sink_prepares_its_selection_without_a_device() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_artefact_sink::<Files>()
        .expect("should register");
    let args = json!({ "include": ["filter.*"] });

    let prepared = registry
        .prepare_artefact_sink("test/files", &args)
        .expect("should prepare");
    let selection = prepared.selection();

    assert!(selection.keeps("filter.grain.table"));
    assert!(!selection.keeps("mask.edges.map"));
    assert!(registry.is_artefact_sink("test/files"));
}

#[test]
fn an_artefact_sink_builds_from_its_prepared_arguments() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_artefact_sink::<Files>()
        .expect("should register");
    let prepared = registry
        .prepare_artefact_sink("test/files", &json!({}))
        .expect("should prepare");

    let mut sink = prepared.build().expect("should build");
    let written = sink.write(&[]).expect("writes nothing");

    assert_eq!(written, 0);
}

#[test]
fn an_artefact_sink_with_bad_arguments_is_refused() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_artefact_sink::<Files>()
        .expect("should register");
    let args = json!({ "include": [""] });

    let result = registry.prepare_artefact_sink("test/files", &args);

    assert!(matches!(result, Err(BuildStageError::InvalidArgs { .. })));
}

#[test]
fn an_artefact_sink_declaring_inputs_is_refused() {
    let mut registry = PipelineRegistry::new();

    let result = registry.register_artefact_sink::<ReadingFiles>();

    assert!(matches!(
        result,
        Err(RegistryError::ArtefactSinkTakesInputs { .. })
    ));
}

#[test]
fn an_artefact_sink_is_listed_with_the_stages_as_a_sink() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_artefact_sink::<Files>()
        .expect("should register");

    let listed: Vec<(StageKind, &str)> = registry
        .stages()
        .map(|info| (info.kind(), info.full_identifier()))
        .collect();

    assert_eq!(listed, vec![(StageKind::Sink, "test/files")]);
}

#[test]
fn an_unregistered_identifier_is_not_an_artefact_sink() {
    let registry = PipelineRegistry::new();

    assert!(!registry.is_artefact_sink("test/files"));
}

/// A frame sink under `test/log`, to test the sink kinds cannot share an identifier.
struct LogSink;

impl StageDefinition for LogSink {
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

impl SinkDefinition for LogSink {
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

struct LogMetaStage;

impl StageInstance for LogMetaStage {}

impl MetaSink for LogMetaStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: MetaInputs) -> Result<(), StageError> {
        Ok(())
    }
}

/// A metadata sink under `test/log`, to test the sink kinds cannot share an identifier.
struct LogMetaSink;

impl StageDefinition for LogMetaSink {
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

impl MetaSinkDefinition for LogMetaSink {
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
        Ok(Box::new(LogMetaStage))
    }
}

/// An artefact sink under `test/log`, to test the sink kinds cannot share an identifier.
struct ClashingArtefactSink;

impl StageDefinition for ClashingArtefactSink {
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
        &InputSignature::EMPTY
    }
}

impl ArtefactSinkDefinition for ClashingArtefactSink {
    fn selection(_args: &Self::Args) -> KeyFilter {
        KeyFilter::default()
    }

    fn build(_args: Self::Args) -> Result<Box<dyn ArtefactSink>, StageError> {
        Ok(Box::new(FilesSink))
    }
}

#[test]
fn an_artefact_sink_cannot_share_an_identifier_with_any_sink() {
    let mut sinks = PipelineRegistry::new();
    sinks.register_sink::<LogSink>().expect("should register");
    let mut meta_sinks = PipelineRegistry::new();
    meta_sinks
        .register_meta_sink::<LogMetaSink>()
        .expect("should register");

    let after_sink = sinks.register_artefact_sink::<ClashingArtefactSink>();
    let after_meta_sink = meta_sinks.register_artefact_sink::<ClashingArtefactSink>();

    assert!(matches!(after_sink, Err(RegistryError::DuplicateStage { .. })));
    assert!(matches!(
        after_meta_sink,
        Err(RegistryError::DuplicateStage { .. })
    ));
}

#[test]
fn no_sink_can_share_an_identifier_with_an_artefact_sink() {
    let mut registry = PipelineRegistry::new();
    registry
        .register_artefact_sink::<ClashingArtefactSink>()
        .expect("should register");

    let as_sink = registry.register_sink::<LogSink>();
    let as_meta_sink = registry.register_meta_sink::<LogMetaSink>();

    assert!(matches!(as_sink, Err(RegistryError::DuplicateStage { .. })));
    assert!(matches!(as_meta_sink, Err(RegistryError::DuplicateStage { .. })));
}
