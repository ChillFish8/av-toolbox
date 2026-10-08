use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
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
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    PipelineRegistry,
    RegistryError,
    StageArtefacts,
    StageDefinition,
    StageError,
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
    ArtefactDecl::opaque("table", "tbl"),
    ArtefactDecl::structured("cuts"),
];
const REPEATED: &[ArtefactDecl] = &[ArtefactDecl::structured("cuts"), ArtefactDecl::structured("cuts")];
const DOTTED: &[ArtefactDecl] = &[ArtefactDecl::structured("grain.table")];
const EMPTY_NAME: &[ArtefactDecl] = &[ArtefactDecl::structured("")];

impl<const ARTEFACTS: usize> StageDefinition for Declaring<ARTEFACTS> {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "declaring"
    }

    fn artefacts() -> &'static [ArtefactDecl] {
        match ARTEFACTS {
            0 => GOOD,
            1 => REPEATED,
            2 => DOTTED,
            _ => EMPTY_NAME,
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
fn a_stage_declaring_nothing_merges_with_the_default() {
    let artefacts = StageArtefacts::of::<Declaring<0>>();
    let pieces = vec![piece(None, json!(7))];

    let merged = artefacts.merge("cuts", pieces).expect("one piece merges");

    assert_eq!(merged, ArtefactData::Structured(json!(7)));
    assert!(StageArtefacts::NONE.declared().is_empty());
}
