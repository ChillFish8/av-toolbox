use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    PipelineRegistry,
    Sink,
    Source,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
};
use av_toolbox_graph::{
    CompiledGraph,
    InputRef,
    RenderOptions,
    Span,
    Spanned,
    StageSpec,
    build,
    default_passes,
    optimise,
    render_ascii,
    render_mermaid,
    render_svg,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct ReadStage;

impl StageInstance for ReadStage {}
impl Source for ReadStage {}

struct Read;

impl StageDefinition for Read {
    type Args = NoArgs;
    type Instance = ReadStage;

    fn package() -> &'static str {
        "demo"
    }

    fn identifier() -> &'static str {
        "read"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }

    fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
        Ok(ReadStage)
    }
}

struct WriteStage;

impl StageInstance for WriteStage {}
impl Sink for WriteStage {}

struct Write;

impl StageDefinition for Write {
    type Args = NoArgs;
    type Instance = WriteStage;

    fn package() -> &'static str {
        "demo"
    }

    fn identifier() -> &'static str {
        "write"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }

    fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
        Ok(WriteStage)
    }
}

fn spec(kind: StageKind, name: &str, stage_type: &str, inputs: Vec<&str>, args: Value) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: inputs
            .into_iter()
            .map(|target| {
                Spanned::unknown(InputRef {
                    name: None,
                    target: target.to_string(),
                })
            })
            .collect(),
        args: Spanned::unknown(args),
        span: Span::unknown(),
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<Read>().expect("should register");
    registry.register_sink::<Write>().expect("should register");

    registry
}

#[test]
fn a_pipeline_builds_optimises_renders_and_instantiates() {
    let specs = vec![
        spec(StageKind::Source, "input", "demo/read", vec![], json!({})),
        spec(StageKind::Sink, "out", "demo/write", vec!["input"], json!({})),
    ];

    let registry = registry();
    let built = build(&registry, &specs).expect("should build");
    assert!(built.warnings().is_empty());

    let mut graph: CompiledGraph = built.into_graph();
    let reports = optimise(&mut graph, &default_passes());
    assert_eq!(reports.len(), 3);

    let options = RenderOptions::default();
    assert!(render_ascii(&graph, &options).contains("input"));
    assert!(render_mermaid(&graph, &options).starts_with("flowchart TD"));

    let svg = render_svg(&graph, &options);
    assert!(svg.contains("<svg"));
    assert!(svg.contains("input"));

    let stages = graph.instantiate(&registry).expect("should instantiate");
    assert_eq!(stages.len(), 2);
}

#[test]
fn a_broken_reference_is_reported_against_its_definition() {
    let mut specs = vec![
        spec(StageKind::Source, "input", "demo/read", vec![], json!({})),
        spec(StageKind::Sink, "out", "demo/write", vec!["inputt"], json!({})),
    ];
    specs[1].inputs[0].span = Span::new(120, 127);

    let registry = registry();
    let diagnostics = build(&registry, &specs).expect_err("should fail");
    let rendered = diagnostics.to_string();

    assert!(rendered.contains("sink.out.inputs[0]"));
    assert!(rendered.contains("did you mean input"));

    let reported = diagnostics.iter().next().expect("should report one problem");
    assert_eq!(reported.span(), Span::new(120, 127));
}
