use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    PipelineRegistry,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourInfo,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
use av_toolbox_graph::{
    CompiledGraph,
    RenderOptions,
    default_passes,
    optimise,
    render_ascii,
    render_html,
    solve,
};
use av_toolbox_graph::{InputRef, Span, Spanned, StageSpec, build};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct ReadStage;

impl StageInstance for ReadStage {}

impl Source for ReadStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

struct Read;

impl StageDefinition for Read {
    type Args = NoArgs;

    fn package() -> &'static str {
        "demo"
    }

    fn identifier() -> &'static str {
        "read"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Read {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(())
    }

    fn produces(_capabilities: &Capabilities, _opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = fixture_format();

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(ReadStage))
    }
}

struct WriteStage;

impl StageInstance for WriteStage {}

impl Sink for WriteStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        Ok(())
    }
}

struct Write;

impl StageDefinition for Write {
    type Args = NoArgs;

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
}

impl SinkDefinition for Write {
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
        Ok(Box::new(WriteStage))
    }
}

/// A fixed frame format, used as the demo source's arbitrary starting point.
fn fixture_format() -> StreamFormat {
    StreamFormat::Frames(FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    })
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
        workers: None,
        span: Span::unknown(),
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<Read>().expect("should register");
    registry.register_sink::<Write>().expect("should register");

    registry
}

#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
#[test]
fn a_pipeline_builds_optimises_renders_and_instantiates() {
    use av_toolbox_gpu::{Backend, Device};

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

    let html = render_html(&graph, &options);
    assert!(html.starts_with("<!DOCTYPE html>"));
    assert!(html.contains("\"input\""));

    let backend = Backend::open_default(Device::Default).expect("should open");
    let solved = solve(&graph, &registry, backend.capabilities()).expect("should solve");

    let stages = graph
        .instantiate(&registry, &backend, &solved)
        .expect("should instantiate");
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
