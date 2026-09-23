// The engine driven through its real entry point against a real device. It is here because
// nothing else in the workspace calls `Engine::build`, and a surface nobody calls is a surface
// nobody has checked.
#![cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::sync::Mutex;

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    PipelineRegistry,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    Capabilities,
    ColourInfo,
    Device,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use av_toolbox_graph::{InputRef, Span, Spanned, StageSpec, build, plan, solve};
use av_toolbox_runtime::{Engine, Settings};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

/// How many frames the fixture source yields before ending the stream.
const FRAME_COUNT: u64 = 8;

/// How many workers the widened test asks its filter to run.
const WORKERS: usize = 4;

fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    }
}

/// The arguments every fixture here takes, which is none. The registry still needs a type to
/// validate against.
#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct FramesInstance {
    emitted: u64,
}

impl StageInstance for FramesInstance {}

impl Source for FramesInstance {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        if self.emitted >= FRAME_COUNT {
            return Ok(None);
        }

        // Every frame gets its own chunk id, so a filter widened downstream has something to
        // route by. A run that never widens never reads it.
        let meta = FrameMeta::at(self.emitted).in_chunk(self.emitted);
        let frame = context.alloc_frame(&frame_format(), meta);
        self.emitted += 1;

        Ok(Some(frame))
    }
}

/// A source yielding a fixed number of frames, allocated from its own context.
struct Frames;

impl StageDefinition for Frames {
    type Args = NoArgs;

    fn package() -> &'static str {
        "device-test"
    }

    fn identifier() -> &'static str {
        "source"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
        SIGNATURE
    }
}

impl SourceDefinition for Frames {
    fn produces(_capabilities: &Capabilities, _args: &Self::Args) -> Result<OutputFormats, StageError> {
        let format = StreamFormat::Frames(frame_format());

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
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(FramesInstance { emitted: 0 }))
    }
}

struct PassInstance;

impl StageInstance for PassInstance {}

impl Filter for PassInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

/// A filter returning its input unchanged, granular enough to be widened.
struct Pass;

impl StageDefinition for Pass {
    type Args = NoArgs;

    fn package() -> &'static str {
        "device-test"
    }

    fn identifier() -> &'static str {
        "filter"
    }

    // Per frame, so a pipeline that asks for more than one worker can actually get it. A
    // stage left at the default, whole-stream granularity can never widen at all.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("source").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Pass {
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
        let format = chosen.iter().flatten().next().copied();
        let format = format.ok_or_else(|| StageError::msg("no input was bound"))?;

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
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassInstance))
    }
}

/// Every frame index the sink has seen, across every worker.
///
/// A run builds exactly one sink instance, since only a source's own region can widen without
/// this engine refusing it, so nothing here races two instances onto the same run.
static SEEN_INDICES: Mutex<Vec<u64>> = Mutex::new(Vec::new());

struct RecordingInstance;

impl StageInstance for RecordingInstance {}

impl Sink for RecordingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        SEEN_INDICES
            .lock()
            .expect("not poisoned")
            .push(frame.meta().index);

        Ok(())
    }
}

/// A sink recording the index of every frame it was given.
struct Recording;

impl StageDefinition for Recording {
    type Args = NoArgs;

    fn package() -> &'static str {
        "device-test"
    }

    fn identifier() -> &'static str {
        "sink"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl SinkDefinition for Recording {
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
        Ok(Box::new(RecordingInstance))
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<Frames>().expect("should register");
    registry.register_filter::<Pass>().expect("should register");
    registry.register_sink::<Recording>().expect("should register");

    registry
}

fn stage(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        workers: None,
        span: Span::unknown(),
    }
}

fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

/// A source, a filter and a sink, in a straight line, with the filter running one wide.
fn pipeline() -> Vec<StageSpec> {
    let source = stage(StageKind::Source, "source", "device-test/source");

    let mut filter = stage(StageKind::Filter, "filter", "device-test/filter");
    filter.inputs = vec![positional("source")];

    let mut sink = stage(StageKind::Sink, "sink", "device-test/sink");
    sink.inputs = vec![positional("filter")];

    vec![source, filter, sink]
}

/// The same shape, with the filter declared to run [WORKERS] wide.
fn widened_pipeline() -> Vec<StageSpec> {
    let mut specs = pipeline();
    specs[1].workers = Some(WORKERS);

    specs
}

#[test]
fn a_real_graph_runs_on_a_real_device() {
    let registry = registry();
    let specs = pipeline();

    let built = build(&registry, &specs).expect("should build");
    let mut graph = built.into_graph();

    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let formats = solve(&graph, &registry, backend.capabilities()).expect("should solve");
    let made = plan(&graph, &registry).expect("should plan");
    graph.set_execution_plan(made);

    let engine = Engine::build(&graph, &registry, &backend, &formats, Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAME_COUNT);
    assert!(!report.cancelled);
    assert_eq!(report.dropped, 0);
    assert!(report.warnings.is_empty());
}

#[test]
fn a_widened_region_runs_on_a_real_device() {
    let registry = registry();
    let specs = widened_pipeline();

    let built = build(&registry, &specs).expect("should build");
    let mut graph = built.into_graph();

    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let formats = solve(&graph, &registry, backend.capabilities()).expect("should solve");
    let made = plan(&graph, &registry).expect("should plan");

    // A filter can only be widened if its granularity permits it. If that ever regresses, the
    // planner clamps this back to one worker and the run below would still pass, having proven
    // nothing about more than one worker running. Reading the plan's own width catches that
    // before the assertions on its output get a chance to pass by accident.
    let filter_id = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "filter")
        .map(|node| node.id())
        .expect("the graph carries a node named filter");
    let width = made.node(filter_id).width();
    assert_eq!(
        width, WORKERS,
        "the filter did not widen, so this run proves nothing"
    );

    graph.set_execution_plan(made);

    let engine = Engine::build(&graph, &registry, &backend, &formats, Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAME_COUNT);
    assert!(!report.cancelled);
    assert_eq!(report.dropped, 0);
    assert!(report.warnings.is_empty());

    let mut seen = SEEN_INDICES.lock().expect("not poisoned").clone();
    seen.sort_unstable();
    let expected: Vec<u64> = (0..FRAME_COUNT).collect();
    assert_eq!(seen, expected, "every frame should reach the sink exactly once");
}
