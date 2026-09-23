#![cfg(all(feature = "gpu-tests", unix))]

use std::io::Write;
use std::os::fd::{IntoRawFd, RawFd};
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;

use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    PipelineRegistry,
    Sink,
    SinkDefinition,
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
    Device,
    FormatSet,
    FrameContext,
    HostSide,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use av_toolbox_graph::{InputRef, Span, Spanned, StageSpec, build, place_transfers, plan, solve};
use av_toolbox_runtime::{Engine, Settings};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

const FRAME_SIZE: usize = 64;

const FRAME_COUNT: u64 = 5;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

/// Every frame index the sink has seen, in the order it saw them.
static SEEN_INDICES: Mutex<Vec<u64>> = Mutex::new(Vec::new());

struct RecordingInstance;

impl StageInstance for RecordingInstance {}

impl Sink<HostSide> for RecordingInstance {
    fn consume(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<(), StageError> {
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

/// A host sink recording the index of every frame it was given.
struct Recording;

impl StageDefinition for Recording {
    type Args = NoArgs;

    fn package() -> &'static str {
        "end-to-end-test"
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
    type Side = HostSide;

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
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        Ok(Box::new(RecordingInstance))
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    av_toolbox_sources::register(&mut registry).expect("should register the source stages");
    registry.register_sink::<Recording>().expect("should register");

    registry
}

/// A 5-frame, 64 by 64 C420 y4m stream, each frame's luma filled with its index.
fn y4m_stream() -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(25, 1);
    let mut encoder = y4m::encode(FRAME_SIZE, FRAME_SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma_size = FRAME_SIZE * FRAME_SIZE;
    let chroma_size = (FRAME_SIZE / 2) * (FRAME_SIZE / 2);
    for index in 0..FRAME_COUNT {
        let luma = vec![index as u8; luma_size];
        let u_plane = vec![128u8; chroma_size];
        let v_plane = vec![128u8; chroma_size];
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

/// Feeds `bytes` into a new pipe from its own thread, returning the read end's descriptor.
///
/// The write end closes once every byte is written, so the stream can only be read once.
fn piped(bytes: Vec<u8>) -> (RawFd, JoinHandle<()>) {
    let (reader, mut writer) = std::io::pipe().expect("a pipe");
    let feeder = thread::spawn(move || {
        writer.write_all(&bytes).expect("writes the stream");
    });
    let fd = reader.into_raw_fd();

    (fd, feeder)
}

fn stage(kind: StageKind, name: &str, stage_type: &str, args: serde_json::Value) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(args),
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

fn pipeline(fd: RawFd) -> Vec<StageSpec> {
    let source = stage(StageKind::Source, "source", "y4m/pipe", json!({ "fd": fd }));

    let mut sink = stage(StageKind::Sink, "sink", "end-to-end-test/sink", json!({}));
    sink.inputs = vec![positional("source")];

    vec![source, sink]
}

#[test]
fn a_piped_y4m_stream_runs_through_the_engine_on_a_real_device() {
    let stream = y4m_stream();
    let (fd, feeder) = piped(stream);
    let registry = registry();
    let specs = pipeline(fd);

    let built = build(&registry, &specs).expect("should build");
    let mut graph = built.into_graph();

    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let capabilities = backend.capabilities();
    let mut formats = solve(&graph, &registry, capabilities).expect("should solve");
    place_transfers(&mut graph, &registry, &mut formats).expect("should place transfers");
    let made = plan(&graph, &registry).expect("should plan");
    graph.set_execution_plan(made);

    let engine = Engine::build(&graph, &registry, &backend, &formats, Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");
    feeder.join().expect("the feeder finishes");

    assert_eq!(report.frames, FRAME_COUNT);
    assert!(!report.cancelled);
    assert_eq!(report.dropped, 0);
    assert!(report.warnings.is_empty());

    let seen = SEEN_INDICES.lock().expect("not poisoned").clone();
    let expected: Vec<u64> = (0..FRAME_COUNT).collect();
    assert_eq!(seen, expected, "every frame should reach the sink in order");
}
