#![cfg(all(feature = "gpu-tests", unix))]

use std::io::Write;
use std::os::fd::{IntoRawFd, RawFd};
use std::path::Path;
use std::thread::JoinHandle;
use std::{fs, thread};

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    PipelineRegistry,
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
use av_toolbox_graph::{InputRef, Span, Spanned, StageSpec, build, place_transfers, plan, solve};
use av_toolbox_runtime::{Engine, Settings};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

const FRAME_SIZE: usize = 64;

const FRAME_COUNT: u64 = 5;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct PassInstance;

impl StageInstance for PassInstance {}

impl Filter for PassInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

/// A device filter returning its input unchanged.
struct Pass;

impl StageDefinition for Pass {
    type Args = NoArgs;

    fn package() -> &'static str {
        "end-to-end-test"
    }

    fn identifier() -> &'static str {
        "pass"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("source").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Pass {
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

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    av_toolbox_sources::register(&mut registry).expect("should register the sources");
    av_toolbox_sinks::register(&mut registry).expect("should register the sinks");
    registry.register_filter::<Pass>().expect("should register");

    registry
}

/// A 5-frame, 64 by 64 full range C420 stream at 30000/1001, each frame's luma filled with its index.
fn y4m_stream() -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(30000, 1001);
    let range = y4m::VendorExtensionString::new(b"COLORRANGE=FULL".to_vec()).expect("valid");
    let mut encoder = y4m::encode(FRAME_SIZE, FRAME_SIZE, ratio)
        .append_vendor_extension(range)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma_size = FRAME_SIZE * FRAME_SIZE;
    let chroma_size = (FRAME_SIZE / 2) * (FRAME_SIZE / 2);
    for index in 0..FRAME_COUNT {
        let luma = vec![index as u8; luma_size];
        let u_plane = vec![100u8; chroma_size];
        let v_plane = vec![150u8; chroma_size];
        let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

/// Feeds `bytes` into a new pipe from its own thread, returning the read end's descriptor.
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

fn pipeline(fd: RawFd, output: &Path) -> Vec<StageSpec> {
    let source = stage(StageKind::Source, "source", "y4m/pipe", json!({ "fd": fd }));

    let mut pass = stage(StageKind::Filter, "pass", "end-to-end-test/pass", json!({}));
    pass.inputs = vec![positional("source")];

    let mut sink = stage(StageKind::Sink, "sink", "y4m/pipe", json!({ "path": output }));
    sink.inputs = vec![positional("pass")];

    vec![source, pass, sink]
}

#[test]
fn a_y4m_stream_through_a_device_filter_is_written_back_byte_for_byte() {
    let stream = y4m_stream();
    let (fd, feeder) = piped(stream.clone());
    let folder = tempfile::tempdir().expect("a temp dir");
    let output = folder.path().join("out.y4m");
    let registry = registry();
    let specs = pipeline(fd, &output);

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

    let written = fs::read(&output).expect("reads the output");
    assert_eq!(written, stream);
}
