#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::io::Write;
use std::os::fd::{IntoRawFd, RawFd};
use std::thread::JoinHandle;
use std::{fs, thread};

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    StageDefinition,
    StageError,
    StageInstance,
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
use av_toolbox_graph::NodeRole;
use av_toolbox_pipeline::Pipeline;
use av_toolbox_runtime::Settings;
use schemars::JsonSchema;
use serde::Deserialize;

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
        "pipeline-test"
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

#[test]
fn a_toml_pipeline_compiles_and_runs_through_a_device_filter() {
    let stream = y4m_stream();
    let (fd, feeder) = piped(stream.clone());
    let folder = tempfile::tempdir().expect("a temp dir");
    let output = folder.path().join("output.y4m");
    let output_path = output.to_str().expect("a utf-8 path");

    let text = format!(
        r#"
[source.input]
type = "y4m/pipe"
args.fd = {fd}

[filter.pass]
type = "pipeline-test/pass"
inputs = ["input"]

[sink.output]
type = "y4m/pipe"
inputs = ["pass"]
args.path = "{output_path}"
"#
    );

    let mut registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    registry.register_filter::<Pass>().expect("should register");

    let pipeline_directory = tempfile::tempdir().expect("a temp dir");
    let pipeline_path = pipeline_directory.path().join("pipeline.toml");
    fs::write(&pipeline_path, &text).expect("writes the pipeline");
    let supplied = av_toolbox_pipeline::Supplied::new();
    let loaded = av_toolbox_pipeline::load(&pipeline_path, &av_toolbox_pipeline::DiskFiles, &supplied)
        .expect("the file should load");
    let specs = loaded.specs;

    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");

    let mut crossings = 0;
    for node in compiled.graph().nodes() {
        let role = node.role();
        if let NodeRole::Crossing(_) = role {
            crossings += 1;
            assert!(compiled.stage_of(node.id()).is_none());
        }
    }
    assert!(crossings > 0);

    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");
    feeder.join().expect("the feeder finishes");

    assert_eq!(report.frames, FRAME_COUNT);

    let written = fs::read(&output).expect("reads the output");
    assert_eq!(written, stream);
}
