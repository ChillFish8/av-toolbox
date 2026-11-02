#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

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
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
    test_device,
};
use av_toolbox_graph::{NodeRole, StageSpec};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;
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
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
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

    fn description() -> &'static str {
        "A stage used by tests."
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

    fn build(
        _context: &BuildContext,
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

/// Loads the pipeline `text` from a file, as the CLI would.
fn loaded_specs(text: &str) -> Vec<StageSpec> {
    let pipeline_directory = tempfile::tempdir().expect("a temp dir");
    let pipeline_path = pipeline_directory.path().join("pipeline.toml");
    fs::write(&pipeline_path, text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_path, &DiskFiles, &supplied).expect("the file should load");

    loaded.specs
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

[filter.eight]
type = "resample/depth"
inputs = ["pass"]
args.depth = 8

[sink.output]
type = "y4m/pipe"
inputs = ["eight"]
args.path = "{output_path}"
"#
    );

    let mut registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    registry.register_filter::<Pass>().expect("should register");

    let specs = loaded_specs(&text);
    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
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

#[test]
fn an_eight_bit_y4m_pipe_comes_out_at_ten_bits() {
    let (fd, feeder) = piped(y4m_stream());
    let folder = tempfile::tempdir().expect("a temp dir");
    let output = folder.path().join("output.y4m");
    let output_path = output.to_str().expect("a utf-8 path");

    let text = format!(
        r#"
[source.input]
type = "y4m/pipe"
args.fd = {fd}

[sink.output]
type = "y4m/pipe"
inputs = ["input"]
args.path = "{output_path}"
"#
    );

    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let specs = loaded_specs(&text);
    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");

    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");
    feeder.join().expect("the feeder finishes");

    assert_eq!(report.frames, FRAME_COUNT);

    let written = fs::read(&output).expect("reads the output");
    let header_end = written
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("the output has a header line");
    let header = String::from_utf8_lossy(&written[..header_end]);
    assert!(header.contains(" C420p10"), "{header}");
}

/// The first line of the y4m file at `path`.
fn y4m_header(path: &Path) -> String {
    let written = fs::read(path).expect("reads the output");
    let header_end = written
        .iter()
        .position(|byte| *byte == b'\n')
        .expect("the output has a header line");

    String::from_utf8_lossy(&written[..header_end]).into_owned()
}

#[test]
fn a_nine_bit_clip_widens_to_ten_bits_before_a_y4m_sink() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let output = folder.path().join("output.y4m");
    let output_path = output.to_str().expect("a utf-8 path");

    let text = format!(
        r##"
[source.input]
type = "solid/fill"
args.colour = "#808080"
args.width = 64
args.height = 64
args.frames = {FRAME_COUNT}
args.frame_rate = [24, 1]

[filter.nine]
type = "resample/depth"
inputs = ["input"]
args.depth = 9

[sink.output]
type = "y4m/pipe"
inputs = ["nine"]
args.path = "{output_path}"
"##
    );

    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let specs = loaded_specs(&text);
    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");

    let graph = compiled.graph();
    let depths: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .collect();
    assert_eq!(depths.len(), 2);

    let widen = depths
        .iter()
        .find(|node| node.name() != "nine")
        .expect("an injected widen");
    assert_eq!(*widen.args(), json!({ "depth": 10 }));

    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAME_COUNT);

    let header = y4m_header(&output);
    assert!(header.contains(" C420p10"), "{header}");
}

#[test]
fn the_depth_args_the_graph_writes_run_against_the_real_stage() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let output = folder.path().join("output.y4m");
    let output_path = output.to_str().expect("a utf-8 path");

    let text = format!(
        r##"
[source.input]
type = "solid/fill"
args.colour = "#808080"
args.width = 64
args.height = 64
args.frames = {FRAME_COUNT}
args.frame_rate = [24, 1]

[filter.float]
type = "resample/depth"
inputs = ["input"]
args.depth = "f32"

[filter.ten]
type = "resample/depth"
inputs = ["float"]
args.depth = 10
args.at_least = true

[sink.output]
type = "y4m/pipe"
inputs = ["ten"]
args.path = "{output_path}"
"##
    );

    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let specs = loaded_specs(&text);
    let pipeline = Pipeline::build(&registry, &specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");

    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAME_COUNT);

    let header = y4m_header(&output);
    assert!(header.contains(" C420p10"), "{header}");
}
