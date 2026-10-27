#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, SampleFormat, test_device};
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const WIDTH: usize = 64;

const HEIGHT: usize = 48;

const FRAMES: u64 = 2;

const BETWEEN_STAGES: &str = r#"
[filter.between]
type = "resample/depth"
inputs = ["halo"]
args.depth = 8
args.dither = "none"
"#;

/// The limited range 8-bit luma of a clip with bright rings beside dark lines, so both stages change it.
fn luma_plane() -> Vec<u8> {
    let mut plane = Vec::with_capacity(WIDTH * HEIGHT);

    for row in 0..HEIGHT {
        for column in 0..WIDTH {
            let ramp = 40 + column * 2;
            let is_line = column % 16 == 8;
            let is_ring = column % 16 == 7 || column % 16 == 9;
            let value = match (is_line, is_ring) {
                (true, _) => 30 + row % 5,
                (_, true) => 200 + row % 7,
                _ => ramp,
            };
            plane.push(value as u8);
        }
    }

    plane
}

fn clip() -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(WIDTH, HEIGHT, ratio)
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma = luma_plane();
    let chroma = vec![128u8; WIDTH * HEIGHT / 4];

    for _ in 0..FRAMES {
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

/// How a graph's `dehalo/alpha` and `lineart/darken` stages are rounded.
#[derive(Debug, Copy, Clone)]
enum Rounding {
    /// Nothing is added, so the graph injects its own conversion before the sink.
    Injected,
    /// An 8-bit conversion sits before the sink only.
    EightBitSink,
    /// An 8-bit conversion sits between the stages and before the sink.
    EightBitEverywhere,
}

fn pipeline_text(input: &Path, output: &Path, rounding: Rounding) -> String {
    let second_input = match rounding {
        Rounding::EightBitEverywhere => "between",
        _ => "halo",
    };
    let between = match rounding {
        Rounding::EightBitEverywhere => BETWEEN_STAGES,
        _ => "",
    };
    let sink_input = match rounding {
        Rounding::Injected => "lines",
        _ => "sunk",
    };
    let sink_depth = match rounding {
        Rounding::Injected => "",
        _ => EIGHT_BIT_SINK,
    };
    let input = utf8(input);
    let output = utf8(output);

    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.halo]
type = "dehalo/alpha"
inputs = ["input"]
{between}
[filter.lines]
type = "lineart/darken"
inputs = ["{second_input}"]
{sink_depth}
[sink.output]
type = "y4m/pipe"
inputs = ["{sink_input}"]
args.path = "{output}"
"#
    )
}

fn compiled<'r, 'b>(
    text: &str,
    registry: &'r PipelineRegistry,
    backend: &'b Backend,
    folder: &Path,
) -> CompiledPipeline<'r, 'b> {
    let path = folder.join("pipeline.toml");
    fs::write(&path, text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded = av_toolbox_pipeline::load(&path, &DiskFiles, &supplied).expect("the file should load");
    let pipeline = Pipeline::build(registry, &loaded.specs).expect("should build");

    pipeline.compile(backend).expect("should compile")
}

fn run(compiled: CompiledPipeline<'_, '_>) {
    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");

    assert_eq!(report.frames, FRAMES);
}

const EIGHT_BIT_SINK: &str = r#"
[filter.sunk]
type = "resample/depth"
inputs = ["lines"]
args.depth = 8
args.dither = "none"
"#;

/// The luma codes of every frame in an 8-bit y4m stream.
fn luma_codes(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    let mut frames = Vec::new();

    while let Ok(frame) = decoder.read_frame() {
        frames.push(frame.get_y_plane().to_vec());
    }

    frames
}

/// Runs the graph written by `text` for `folder`'s clip and returns the output stream.
fn run_text(text: &str, folder: &Path, output: &Path) -> Vec<u8> {
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = compiled(text, &registry, &backend, folder);
    run(compiled);

    fs::read(output).expect("reads the output")
}

#[test]
fn an_eight_bit_clip_through_two_float_stages_comes_out_ten_bit() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");
    fs::write(&input, clip()).expect("writes the input");

    let text = pipeline_text(&input, &output, Rounding::Injected);
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = compiled(&text, &registry, &backend, folder.path());

    let graph = compiled.graph();
    let depths: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .collect();
    assert_eq!(depths.len(), 1, "only the one before the sink");

    let injected = depths[0];
    let formats = compiled.formats().get(injected.id());
    let injected_input = formats.inputs()[0].expect("the depth has an input");
    assert_eq!(injected_input.sample(), SampleFormat::F32);

    let lines = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "lines")
        .expect("the second stage");
    assert_eq!(injected.inputs()[0].source(), lines.id());

    let unload = graph
        .nodes()
        .iter()
        .find(|node| node.inputs().iter().any(|edge| edge.source() == injected.id()))
        .expect("the node after the depth");
    assert_eq!(unload.identifier(), "builtin/unload");

    run(compiled);

    let stream = fs::read(&output).expect("reads the output");
    let decoder = y4m::decode(stream.as_slice()).expect("reads the header");
    assert!(matches!(decoder.get_colorspace(), y4m::Colorspace::C420p10));
}

#[test]
fn an_explicit_eight_bit_output_matches_the_single_stage_rounding() {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let float_output = folder.path().join("float.y4m");
    let rounded_output = folder.path().join("rounded.y4m");
    fs::write(&input, clip()).expect("writes the input");

    let float_text = pipeline_text(&input, &float_output, Rounding::EightBitSink);
    let float_stream = run_text(&float_text, folder.path(), &float_output);
    let float_frames = luma_codes(&float_stream);

    let rounded_text = pipeline_text(&input, &rounded_output, Rounding::EightBitEverywhere);
    let rounded_stream = run_text(&rounded_text, folder.path(), &rounded_output);
    let rounded_frames = luma_codes(&rounded_stream);

    assert_eq!(float_frames.len(), FRAMES as usize);
    assert_eq!(rounded_frames.len(), FRAMES as usize);

    let mut largest = 0u8;
    for (float_frame, rounded_frame) in float_frames.iter().zip(&rounded_frames) {
        for (float_code, rounded_code) in float_frame.iter().zip(rounded_frame) {
            largest = largest.max(float_code.abs_diff(*rounded_code));
        }
    }

    assert!(largest <= 1, "{largest}");
}
