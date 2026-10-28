#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Backend, SampleFormat, test_device};
use av_toolbox_pipeline::{CompiledPipeline, DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;
use serde_json::json;

/// The size and length of a chain's input clip.
#[derive(Debug, Copy, Clone)]
struct Clip {
    width: usize,
    height: usize,
    frames: u64,
}

/// One frame is enough for a chain whose stages each read a single frame.
const ONE_FRAME: Clip = Clip {
    width: 64,
    height: 64,
    frames: 1,
};

/// Three frames fill QTGMC's and NL4D's temporal windows around the middle frame.
const THREE_FRAMES: Clip = Clip {
    frames: 3,
    ..ONE_FRAME
};

/// A ramp crossed by dark lines with bright rings beside them, plus a little noise.
fn luma_plane(clip: Clip) -> Vec<u8> {
    let mut plane = Vec::with_capacity(clip.width * clip.height);

    for row in 0..clip.height {
        for column in 0..clip.width {
            let ramp = 40 + column * 2;
            let is_line = column % 16 == 8;
            let is_ring = column % 16 == 7 || column % 16 == 9;
            let noise = (column * 7 + row * 13) % 5;
            let value = match (is_line, is_ring) {
                (true, _) => 30 + noise,
                (_, true) => 200 + noise,
                _ => ramp + noise,
            };
            plane.push(value as u8);
        }
    }

    plane
}

/// An 8-bit 4:2:0 y4m stream of `clip`.
fn clip_bytes(clip: Clip) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(clip.width, clip.height, ratio)
        .with_colorspace(y4m::Colorspace::C420)
        .write_header(&mut bytes)
        .expect("writes the header");

    let luma = luma_plane(clip);
    let chroma = vec![128u8; clip.width * clip.height / 4];

    for _ in 0..clip.frames {
        let frame = y4m::Frame::new([&luma, &chroma, &chroma], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
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

/// Checks no `resample/depth` narrows a clip between the stages.
///
/// The graph may widen the 8-bit source into F32 for a stage. The only conversion to integers allowed is the
/// one the graph puts before the sink, and it must read F32.
fn assert_nothing_narrows_between_stages(compiled: &CompiledPipeline<'_, '_>) {
    let graph = compiled.graph();
    let sink_side = json!({ "depth": 10, "at_least": true });
    let widen = json!({ "depth": "f32" });

    let depths: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| node.identifier() == "resample/depth")
        .collect();
    let (widens, narrowing): (Vec<_>, Vec<_>) = depths.into_iter().partition(|node| *node.args() == widen);

    for node in widens {
        let formats = compiled.formats().get(node.id());
        let widen_input = formats.inputs()[0].expect("the widen has an input");
        let name = node.name();
        assert_eq!(
            widen_input.sample(),
            SampleFormat::U8,
            "the widen {name} should only read the 8-bit source"
        );
    }

    assert_eq!(narrowing.len(), 1, "only the sink-side depth narrows");

    let injected = narrowing[0];
    assert_eq!(*injected.args(), sink_side);

    let formats = compiled.formats().get(injected.id());
    let injected_input = formats.inputs()[0].expect("the depth has an input");
    assert_eq!(injected_input.sample(), SampleFormat::F32);
}

/// Runs a chain on `clip` and checks it writes `frames` frames at 10 bits.
///
/// `stages` writes the graph from the input and output paths. The compiled graph must not narrow between its
/// stages.
fn assert_runs_to_ten_bit(stages: impl Fn(&str, &str) -> String, clip: Clip, frames: u64) {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");
    let input_bytes = clip_bytes(clip);
    fs::write(&input, input_bytes).expect("writes the input");

    let input_path = utf8(&input);
    let output_path = utf8(&output);
    let text = stages(input_path, output_path);
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = compiled(&text, &registry, &backend, folder.path());

    assert_nothing_narrows_between_stages(&compiled);

    let engine = compiled
        .engine(Settings::default())
        .expect("should build the engine");
    let report = engine.run().expect("should run");
    assert_eq!(report.frames, frames);

    let stream = fs::read(&output).expect("reads the output");
    let decoder = y4m::decode(stream.as_slice()).expect("reads the header");
    assert!(matches!(decoder.get_colorspace(), y4m::Colorspace::C420p10));
}

fn dehalo_into_deband_text(input: &str, output: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.halo]
type = "dehalo/alpha"
inputs = ["input"]

[filter.deband]
type = "deband/placebo"
inputs = ["halo"]

[sink.output]
type = "y4m/pipe"
inputs = ["deband"]
args.path = "{output}"
"#
    )
}

fn qtgmc_into_artcnn_text(input: &str, output: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.deinterlaced]
type = "deinterlace/qtgmc"
inputs = ["input"]
args.order = "top"

[filter.upscale]
type = "upscale/artcnn"
inputs = ["deinterlaced"]

[sink.output]
type = "y4m/pipe"
inputs = ["upscale"]
args.path = "{output}"
"#
    )
}

fn darken_through_an_edge_mask_text(input: &str, output: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.darkened]
type = "lineart/darken"
inputs = ["input"]

[mask.edges]
type = "edges/sobel"
inputs = ["darkened"]

[filter.blended]
type = "blend/masked"
inputs = ["input", "darkened", "edges"]

[sink.output]
type = "y4m/pipe"
inputs = ["blended"]
args.path = "{output}"
"#
    )
}

fn denoise_dehalo_and_deband_masked_text(input: &str, output: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]
args.preset = "veryfast"
# Denoising chroma too builds a second engine, which adds about a minute.
args.planes = ["y"]

[filter.halo]
type = "dehalo/alpha"
inputs = ["denoise"]

[mask.denoised_detail]
type = "detail/mask"
inputs = ["denoise"]

[filter.dehaloed]
type = "blend/masked"
inputs = ["denoise", "halo", "denoised_detail"]

[filter.deband]
type = "deband/placebo"
inputs = ["dehaloed"]

[mask.dehaloed_detail]
type = "detail/mask"
inputs = ["dehaloed"]

[filter.debanded]
type = "blend/masked"
inputs = ["deband", "dehaloed", "dehaloed_detail"]

[sink.output]
type = "y4m/pipe"
inputs = ["debanded"]
args.path = "{output}"
"#
    )
}

fn resize_into_solid_like_text(input: &str, output: &str) -> String {
    format!(
        r##"
[source.input]
type = "y4m/pipe"
args.path = "{input}"

[filter.resized]
type = "resize/bicubic"
inputs = ["input"]
args.width = 96
args.height = 96

[filter.solid]
type = "solid/like"
inputs = ["resized"]
args.colour = "#404040"

[mask.edges]
type = "edges/sobel"
inputs = ["resized"]

[filter.blended]
type = "blend/masked"
inputs = ["resized", "solid", "edges"]

[sink.output]
type = "y4m/pipe"
inputs = ["blended"]
args.path = "{output}"
"##
    )
}

#[test]
fn dehalo_into_deband() {
    assert_runs_to_ten_bit(dehalo_into_deband_text, ONE_FRAME, 1);
}

#[test]
fn qtgmc_into_artcnn() {
    assert_runs_to_ten_bit(qtgmc_into_artcnn_text, THREE_FRAMES, 6);
}

#[test]
fn darken_through_an_edge_mask() {
    assert_runs_to_ten_bit(darken_through_an_edge_mask_text, ONE_FRAME, 1);
}

#[test]
fn denoise_dehalo_and_deband_masked() {
    assert_runs_to_ten_bit(denoise_dehalo_and_deband_masked_text, THREE_FRAMES, 3);
}

#[test]
fn resize_into_solid_like() {
    assert_runs_to_ten_bit(resize_into_solid_like_text, ONE_FRAME, 1);
}
