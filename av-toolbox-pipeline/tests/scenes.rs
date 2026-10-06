#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::fs;
use std::path::Path;

use av_toolbox_gpu::{Backend, test_device};
use av_toolbox_pipeline::{DiskFiles, Pipeline, Supplied};
use av_toolbox_runtime::Settings;

const SIZE: usize = 64;

const SCENE_FRAMES: usize = 8;

const FLAT_U: u8 = 90;

const FLAT_V: u8 = 180;

/// How far noise moves a sample either way.
const NOISE_SPREAD: i32 = 12;

/// The planes of one decoded frame, one after another.
type FrameBytes = Vec<u8>;

/// What one pipeline run wrote.
struct RunOutput {
    frames: Vec<FrameBytes>,
    log: Option<String>,
}

/// A horizontal ramp from black to near white.
fn gradient_luma() -> Vec<u8> {
    let row: Vec<u8> = (0..SIZE).map(|column| column as u8 * 4).collect();
    row.repeat(SIZE)
}

/// The ramp of `gradient_luma` running from white to near black.
fn inverse_gradient_luma() -> Vec<u8> {
    let gradient = gradient_luma();
    gradient.iter().map(|sample| 255 - sample).collect()
}

/// `clean` with deterministic noise of up to `NOISE_SPREAD` either way, drawn afresh for each `frame`.
fn noisy(clean: &[u8], frame: usize) -> Vec<u8> {
    let span = 2 * NOISE_SPREAD as u32 + 1;

    clean
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let seed = (index as u32).wrapping_mul(2_654_435_761) ^ (frame as u32).wrapping_mul(97);
            let noise = (seed % span) as i32 - NOISE_SPREAD;
            let noisy = i32::from(*sample) + noise;
            noisy.clamp(0, 255) as u8
        })
        .collect()
}

/// `frames` noisy copies of `clean`, numbered from `first_frame` so each one draws different noise.
fn noisy_scene(clean: &[u8], first_frame: usize, frames: usize) -> Vec<Vec<u8>> {
    (first_frame..first_frame + frames)
        .map(|frame| noisy(clean, frame))
        .collect()
}

/// The lumas of a noisy gradient scene followed by a noisy inverse gradient scene.
fn two_scene_lumas() -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let gradient = gradient_luma();
    let inverse = inverse_gradient_luma();
    let opening = noisy_scene(&gradient, 0, SCENE_FRAMES);
    let closing = noisy_scene(&inverse, SCENE_FRAMES, SCENE_FRAMES);

    (opening, closing)
}

/// A C420 y4m clip with one frame per entry of `lumas`, each with flat chroma.
fn clip_with_lumas(lumas: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(24, 1);
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let chroma_samples = (SIZE / 2) * (SIZE / 2);
    let u_plane = vec![FLAT_U; chroma_samples];
    let v_plane = vec![FLAT_V; chroma_samples];
    for luma in lumas {
        let frame = y4m::Frame::new([luma, &u_plane, &v_plane], None);
        encoder.write_frame(&frame).expect("writes a frame");
    }

    bytes
}

/// Decodes a C420 y4m stream into the bytes of each frame.
fn decode(stream: &[u8]) -> Vec<FrameBytes> {
    let mut decoder = y4m::decode(stream).expect("reads the header");
    assert!(matches!(decoder.get_colorspace(), y4m::Colorspace::C420));

    let mut frames = Vec::new();
    while let Ok(frame) = decoder.read_frame() {
        let mut bytes = frame.get_y_plane().to_vec();
        bytes.extend_from_slice(frame.get_u_plane());
        bytes.extend_from_slice(frame.get_v_plane());
        frames.push(bytes);
    }

    frames
}

fn utf8(path: &Path) -> &str {
    path.to_str().expect("a utf-8 path")
}

/// A pipeline reading `{input}`, running `filter` as the stage `denoise`, and writing it to `{output}`.
fn denoise_pipeline(filter: &str) -> String {
    format!(
        r#"
[source.input]
type = "y4m/pipe"
args.path = "{{input}}"

[filter.denoise]
inputs = ["input"]
{filter}

[sink.output]
type = "y4m/pipe"
inputs = ["denoise"]
args.path = "{{output}}"
"#
    )
}

/// `denoise_pipeline` plus a sink logging the denoised frames' chunks in the layer `breaks` asks for.
fn logged_denoise_pipeline(filter: &str, breaks: &str) -> String {
    let pipeline = denoise_pipeline(filter);
    let log = format!(
        r#"
[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "{{log}}"
args.scenes = {{ breaks = {breaks} }}
"#
    );

    pipeline + &log
}

/// Writes `clip` and the pipeline `template` to a temp dir, runs it, and reads back what it wrote.
///
/// The template's `{input}`, `{output}` and `{log}` are replaced with paths in the temp dir. Every sink is
/// expected to see each written frame once.
fn run_pipeline(template: &str, clip: &[u8]) -> RunOutput {
    let folder = tempfile::tempdir().expect("a temp dir");
    let input = folder.path().join("input.y4m");
    let output = folder.path().join("output.y4m");
    let log = folder.path().join("log.ndjson");
    let pipeline_path = folder.path().join("pipeline.toml");

    fs::write(&input, clip).expect("writes the input");

    let pipeline_text = template
        .replace("{input}", utf8(&input))
        .replace("{output}", utf8(&output))
        .replace("{log}", utf8(&log));
    fs::write(&pipeline_path, pipeline_text).expect("writes the pipeline");

    let supplied = Supplied::new();
    let loaded =
        av_toolbox_pipeline::load(&pipeline_path, &DiskFiles, &supplied).expect("the pipeline loads");
    let registry = av_toolbox_pipeline::registry().expect("the built-in stages register");
    let pipeline = Pipeline::build(&registry, &loaded.specs).expect("should build");
    let backend = Backend::open_default(test_device()).expect("should open a device");
    let compiled = pipeline.compile(&backend).expect("should compile");
    let settings = Settings::default();
    let engine = compiled.engine(settings).expect("should build the engine");
    let report = engine.run().expect("should run");

    let written = fs::read(&output).expect("reads the output");
    let frames = decode(&written);
    let log = fs::read_to_string(&log).ok();

    let sinks = if log.is_some() { 2 } else { 1 };
    assert_eq!(report.frames, sinks * frames.len() as u64);

    RunOutput { frames, log }
}

/// `denoise_pipeline` plus a sink logging the denoised frames with no scenes asked for.
fn unscened_denoise_pipeline(filter: &str) -> String {
    let pipeline = denoise_pipeline(filter);
    let log = r#"
[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "{log}"
"#;

    pipeline + log
}

/// The `scene_score` of every line of an ndjson log, in order.
fn logged_scores(log: &str) -> Vec<serde_json::Value> {
    log.lines()
        .map(|line| {
            let mut value: serde_json::Value = serde_json::from_str(line).expect("a json line");
            value["scene_score"].take()
        })
        .collect()
}

/// The `chunk` of every line of an ndjson log, in order.
fn logged_chunks(log: &str) -> Vec<u64> {
    log.lines()
        .map(|line| {
            let value: serde_json::Value = serde_json::from_str(line).expect("a json line");
            value["chunk"].as_u64().expect("a chunk on every line")
        })
        .collect()
}

/// How many frames each run of equal neighbouring chunks spans.
fn chunk_lengths(chunks: &[u64]) -> Vec<usize> {
    chunks
        .chunk_by(|left, right| left == right)
        .map(<[u64]>::len)
        .collect()
}

/// Runs `filter` over each scene as its own clip and joins the outputs.
fn denoise_each_scene(filter: &str, scenes: &[&[Vec<u8>]]) -> Vec<FrameBytes> {
    let pipeline = denoise_pipeline(filter);

    let mut frames = Vec::new();
    for lumas in scenes {
        let clip = clip_with_lumas(lumas);
        let output = run_pipeline(&pipeline, &clip);
        frames.extend(output.frames);
    }

    frames
}

#[test]
fn a_cut_restarts_av_denoise() {
    let filter = r#"type = "av-denoise/nlmeans""#;
    let (opening, closing) = two_scene_lumas();
    let lumas = [opening.clone(), closing.clone()].concat();
    let clip = clip_with_lumas(&lumas);

    let pipeline = logged_denoise_pipeline(filter, r#"{ mode = "natural" }"#);
    let together = run_pipeline(&pipeline, &clip);
    let log = together.log.expect("the log is written");
    let chunks = logged_chunks(&log);

    assert_eq!(chunk_lengths(&chunks), [SCENE_FRAMES, SCENE_FRAMES], "{chunks:?}");

    let input_frames = decode(&clip);
    assert_ne!(together.frames, input_frames, "the denoise changed nothing");

    let alone = denoise_each_scene(filter, &[&opening, &closing]);

    assert_eq!(together.frames.len(), 2 * SCENE_FRAMES);
    assert!(
        together.frames == alone,
        "the two-scene run differs from the per-scene runs"
    );
}

#[test]
fn nl4d_at_two_workers_matches_one() {
    let (opening, closing) = two_scene_lumas();
    let lumas = [opening, closing].concat();
    let clip = clip_with_lumas(&lumas);

    let two_workers = denoise_pipeline("type = \"av-denoise/nl4d\"\nworkers = 2");
    let one_worker = denoise_pipeline("type = \"av-denoise/nl4d\"\nworkers = 1");
    let parallel = run_pipeline(&two_workers, &clip);
    let serial = run_pipeline(&one_worker, &clip);

    assert_eq!(parallel.frames.len(), 2 * SCENE_FRAMES);
    assert!(parallel.frames == serial.frames, "two workers differ from one");
}

#[test]
fn two_layers_on_one_path_restart_independently() {
    let filter = r#"type = "av-denoise/nlmeans""#;
    let (opening, closing) = two_scene_lumas();
    let lumas = [opening.clone(), closing.clone()].concat();
    let clip = clip_with_lumas(&lumas);

    let pipeline = logged_denoise_pipeline(filter, r#"{ mode = "limited", max_frames = 4 }"#);
    let together = run_pipeline(&pipeline, &clip);
    let log = together.log.expect("the log is written");
    let chunks = logged_chunks(&log);
    let lengths = chunk_lengths(&chunks);

    assert_eq!(chunks.len(), 2 * SCENE_FRAMES);
    assert!(lengths.iter().all(|length| *length <= 4), "{chunks:?}");
    assert_ne!(chunks[SCENE_FRAMES - 1], chunks[SCENE_FRAMES], "{chunks:?}");

    let alone = denoise_each_scene(filter, &[&opening, &closing]);
    assert!(
        together.frames == alone,
        "the two-layer run differs from the per-scene runs"
    );
}

#[test]
fn a_clip_without_cuts_is_one_chunk() {
    let gradient = gradient_luma();
    let lumas = noisy_scene(&gradient, 0, 12);
    let clip = clip_with_lumas(&lumas);

    let two_workers = logged_denoise_pipeline(
        "type = \"av-denoise/nl4d\"\nworkers = 2",
        r#"{ mode = "natural" }"#,
    );
    let one_worker = denoise_pipeline("type = \"av-denoise/nl4d\"\nworkers = 1");
    let parallel = run_pipeline(&two_workers, &clip);
    let serial = run_pipeline(&one_worker, &clip);
    let log = parallel.log.expect("the log is written");
    let chunks = logged_chunks(&log);

    assert_eq!(chunk_lengths(&chunks), [12], "{chunks:?}");
    assert_eq!(parallel.frames.len(), 12);
    assert!(parallel.frames == serial.frames, "two workers differ from one");
}

#[test]
fn the_metadata_sink_writes_scores_only_when_it_demands_scenes() {
    let filter = r#"type = "av-denoise/nlmeans""#;
    let (opening, closing) = two_scene_lumas();
    let lumas = [opening, closing].concat();
    let clip = clip_with_lumas(&lumas);

    let demanding = logged_denoise_pipeline(filter, r#"{ mode = "natural" }"#);
    let demanded = run_pipeline(&demanding, &clip);
    let demanded_log = demanded.log.expect("the log is written");
    let demanded_scores = logged_scores(&demanded_log);

    let unscened = unscened_denoise_pipeline(filter);
    let undemanded = run_pipeline(&unscened, &clip);
    let undemanded_log = undemanded.log.expect("the log is written");
    let undemanded_scores = logged_scores(&undemanded_log);

    assert!(
        demanded_scores.iter().any(serde_json::Value::is_number),
        "{demanded_scores:?}"
    );
    assert_eq!(undemanded_scores.len(), 2 * SCENE_FRAMES);
    assert!(
        undemanded_scores.iter().all(serde_json::Value::is_null),
        "{undemanded_scores:?}"
    );
}
