#![cfg(all(feature = "gpu-tests", unix))]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, PoisonError};

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
    FrameFormat,
    FrameLayout,
    HostSide,
    Offered,
    Primaries,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    Transfer,
};
use av_toolbox_graph::{ArgSpans, InputRef, Span, Spanned, StageSpec, build, place_transfers, plan, solve};
use av_toolbox_runtime::{Engine, RunReport, Settings};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

const FRAME_SIZE: usize = 64;

const FRAME_COUNT: u64 = 5;

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

/// A frame the sink was given.
#[derive(Debug, Clone, PartialEq)]
struct Recorded {
    index: u64,
    format: FrameFormat,
    planes: Vec<Vec<u8>>,
}

static RECORDED: Mutex<Vec<Recorded>> = Mutex::new(Vec::new());

/// Held by each test for its whole run, since every test records into the same list.
static RUNNING: Mutex<()> = Mutex::new(());

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

        let format = *frame.format();
        let planes = format
            .channels()
            .iter()
            .map(|channel| {
                let plane = frame.plane(*channel).expect("the frame carries its channels");
                plane.bytes().to_vec()
            })
            .collect();
        let recorded = Recorded {
            index: frame.meta().index,
            format,
            planes,
        };

        RECORDED.lock().expect("not poisoned").push(recorded);

        Ok(())
    }
}

/// A host sink recording every frame it was given.
struct Recording;

impl StageDefinition for Recording {
    type Args = NoArgs;

    fn package() -> &'static str {
        "constant-test"
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

fn stage(kind: StageKind, name: &str, stage_type: &str, args: serde_json::Value) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(args),
        arg_spans: ArgSpans::default(),
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

fn sink_of(target: &str) -> StageSpec {
    let mut sink = stage(StageKind::Sink, "sink", "constant-test/sink", json!({}));
    sink.inputs = vec![positional(target)];

    sink
}

/// Compiles and runs `specs` on the default device, or hands back the first failure rendered for a reader.
fn run(specs: &[StageSpec]) -> Result<RunReport, String> {
    let registry = registry();
    let built = build(&registry, specs).map_err(|error| format!("{error:?}"))?;
    let mut graph = built.into_graph();

    let backend = Backend::open_default(Device::Default).expect("should open a device");
    let capabilities = backend.capabilities();
    let mut formats = solve(&graph, &registry, capabilities).map_err(|error| format!("{error:?}"))?;
    place_transfers(&mut graph, &registry, &mut formats).map_err(|error| format!("{error:?}"))?;
    let made = plan(&graph, &registry).map_err(|error| format!("{error:?}"))?;
    graph.set_execution_plan(made);

    let engine = Engine::build(&graph, &registry, &backend, &formats, Settings::default())
        .map_err(|error| format!("{error:?}"))?;

    engine.run().map_err(|error| format!("{error:?}"))
}

/// Waits for any other test to finish, then clears what it recorded.
fn alone() -> MutexGuard<'static, ()> {
    let guard = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
    RECORDED.lock().expect("not poisoned").clear();

    guard
}

fn recorded() -> Vec<Recorded> {
    RECORDED.lock().expect("not poisoned").clone()
}

/// Hands back every recorded frame, leaving the list empty for the next run.
fn take_recorded() -> Vec<Recorded> {
    let mut recorded = RECORDED.lock().expect("not poisoned");

    std::mem::take(&mut *recorded)
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

/// Writes the y4m stream into `folder`, returning its path.
///
/// The reference is read from a file rather than a descriptor, since a claimed descriptor is never given back
/// and every test here shares one process.
fn reference_file(folder: &Path) -> PathBuf {
    let path = folder.join("reference.y4m");
    let bytes = y4m_stream();
    std::fs::write(&path, bytes).expect("writes the stream");

    path
}

fn reference_stage(reference_path: &Path) -> StageSpec {
    let reference_args = json!({ "path": reference_path });

    stage(StageKind::Source, "reference", "y4m/pipe", reference_args)
}

/// The format of the reference's frames, found by running it straight into the recording sink.
fn format_of_reference(reference_path: &Path) -> FrameFormat {
    let reference = reference_stage(reference_path);
    let specs = [reference, sink_of("reference")];
    run(&specs).expect("the reference runs alone");

    let frames = take_recorded();

    frames[0].format
}

/// A y4m reference, the `like` stage following it, and the recording sink.
fn following(stage_type: &str, args: serde_json::Value, reference_path: &Path) -> Vec<StageSpec> {
    let reference = reference_stage(reference_path);
    let mut like = stage(StageKind::Filter, "like", stage_type, args);
    like.inputs = vec![positional("reference")];

    vec![reference, like, sink_of("like")]
}

#[test]
fn solid_fill_produces_its_colour_for_every_frame() {
    let _alone = alone();

    let args = json!({ "width": 4, "height": 2, "frames": 3, "colour": "#ff0000" });
    let source = stage(StageKind::Source, "source", "solid/fill", args);
    let specs = [source, sink_of("source")];

    let report = run(&specs).expect("should run");

    let seen = recorded();
    let indices: Vec<u64> = seen.iter().map(|frame| frame.index).collect();
    assert_eq!(report.frames, 3);
    assert_eq!(indices, vec![0, 1, 2]);
    for frame in seen {
        assert_eq!(frame.planes, vec![vec![63; 8], vec![102; 2], vec![240; 2]]);
    }
}

#[test]
fn image_file_shows_its_image_for_every_frame() {
    let _alone = alone();

    let folder = tempfile::tempdir().expect("a folder");
    let path = folder.path().join("orange.png");
    let orange = image::RgbImage::from_pixel(4, 2, image::Rgb([255, 128, 0]));
    orange.save(&path).expect("writes the image");

    let args = json!({ "path": path, "frames": 2 });
    let source = stage(StageKind::Source, "source", "image/file", args);
    let specs = [source, sink_of("source")];

    let report = run(&specs).expect("should run");

    let seen = recorded();
    assert_eq!(report.frames, 2);
    assert_eq!(seen[0].format.width, 4);
    assert_eq!(seen[0].format.layout, FrameLayout::Rgb);
    assert_eq!(seen[0].format.colour.primaries, Primaries::Bt709);
    assert_eq!(seen[0].format.colour.transfer, Transfer::Srgb);
    assert_eq!(seen[1].planes, vec![vec![255; 8], vec![128; 8], vec![0; 8]]);
}

#[test]
fn solid_fill_refuses_zero_frames() {
    let _alone = alone();

    let args = json!({ "width": 4, "height": 2, "frames": 0, "colour": "#ff0000" });
    let source = stage(StageKind::Source, "source", "solid/fill", args);
    let specs = [source, sink_of("source")];

    let error = run(&specs).expect_err("zero frames is refused");

    assert!(error.contains("frames"), "{error}");
}

#[test]
fn solid_like_follows_its_reference() {
    let _alone = alone();

    let folder = tempfile::tempdir().expect("a folder");
    let reference_path = reference_file(folder.path());
    let reference_format = format_of_reference(&reference_path);
    let specs = following("solid/like", json!({ "colour": "#000000" }), &reference_path);

    let report = run(&specs).expect("should run");

    let seen = recorded();
    let indices: Vec<u64> = seen.iter().map(|frame| frame.index).collect();
    let luma_size = FRAME_SIZE * FRAME_SIZE;
    let chroma_size = luma_size / 4;
    assert_eq!(report.frames, FRAME_COUNT);
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    assert_eq!(seen[0].format.width, FRAME_SIZE as u32);
    for frame in seen {
        assert_eq!(frame.format, reference_format);
        assert_eq!(
            frame.planes,
            vec![
                vec![16; luma_size],
                vec![128; chroma_size],
                vec![128; chroma_size]
            ]
        );
    }
}

#[test]
fn image_like_follows_its_reference() {
    let _alone = alone();

    let folder = tempfile::tempdir().expect("a folder");
    let path = folder.path().join("white.png");
    let size = FRAME_SIZE as u32;
    let white = image::GrayImage::from_pixel(size, size, image::Luma([255]));
    white.save(&path).expect("writes the image");

    let reference_path = reference_file(folder.path());
    let reference_format = format_of_reference(&reference_path);
    let specs = following("image/like", json!({ "path": path }), &reference_path);

    let report = run(&specs).expect("should run");

    let seen = recorded();
    let indices: Vec<u64> = seen.iter().map(|frame| frame.index).collect();
    let luma_size = FRAME_SIZE * FRAME_SIZE;
    assert_eq!(report.frames, FRAME_COUNT);
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    assert_eq!(seen[4].index, 4);
    assert_eq!(seen[4].planes[0], vec![235; luma_size]);
    for frame in seen {
        assert_eq!(frame.format, reference_format);
    }
}

#[test]
fn image_like_refuses_an_image_of_another_size() {
    let _alone = alone();

    let folder = tempfile::tempdir().expect("a folder");
    let path = folder.path().join("small.png");
    let small = image::GrayImage::from_pixel(8, 8, image::Luma([255]));
    small.save(&path).expect("writes the image");

    let reference_path = reference_file(folder.path());
    let specs = following("image/like", json!({ "path": path }), &reference_path);

    let error = run(&specs).expect_err("8x8 is not 64x64");

    assert!(error.contains("resize/"), "{error}");
    assert!(error.contains("8x8"), "{error}");
}
