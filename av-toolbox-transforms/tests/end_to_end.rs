#![cfg(all(feature = "gpu-tests", feature = "scene", unix))]

use std::io::Write;
use std::os::fd::{IntoRawFd, RawFd};
use std::sync::Mutex;
use std::thread;
use std::thread::JoinHandle;

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
    FrameMeta,
    HostFrame,
    HostSide,
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

const SIZE: usize = 64;

/// Two scenes of 30 frames, with a hard cut at frame 30.
const SCENES: &[(u64, u64)] = &[(1, 30), (2, 30)];

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

/// Every `(index, chunk)` the chunk-parallel filter saw.
static FILTERED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Every `(index, chunk)` the sink after the filter saw.
static WHOLE: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Every `(index, chunk)` the sink after `scene/split` saw.
static PIECES: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

fn record(seen: &Mutex<Vec<(u64, u64)>>, meta: &FrameMeta) {
    seen.lock().expect("not poisoned").push((meta.index, meta.chunk));
}

struct ChunkedFilter;

impl StageInstance for ChunkedFilter {}

impl Filter<HostSide> for ChunkedFilter {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<HostFrame, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        record(&FILTERED, frame.meta());

        Ok(frame)
    }
}

/// A host filter, run one chunk at a time, that records every frame it sees.
struct Chunked;

impl StageDefinition for Chunked {
    type Args = NoArgs;

    fn package() -> &'static str {
        "end-to-end-test"
    }

    fn identifier() -> &'static str {
        "chunked"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk
    }
}

impl FilterDefinition for Chunked {
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

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied();
        let format =
            format.ok_or_else(|| StageError::msg("end-to-end-test/chunked needs its input bound"))?;

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
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        Ok(Box::new(ChunkedFilter))
    }
}

/// A host sink recording every frame's index and chunk into a shared static.
struct RecordingInstance(&'static Mutex<Vec<(u64, u64)>>);

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

        record(self.0, frame.meta());

        Ok(())
    }
}

struct Whole;

impl StageDefinition for Whole {
    type Args = NoArgs;

    fn package() -> &'static str {
        "end-to-end-test"
    }

    fn identifier() -> &'static str {
        "whole"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl SinkDefinition for Whole {
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
        Ok(Box::new(RecordingInstance(&WHOLE)))
    }
}

struct Pieces;

impl StageDefinition for Pieces {
    type Args = NoArgs;

    fn package() -> &'static str {
        "end-to-end-test"
    }

    fn identifier() -> &'static str {
        "pieces"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl SinkDefinition for Pieces {
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
        Ok(Box::new(RecordingInstance(&PIECES)))
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    av_toolbox_sources::register(&mut registry).expect("should register the source stages");
    av_toolbox_transforms::register(&mut registry).expect("should register the transform stages");
    registry.register_filter::<Chunked>().expect("should register");
    registry.register_sink::<Whole>().expect("should register");
    registry.register_sink::<Pieces>().expect("should register");

    registry
}

/// A 60-frame, 64 by 64 C420 y4m stream of two 30-frame scenes.
///
/// Luma uses the same 4 by 4 cell noise as the unit tests, drifting one pixel a frame within a
/// scene so a new seed reads as a hard cut. Chroma stays neutral.
fn y4m_stream() -> Vec<u8> {
    let mut bytes = Vec::new();
    let ratio = y4m::Ratio::new(25, 1);
    let mut encoder = y4m::encode(SIZE, SIZE, ratio)
        .write_header(&mut bytes)
        .expect("writes the header");

    let chroma_size = (SIZE / 2) * (SIZE / 2);
    for &(seed, length) in SCENES {
        for offset in 0..length {
            let luma = luma_plane(seed, offset);
            let u_plane = vec![128u8; chroma_size];
            let v_plane = vec![128u8; chroma_size];
            let frame = y4m::Frame::new([&luma, &u_plane, &v_plane], None);
            encoder.write_frame(&frame).expect("writes a frame");
        }
    }

    bytes
}

/// Fills a plane with 4 by 4 cells of noise from `seed`, moved `offset` pixels to the left.
fn luma_plane(seed: u64, offset: u64) -> Vec<u8> {
    let mut plane = vec![0u8; SIZE * SIZE];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let shifted = x as u64 + offset;
            let value = noise(seed, shifted, y as u64) % 256;
            plane[y * SIZE + x] = value as u8;
        }
    }

    plane
}

fn noise(seed: u64, x: u64, y: u64) -> u64 {
    let cell_x = x / 4;
    let cell_y = y / 4;

    let mut hash = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ cell_x.wrapping_mul(0x85EB_CA6B)
        ^ cell_y.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    hash ^= hash >> 33;
    hash = hash.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    hash ^= hash >> 33;

    hash
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

    let mut detect = stage(
        StageKind::Transform,
        "detect",
        "scene/detect",
        json!({ "mode": "natural" }),
    );
    detect.inputs = vec![positional("source")];

    let mut chunked = stage(StageKind::Filter, "chunked", "end-to-end-test/chunked", json!({}));
    chunked.inputs = vec![positional("detect")];
    chunked.workers = Some(2);

    let mut whole = stage(StageKind::Sink, "whole", "end-to-end-test/whole", json!({}));
    whole.inputs = vec![positional("chunked")];

    let mut split = stage(
        StageKind::Transform,
        "split",
        "scene/split",
        json!({ "max_frames": 12 }),
    );
    split.inputs = vec![positional("detect")];

    let mut pieces = stage(StageKind::Sink, "pieces", "end-to-end-test/pieces", json!({}));
    pieces.inputs = vec![positional("split")];

    vec![source, detect, chunked, whole, split, pieces]
}

/// Room for four of the longest scene's frames at once.
fn frame_budget() -> usize {
    const HELD_SCENES: usize = 4;

    let bytes_per_frame = SIZE * SIZE + 2 * (SIZE / 2) * (SIZE / 2);
    let longest_scene = SCENES
        .iter()
        .map(|&(_, length)| length)
        .max()
        .expect("some scenes");

    bytes_per_frame * longest_scene as usize * HELD_SCENES
}

/// Checks chunk ids start at 0, never go down, go up by at most one at a time, and hold no more
/// than `limit` frames.
fn assert_split_chunks(pieces: &[(u64, u64)], limit: usize) {
    let mut length = 0;
    let mut previous = 0;
    for (position, &(_, chunk)) in pieces.iter().enumerate() {
        if position == 0 {
            assert_eq!(chunk, 0, "chunk ids start at 0");
        } else if chunk == previous + 1 {
            length = 0;
        } else {
            assert_eq!(chunk, previous, "chunk ids go up by one at a time");
        }

        length += 1;
        assert!(length <= limit, "chunk {chunk} holds more than {limit} frames");
        previous = chunk;
    }
}

#[test]
fn scene_detect_and_scene_split_run_through_the_engine_on_a_real_device() {
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

    let settings = Settings {
        frame_budget: Some(frame_budget()),
        ..Settings::default()
    };
    let engine =
        Engine::build(&graph, &registry, &backend, &formats, settings).expect("should build the engine");
    let report = engine.run().expect("should run");
    feeder.join().expect("the feeder finishes");

    assert!(!report.cancelled);
    assert_eq!(report.dropped, 0);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    let expected_whole: Vec<(u64, u64)> = (0..60).map(|index| (index, index / 30)).collect();

    let mut filtered = FILTERED.lock().expect("not poisoned").clone();
    filtered.sort_unstable();
    assert_eq!(
        filtered, expected_whole,
        "the filter sees every frame in its scene's chunk"
    );

    let mut whole = WHOLE.lock().expect("not poisoned").clone();
    whole.sort_unstable();
    assert_eq!(whole, expected_whole);

    let pieces = PIECES.lock().expect("not poisoned").clone();
    let indices: Vec<u64> = pieces.iter().map(|(index, _)| *index).collect();
    let expected_indices: Vec<u64> = (0..60).collect();
    assert_eq!(indices, expected_indices, "the split branch keeps stream order");
    assert_split_chunks(&pieces, 12);
    assert_ne!(pieces[29].1, pieces[30].1, "the scene cut stays a chunk boundary");

    let chunks: Vec<u64> = pieces.iter().map(|(_, chunk)| *chunk).collect();
    let expected_chunks: Vec<u64> = [0, 1, 2, 3, 4, 5]
        .into_iter()
        .flat_map(|chunk| std::iter::repeat_n(chunk, 10))
        .collect();
    assert_eq!(
        chunks, expected_chunks,
        "the split branch's chunk ids are deterministic"
    );
}
