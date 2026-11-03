#![cfg(all(unix, feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]

use std::path::Path;
use std::sync::mpsc::{self, SyncSender};
use std::thread;
use std::time::{Duration, Instant};

use av_toolbox::{
    FrameFormat,
    InputFrame,
    NodeRole,
    OutputFrame,
    PipelineInput,
    RunReport,
    Session,
    SessionError,
    Settings,
    StreamHandle,
};
use av_toolbox_gpu::{Backend, ColourInfo, FrameLayout, FrameRate, SampleFormat, Subsampling, test_device};

const SIZE: u32 = 64;

const SCENE_FRAMES: usize = 8;

/// How long a stop may take before a test calls it hung.
const STOP_DEADLINE: Duration = Duration::from_secs(10);

const PASSTHROUGH: &str = r#"
[source.input]
type = "channel/frames"

[filter.crop]
type = "crop/manual"
inputs = ["input"]

[sink.output]
type = "channel/frames"
inputs = ["crop"]
"#;

/// What a stream handed back, with how its run ended.
struct Finished {
    frames: Vec<OutputFrame>,
    joined: Result<RunReport, SessionError>,
}

fn format_of(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: FrameRate::new(24, 1),
        field_order: None,
    }
}

fn format() -> FrameFormat {
    format_of(SIZE, SIZE)
}

fn backend() -> Backend {
    Backend::open_default(test_device()).expect("opens the test device")
}

fn text_input(text: &str) -> PipelineInput {
    PipelineInput::Text {
        name: "pipeline.toml".to_owned(),
        text: text.to_owned(),
    }
}

fn session_with(text: &str, vars: &[(String, String)], frame_format: FrameFormat) -> Session {
    let input = text_input(text);
    let channels = [("input".to_owned(), frame_format)];

    Session::load(input, vars, backend(), &channels).expect("the session loads")
}

fn session(text: &str) -> Session {
    session_with(text, &[], format())
}

/// Repeatable noise from `seed`, one byte per sample.
fn pattern(seed: u32, samples: usize) -> Vec<u8> {
    let mut state = seed.max(1);
    let mut bytes = Vec::with_capacity(samples);

    for _ in 0..samples {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        bytes.push((state & 0xff) as u8);
    }

    bytes
}

/// A frame of `frame_format` whose luma is noise drawn from `seed`, with flat chroma.
fn noise_frame(frame_format: &FrameFormat, seed: u32) -> InputFrame {
    let sizes = frame_format.plane_sizes();
    let luma = pattern(seed, sizes[0]);
    let u_plane = vec![90; sizes[1]];
    let v_plane = vec![180; sizes[2]];

    InputFrame::new(vec![luma, u_plane, v_plane])
}

fn noise_frames(count: usize, first_seed: u32) -> Vec<InputFrame> {
    let frame_format = format();

    (0..count)
        .map(|index| noise_frame(&frame_format, first_seed + index as u32))
        .collect()
}

/// A frame from one of two scenes, a ramp or the inverted ramp, under light noise.
fn scene_frame(index: usize) -> InputFrame {
    let side = SIZE as usize;
    let inverted = index >= SCENE_FRAMES;
    let grain = pattern(index as u32 * 7919 + 1, side * side);

    let luma = grain
        .iter()
        .enumerate()
        .map(|(pixel, sample)| {
            let ramp = (pixel % side) as i16 * 4;
            let level = if inverted { 255 - ramp } else { ramp };
            let wobble = i16::from(*sample % 9) - 4;
            (level + wobble).clamp(0, 255) as u8
        })
        .collect();
    let chroma_samples = (side / 2) * (side / 2);

    InputFrame::new(vec![luma, vec![128; chroma_samples], vec![128; chroma_samples]])
}

/// A 192x128 frame of three flat luma bands under fresh grain, swapping sides every 20 frames.
fn grainy_frame(index: usize) -> InputFrame {
    let width = 192;
    let height = 128;
    let scene = index / 20;
    let levels: [i16; 3] = if scene.is_multiple_of(2) {
        [60, 120, 180]
    } else {
        [180, 120, 60]
    };
    let grain = pattern(index as u32 * 7919 + 12345, width * height);

    let luma = grain
        .iter()
        .enumerate()
        .map(|(pixel, sample)| {
            let band = (pixel % width) * 3 / width;
            let wobble = i16::from(*sample % 9) - 4;
            (levels[band] + wobble) as u8
        })
        .collect();
    let chroma_samples = width * height / 4;

    InputFrame::new(vec![luma, vec![128; chroma_samples], vec![128; chroma_samples]])
}

/// Whether `work` finishes on a thread of its own within [STOP_DEADLINE].
fn finishes_in_time<Work>(work: Work) -> bool
where
    Work: FnOnce() + Send + 'static,
{
    let (done_sender, done_receiver) = mpsc::channel();
    thread::spawn(move || {
        work();
        let _ = done_sender.send(());
    });

    done_receiver.recv_timeout(STOP_DEADLINE).is_ok()
}

/// Whether the run's only sink consumes `steps` items within [STOP_DEADLINE].
fn sink_reaches(handle: &StreamHandle, steps: u64) -> bool {
    let progress = handle.progress();
    let started = Instant::now();

    while started.elapsed() < STOP_DEADLINE {
        let sinks = progress.sinks();
        let reached = sinks.iter().all(|sink| sink.steps >= steps);
        if reached {
            return true;
        }

        thread::sleep(Duration::from_millis(5));
    }

    false
}

fn feed(sender: SyncSender<InputFrame>, frames: Vec<InputFrame>) {
    for frame in frames {
        if sender.send(frame).is_err() {
            return;
        }
    }
}

/// Feeds `frames` from one thread and collects every frame that comes back on another.
fn round_trip(session: &mut Session, frames: Vec<InputFrame>) -> Finished {
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();

    let feeder = thread::spawn(move || feed(sender, frames));
    let collector = thread::spawn(move || receiver.iter().collect::<Vec<_>>());

    feeder.join().expect("the feeder finishes");
    let received = collector.join().expect("the collector finishes");
    let joined = handle.join();

    Finished {
        frames: received,
        joined,
    }
}

fn planes_of(frames: &[OutputFrame]) -> Vec<Vec<Vec<u8>>> {
    frames.iter().map(|frame| frame.planes.clone()).collect()
}

fn input_planes(frames: &[InputFrame]) -> Vec<Vec<Vec<u8>>> {
    frames.iter().map(|frame| frame.planes().to_vec()).collect()
}

fn indices(frames: &[OutputFrame]) -> Vec<u64> {
    frames.iter().map(|frame| frame.index).collect()
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

const GRAIN_PIPELINE: &str = r#"
[variable.grain]
type = "path"
description = "The directory the grain table is written into"

[source.input]
type = "channel/frames"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]

[sink.output]
type = "channel/frames"
inputs = ["denoise"]

[sink.grain]
type = "artefacts/files"
inputs = ["denoise"]
args.directory = "{{ grain }}"
args.include = ["grain_table"]
"#;

/// A session on [GRAIN_PIPELINE] writing its grain table into `grain_path`.
fn grain_session(grain_path: &Path) -> Session {
    let grain_text = grain_path.to_str().expect("a utf-8 path");
    let vars = [("grain".to_owned(), grain_text.to_owned())];

    session_with(GRAIN_PIPELINE, &vars, format_of(192, 128))
}

/// A channel pipeline denoising with NL4D on `workers` workers.
fn denoise_pipeline(workers: usize) -> String {
    format!(
        r#"
[source.input]
type = "channel/frames"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]
workers = {workers}

[sink.output]
type = "channel/frames"
inputs = ["denoise"]
"#
    )
}

#[test]
fn a_stream_round_trips_frames_on_two_threads() {
    let mut session = session(PASSTHROUGH);
    let frames = noise_frames(10, 1);
    let expected = input_planes(&frames);

    let finished = round_trip(&mut session, frames);
    let report = finished.joined.expect("the run succeeds");

    assert_eq!(indices(&finished.frames), (0..10).collect::<Vec<u64>>());
    assert!(
        planes_of(&finished.frames) == expected,
        "the frames came back changed"
    );
    assert_eq!(report.frames, 10);
}

#[test]
fn two_streams_run_one_after_another_on_one_session() {
    let mut session = session(PASSTHROUGH);
    let first_frames = noise_frames(4, 1);
    let second_frames = noise_frames(6, 100);
    let second_expected = input_planes(&second_frames);

    let first = round_trip(&mut session, first_frames);
    let second = round_trip(&mut session, second_frames);

    assert_eq!(first.joined.expect("the first run succeeds").frames, 4);
    assert_eq!(second.joined.expect("the second run succeeds").frames, 6);
    assert_eq!(indices(&second.frames), (0..6).collect::<Vec<u64>>());
    assert!(planes_of(&second.frames) == second_expected);
}

#[test]
fn a_chunked_pipeline_keeps_frame_order() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let log_path = directory.path().join("chunks.ndjson");
    let log_text = log_path.to_str().expect("a utf-8 path");
    let log_sink = format!(
        r#"
[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "{log_text}"
args.scenes = {{ breaks = {{ mode = "natural" }} }}
"#
    );
    let two_workers_text = denoise_pipeline(2) + &log_sink;
    let one_worker_text = denoise_pipeline(1);
    let mut parallel_session = session(&two_workers_text);
    let mut serial_session = session(&one_worker_text);
    let frames: Vec<InputFrame> = (0..2 * SCENE_FRAMES).map(scene_frame).collect();

    let parallel = round_trip(&mut parallel_session, frames.clone());
    let serial = round_trip(&mut serial_session, frames);

    parallel.joined.expect("the two-worker run succeeds");
    serial.joined.expect("the one-worker run succeeds");

    let log = std::fs::read_to_string(&log_path).expect("the log was written");
    let chunks = logged_chunks(&log);
    assert_eq!(chunks.len(), 2 * SCENE_FRAMES);
    assert_ne!(chunks[SCENE_FRAMES - 1], chunks[SCENE_FRAMES], "{chunks:?}");

    let expected: Vec<u64> = (0..2 * SCENE_FRAMES as u64).collect();
    assert_eq!(indices(&parallel.frames), expected);
    assert!(
        planes_of(&parallel.frames) == planes_of(&serial.frames),
        "two workers differ from one"
    );
}

#[test]
fn closing_the_sender_early_ends_the_stream_cleanly() {
    let mut session = session(PASSTHROUGH);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();
    let frames = noise_frames(3, 1);

    let feeder = thread::spawn(move || feed(sender, frames));
    let collector = thread::spawn(move || receiver.iter().count());
    feeder.join().expect("the feeder finishes");
    let received = collector.join().expect("the collector finishes");
    let report = handle.join().expect("the run succeeds");

    assert_eq!(received, 3);
    assert_eq!(report.frames, 3);
    assert!(!report.cancelled);
}

#[test]
fn dropping_the_receiver_ends_the_stream_with_an_error() {
    let mut session = session(PASSTHROUGH);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();
    let frames = noise_frames(40, 1);

    let feeder = thread::spawn(move || feed(sender, frames));
    let first = receiver.recv().expect("one frame arrives");
    drop(receiver);
    feeder.join().expect("the feeder finishes");
    let joined = handle.join();

    assert_eq!(first.index, 0);
    let error = joined.expect_err("the run fails");
    let message = error.to_string();
    assert!(message.contains("output"), "{message}");
}

#[test]
fn a_second_stream_waits_for_the_first() {
    let mut session = session(PASSTHROUGH);
    let mut first = session
        .start_stream(Settings::default())
        .expect("the first stream starts");

    let refused = session.start_stream(Settings::default());

    let error = refused.err().expect("a stream is still running");
    let message = error.to_string();
    assert!(message.contains("still running"), "{message}");

    let sender = first.take_sender("input").expect("a sender for the source");
    let receiver = first.take_receiver("output").expect("a receiver for the sink");
    drop(sender);
    let drained = receiver.iter().count();
    first.into_handle().join().expect("the first run succeeds");

    assert_eq!(drained, 0);
    let second = session.start_stream(Settings::default());
    assert!(second.is_ok());
}

#[test]
fn two_sessions_run_side_by_side() {
    let mut left_session = session(PASSTHROUGH);
    let mut right_session = session(PASSTHROUGH);
    let left_frames = noise_frames(8, 1);
    let right_frames = noise_frames(8, 500);
    let left_expected = input_planes(&left_frames);
    let right_expected = input_planes(&right_frames);

    let left = thread::spawn(move || round_trip(&mut left_session, left_frames));
    let right = thread::spawn(move || round_trip(&mut right_session, right_frames));
    let left = left.join().expect("the left session finishes");
    let right = right.join().expect("the right session finishes");

    left.joined.expect("the left run succeeds");
    right.joined.expect("the right run succeeds");
    assert!(planes_of(&left.frames) == left_expected, "the left frames mixed");
    assert!(
        planes_of(&right.frames) == right_expected,
        "the right frames mixed"
    );
}

#[test]
fn a_channel_source_without_a_format_is_refused() {
    let input = text_input(PASSTHROUGH);

    let loaded = Session::load(input, &[], backend(), &[]);

    let error = loaded.err().expect("the source has no format");
    let message = error.to_string();
    assert!(message.contains("input"), "{message}");
}

#[test]
fn a_format_for_an_unknown_stage_is_refused() {
    let input = text_input(PASSTHROUGH);
    let channels = [("input".to_owned(), format()), ("other".to_owned(), format())];

    let loaded = Session::load(input, &[], backend(), &channels);

    let error = loaded.err().expect("no source is named other");
    let message = error.to_string();
    assert!(message.contains("other"), "{message}");
}

#[test]
fn an_artefact_sink_still_writes_its_directory() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let grain_path = directory.path().join("grain");
    let mut session = grain_session(&grain_path);
    let frames: Vec<InputFrame> = (0..60).map(grainy_frame).collect();

    let finished = round_trip(&mut session, frames);
    let report = finished.joined.expect("the run succeeds");

    assert_eq!(finished.frames.len(), 60);
    assert!(report.artefacts > 0);
    let table_path = grain_path.join("filter.denoise.grain_table.tbl");
    let table = std::fs::read_to_string(table_path).expect("the table was written");
    assert!(table.starts_with("filmgrn1\n"), "{table}");
}

#[test]
fn a_session_discarding_artefacts_writes_none() {
    let directory = tempfile::tempdir().expect("a temp dir");
    let grain_path = directory.path().join("grain");
    let mut session = grain_session(&grain_path);
    session.discard_artefacts(true);
    let frames: Vec<InputFrame> = (0..60).map(grainy_frame).collect();

    let finished = round_trip(&mut session, frames);
    finished.joined.expect("the run succeeds");

    assert_eq!(finished.frames.len(), 60);
    let table_path = grain_path.join("filter.denoise.grain_table.tbl");
    assert!(!table_path.exists());
}

#[test]
fn the_plan_gives_every_node_its_width() {
    let pipeline = denoise_pipeline(2);
    let mut session = session(&pipeline);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let handle = stream.into_handle();
    drop(sender);

    let paths: Vec<&str> = handle.plan().iter().map(|node| node.path.as_str()).collect();
    let denoise = handle
        .plan()
        .iter()
        .find(|node| node.path == "filter.denoise")
        .expect("the plan holds the denoiser");
    let crossings = handle
        .plan()
        .iter()
        .filter(|node| matches!(node.role, NodeRole::Crossing(_)))
        .count();

    assert_eq!(paths.first(), Some(&"source.input"));
    assert!(paths.contains(&"sink.output"), "{paths:?}");
    assert_eq!(denoise.width, 2);
    assert_eq!(crossings, 2);
    handle.join().expect("the empty run succeeds");
}

#[test]
fn dropping_an_unjoined_handle_cancels_the_run() {
    let mut session = session(PASSTHROUGH);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();

    let dropped = finishes_in_time(move || drop(handle));

    assert!(dropped, "dropping the handle hung on a source waiting for frames");
    drop(sender);
    drop(receiver);
}

#[test]
fn dropping_an_unjoined_handle_stops_a_sink_nobody_drains() {
    let mut session = session(PASSTHROUGH);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();
    let feeder_sender = sender.clone();
    let frames = noise_frames(8, 1);

    thread::spawn(move || feed(feeder_sender, frames));
    let first = receiver.recv().expect("frames reach the sink");
    let filled = sink_reaches(&handle, 3);
    let dropped = finishes_in_time(move || drop(handle));

    assert!(filled, "the sink never filled its channel");

    assert_eq!(first.index, 0);
    assert!(dropped, "dropping the handle hung on a sink with a full channel");
    drop(sender);
    drop(receiver);
}

#[test]
fn cancelling_a_held_stream_reports_it_cancelled() {
    let mut session = session(PASSTHROUGH);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();
    let (joined_sender, joined_receiver) = mpsc::channel();

    handle.cancel();
    thread::spawn(move || {
        let joined = handle.join();
        let _ = joined_sender.send(joined);
    });
    let joined = joined_receiver
        .recv_timeout(STOP_DEADLINE)
        .expect("the join hung while the caller held both ends");

    let report = joined.expect("a cancelled run is not an error");
    assert!(report.cancelled);
    drop(sender);
    drop(receiver);
}

#[test]
fn dropping_the_session_leaves_its_stream_running() {
    let mut session = session(PASSTHROUGH);
    let frames = noise_frames(5, 1);
    let mut stream = session
        .start_stream(Settings::default())
        .expect("the stream starts");
    let sender = stream.take_sender("input").expect("a sender for the source");
    let receiver = stream.take_receiver("output").expect("a receiver for the sink");
    let handle = stream.into_handle();

    let dropped = finishes_in_time(move || drop(session));
    let feeder = thread::spawn(move || feed(sender, frames));
    let collector = thread::spawn(move || receiver.iter().count());
    feeder.join().expect("the feeder finishes");
    let received = collector.join().expect("the collector finishes");
    let report = handle.join().expect("the run succeeds");

    assert!(dropped, "dropping the session hung");
    assert_eq!(received, 5);
    assert_eq!(report.frames, 5);
}

#[test]
fn a_format_given_twice_is_refused() {
    let input = text_input(PASSTHROUGH);
    let channels = [("input".to_owned(), format()), ("input".to_owned(), format())];

    let loaded = Session::load(input, &[], backend(), &channels);

    let error = loaded.err().expect("input has two formats");
    let message = error.to_string();
    assert!(message.contains("more than once"), "{message}");
}
