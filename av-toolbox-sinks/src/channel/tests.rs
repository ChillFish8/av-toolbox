use std::sync::mpsc::{Receiver, sync_channel};
use std::thread;
use std::time::Instant;

use av_toolbox_core::{
    CHANNEL_POLL,
    ChannelHub,
    ChannelStop,
    ChannelToken,
    Inputs,
    OutputFrame,
    Sink,
    SinkDefinition,
};
use av_toolbox_gpu::{
    Capabilities,
    Channel,
    Depth,
    FrameFormat,
    FrameRate,
    HostStreamItem,
    SampleFormat,
    StreamFormat,
    default_accelerators,
    testing_context,
};

use super::*;
use crate::shared::testing::{consume, filled_frame, yuv420};

fn rate() -> Option<FrameRate> {
    FrameRate::new(25, 1)
}

fn sixteen_bit() -> SampleFormat {
    SampleFormat::U16 { depth: Depth::TEN }
}

fn args(token: &ChannelToken) -> ChannelArgs {
    ChannelArgs {
        session: Some(token.as_str().to_owned()),
        stage: Some("output".to_owned()),
    }
}

fn sink_with(depth: usize) -> (FramesSink, Receiver<OutputFrame>) {
    let stop = ChannelStop::new();
    sink_stopped_by(depth, stop)
}

fn sink_stopped_by(depth: usize, stop: ChannelStop) -> (FramesSink, Receiver<OutputFrame>) {
    let hub = ChannelHub::default();
    let token = ChannelToken::new();
    let (sender, receiver) = sync_channel(depth);
    hub.bind_sink(&token, "output", sender, stop);

    let sink = sink_from(&hub, &args(&token)).expect("opens");

    (sink, receiver)
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn assert_round_trip(format: FrameFormat) {
    let (mut sink, receiver) = sink_with(4);
    consume(&mut sink, filled_frame(&format, 7, 9));

    let sent = receiver.try_recv().expect("a frame was sent");

    assert_eq!(sent.index, 7);
    assert_eq!(sent.format, format);
    assert_eq!(sent.planes.len(), format.channels().len());
    for (channel, plane) in format.channels().iter().zip(&sent.planes) {
        let expected = format.channel_bytes(*channel).expect("the layout carries it");
        assert_eq!(plane.len(), expected);
        assert!(plane.iter().all(|&byte| byte == 9));
    }
}

#[test]
fn frames_are_sent_with_their_index_and_format() {
    assert_round_trip(yuv420(4, 2, SampleFormat::U8, rate()));
}

#[test]
fn sixteen_bit_frames_are_copied_whole() {
    assert_round_trip(yuv420(4, 2, sixteen_bit(), rate()));
}

#[test]
fn odd_sized_frames_are_copied_whole() {
    assert_round_trip(yuv420(5, 3, SampleFormat::U8, rate()));
    assert_round_trip(yuv420(5, 3, sixteen_bit(), rate()));
}

#[test]
fn the_receiver_ends_at_the_end_of_the_stream() {
    let (mut sink, receiver) = sink_with(4);
    let mut context = testing_context();

    sink.end_of_stream(&mut context).expect("ends");

    assert!(receiver.recv().is_err());
}

#[test]
fn a_dropped_receiver_fails_the_sink() {
    let format = yuv420(4, 2, SampleFormat::U8, rate());
    let (mut sink, receiver) = sink_with(4);
    drop(receiver);

    let mut context = testing_context();
    let item = HostStreamItem::from(filled_frame(&format, 0, 1));
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(item)]);
    let error = sink.consume(&mut context, inputs).expect_err("nobody receives");

    let message = error.to_string();
    assert!(message.contains("stopped receiving"), "{message}");
    assert!(message.contains("output"), "{message}");
}

#[test]
fn a_stopped_run_frees_a_sink_whose_caller_stopped_draining() {
    let format = yuv420(4, 2, SampleFormat::U8, rate());
    let stop = ChannelStop::new();
    let (mut sink, receiver) = sink_stopped_by(1, stop.clone());
    consume(&mut sink, filled_frame(&format, 0, 1));

    let stopper = thread::spawn(move || {
        thread::sleep(CHANNEL_POLL);
        stop.stop();
    });
    let mut context = testing_context();
    let item = HostStreamItem::from(filled_frame(&format, 1, 1));
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(item)]);
    let started = Instant::now();
    let consumed = sink.consume(&mut context, inputs);
    let waited = started.elapsed();
    stopper.join().expect("the stopper finishes");

    assert!(consumed.is_ok(), "{consumed:?}");
    assert!(waited < 10 * CHANNEL_POLL, "{waited:?}");
    let held = receiver.try_recv().expect("the first frame is still waiting");
    assert_eq!(held.index, 0);
}

#[test]
fn a_sink_without_a_session_names_the_library() {
    let hub = ChannelHub::default();

    let error = sink_from(&hub, &ChannelArgs::default())
        .err()
        .expect("no session");

    assert!(error.to_string().contains("av-toolbox library"), "{error}");
}

#[test]
fn the_offered_format_is_accepted_unchanged() {
    let format = yuv420(4, 2, sixteen_bit(), rate());

    let accepted = Frames::accepts_input(
        &capabilities(),
        &ChannelArgs::default(),
        0,
        StreamFormat::Frames(format),
        &[],
    )
    .expect("accepts");

    assert_eq!(accepted, FormatSet::of([sixteen_bit()]));
}

#[test]
fn the_channel_sink_is_registered() {
    let mut registry = av_toolbox_core::PipelineRegistry::new();
    crate::register(&mut registry).expect("the sinks register");

    let found = registry
        .stages()
        .any(|info| info.full_identifier() == "channel/frames");

    assert!(found);
}

#[test]
fn a_frame_after_the_end_of_the_stream_fails_naming_the_stage() {
    let format = yuv420(4, 2, SampleFormat::U8, rate());
    let (mut sink, _receiver) = sink_with(4);
    let mut context = testing_context();
    sink.end_of_stream(&mut context).expect("ends");

    let item = HostStreamItem::from(filled_frame(&format, 0, 1));
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(item)]);
    let error = sink.consume(&mut context, inputs).expect_err("already ended");

    let message = error.to_string();
    assert!(message.contains("output"), "{message}");
    assert!(message.contains("end of the stream"), "{message}");
}

#[test]
fn a_missing_plane_error_names_the_stage_and_channel() {
    let error = missing_plane_error("output", Channel::U);

    let message = error.to_string();
    assert!(message.contains("output"), "{message}");
    assert!(message.contains("U"), "{message}");
}
