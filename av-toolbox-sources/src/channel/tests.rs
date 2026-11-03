use std::sync::mpsc::channel;
use std::time::Duration;

use av_toolbox_core::{ChannelHub, ChannelToken, InputFrame, SourceDefinition};
use av_toolbox_gpu::{
    Capabilities,
    Channel,
    ColourInfo,
    FrameFormat,
    FrameLayout,
    FrameRate,
    SampleFormat,
    Subsampling,
    default_accelerators,
    testing_context,
};

use super::*;
use crate::shared::produced_by;

fn format() -> FrameFormat {
    FrameFormat {
        width: 4,
        height: 2,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: FrameRate::new(25, 1),
        field_order: None,
    }
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn frame(fill: u8) -> InputFrame {
    InputFrame::new(vec![vec![fill; 8], vec![fill; 2], vec![fill; 2]])
}

fn args(token: &ChannelToken) -> ChannelArgs {
    ChannelArgs {
        session: Some(token.as_str().to_owned()),
        stage: Some("input".to_owned()),
    }
}

fn source_with(frames: Vec<InputFrame>) -> FramesSource {
    let hub = ChannelHub::default();
    let token = ChannelToken::new();
    let (sender, receiver) = channel();
    hub.bind_source(&token, "input", format(), receiver);
    for frame in frames {
        sender.send(frame).expect("the receiver is bound");
    }
    drop(sender);

    let opened = open_from(&hub, &args(&token)).expect("opens");

    FramesSource {
        opened,
        next_index: 0,
    }
}

#[test]
fn frames_come_out_in_order_with_their_indices() {
    let mut context = testing_context();
    let mut source = source_with(vec![frame(1), frame(2), frame(3)]);

    let mut frames = Vec::new();
    while let Some(next) = source.next(&mut context).expect("reads") {
        frames.push(next);
    }

    assert_eq!(frames.len(), 3);
    let last = &frames[2];
    assert_eq!(last.meta().index, 2);
    assert_eq!(last.meta().timestamp, Some(Duration::from_millis(80)));
    let luma = last.plane(Channel::Y).expect("luma").bytes();
    assert!(luma.iter().all(|&byte| byte == 3));
}

#[test]
fn a_closed_channel_ends_the_stream() {
    let mut context = testing_context();
    let mut source = source_with(Vec::new());

    let next = source.next(&mut context).expect("reads");

    assert!(next.is_none());
}

#[test]
fn a_frame_with_a_short_plane_fails_naming_it() {
    let mut context = testing_context();
    let short = InputFrame::new(vec![vec![0; 8], vec![0; 1], vec![0; 2]]);
    let mut source = source_with(vec![short]);

    let error = source.next(&mut context).expect_err("a short plane");

    let message = error.to_string();
    assert!(message.contains("input"), "{message}");
    assert!(message.contains("plane 1"), "{message}");
    assert!(message.contains("2 bytes"), "{message}");
    assert!(message.contains("was 1"), "{message}");
}

#[test]
fn a_frame_with_too_few_planes_fails() {
    let mut context = testing_context();
    let few = InputFrame::new(vec![vec![0; 8]]);
    let mut source = source_with(vec![few]);

    let error = source.next(&mut context).expect_err("too few planes");

    let message = error.to_string();
    assert!(message.contains("3 planes"), "{message}");
    assert!(message.contains("got 1"), "{message}");
}

#[test]
fn the_bound_format_is_what_it_produces() {
    let source = source_with(Vec::new());
    let capabilities = capabilities();

    let produced = Frames::produces(&capabilities, &source.opened).expect("produces");

    let expected = produced_by(&format());
    assert_eq!(produced.template, expected.template);
    assert_eq!(produced.samples, expected.samples);
}

#[test]
fn a_source_without_a_session_names_the_library() {
    let hub = ChannelHub::default();

    let error = open_from(&hub, &ChannelArgs::default())
        .err()
        .expect("no session");

    let message = error.to_string();
    assert!(
        message.contains("only runs through the av-toolbox library"),
        "{message}"
    );
}

#[test]
fn the_channel_source_is_registered() {
    let mut registry = av_toolbox_core::PipelineRegistry::new();
    crate::register(&mut registry).expect("the sources register");

    let found = registry
        .stages()
        .any(|info| info.full_identifier() == "channel/frames");

    assert!(found);
}
