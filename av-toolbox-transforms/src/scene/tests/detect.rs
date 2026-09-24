use av_toolbox_core::{
    BuildStageError,
    InputArgument,
    Inputs,
    PipelineRegistry,
    PreparedStage,
    StageHandle,
    StageKind,
    StreamType,
    Transform,
    TransformDefinition,
};
use av_toolbox_gpu::{
    Depth,
    FrameContext,
    FrameFormat,
    FrameLayout,
    HostFrame,
    HostSide,
    HostStreamItem,
    SampleFormat,
    StreamFormat,
    testing_context,
};
use serde_json::json;

use super::clip::{THREE_SCENES, clip, grey8};
use crate::scene::SceneDetect;
use crate::scene::args::DetectArgs;
use crate::scene::detect::Detect;

fn args(value: serde_json::Value) -> DetectArgs {
    serde_json::from_value(value).expect("the arguments parse")
}

fn process(detect: &mut Detect, context: &mut FrameContext, frame: HostFrame) -> Vec<HostStreamItem> {
    let slot = HostStreamItem::Frame(frame);
    let inputs = Inputs::<HostSide>::from_slots(vec![Some(slot)]);

    detect.process(context, inputs).expect("processes")
}

fn run(detect: &mut Detect, context: &mut FrameContext, frames: Vec<HostFrame>) -> Vec<HostStreamItem> {
    let mut emitted = Vec::new();
    for frame in frames {
        let out = process(detect, context, frame);
        emitted.extend(out);
    }

    let flushed = detect.end_of_stream(context).expect("flushes");
    emitted.extend(flushed);

    emitted
}

fn prepare(value: serde_json::Value) -> Result<PreparedStage, BuildStageError> {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("registers");
    let input = InputArgument::positional(StageHandle::new(0), StreamType::Frames);

    registry.prepare(StageKind::Transform, "scene/detect", &value, &[input])
}

#[test]
fn natural_mode_stamps_one_chunk_per_scene() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let natural = args(json!({ "mode": "natural" }));
    let mut detect = Detect::new(&natural, &grey8()).expect("builds");

    let emitted = run(&mut detect, &mut context, frames);

    let indices: Vec<u64> = emitted.iter().map(|item| item.meta().index).collect();
    let expected_indices: Vec<u64> = (0..90).collect();
    assert_eq!(indices, expected_indices);

    let chunks: Vec<u64> = emitted.iter().map(|item| item.meta().chunk).collect();
    let expected_chunks: Vec<u64> = (0..90).map(|index| index / 30).collect();
    assert_eq!(chunks, expected_chunks);
}

#[test]
fn natural_mode_stamps_scores() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let natural = args(json!({ "mode": "natural" }));
    let mut detect = Detect::new(&natural, &grey8()).expect("builds");

    let emitted = run(&mut detect, &mut context, frames);

    assert_eq!(emitted[0].meta().scene_score, None);
    assert!(emitted[30].meta().scene_score.is_some());
}

#[test]
fn natural_mode_holds_only_the_lookahead() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let natural = args(json!({ "mode": "natural" }));
    let mut detect = Detect::new(&natural, &grey8()).expect("builds");

    let mut emitted = 0;
    for frame in frames.into_iter().take(10) {
        let out = process(&mut detect, &mut context, frame);
        emitted += out.len();
    }

    assert_eq!(emitted, 5);
}

#[test]
fn chunks_start_from_zero_whatever_came_in() {
    let mut context = testing_context();
    let mut frames = clip(&context, &grey8(), THREE_SCENES);
    for frame in &mut frames {
        frame.meta_mut().chunk = 7;
    }

    let natural = args(json!({ "mode": "natural" }));
    let mut detect = Detect::new(&natural, &grey8()).expect("builds");

    let emitted = run(&mut detect, &mut context, frames);

    assert_eq!(emitted[0].meta().chunk, 0);
    assert_eq!(emitted[89].meta().chunk, 2);
}

#[test]
fn a_single_frame_stream_is_one_chunk() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), &[(1, 1)]);
    let natural = args(json!({ "mode": "natural" }));
    let mut detect = Detect::new(&natural, &grey8()).expect("builds");

    let emitted = run(&mut detect, &mut context, frames);

    assert_eq!(emitted.len(), 1);
    assert_eq!(emitted[0].meta().chunk, 0);
}

#[test]
fn integer_samples_are_accepted() {
    let context = testing_context();
    let natural = args(json!({ "mode": "natural" }));
    let eight_bit = StreamFormat::Frames(grey8());
    let ten_bit_format = FrameFormat {
        sample: SampleFormat::U16 { depth: Depth::TEN },
        ..grey8()
    };
    let ten_bit = StreamFormat::Frames(ten_bit_format);

    let eight =
        SceneDetect::accepts_input(context.capabilities(), &natural, 0, eight_bit, &[]).expect("accepts");
    let ten = SceneDetect::accepts_input(context.capabilities(), &natural, 0, ten_bit, &[]).expect("accepts");

    assert_eq!(eight.preferences(), &[SampleFormat::U8]);
    assert_eq!(ten.preferences(), &[SampleFormat::U16 { depth: Depth::TEN }]);
}

#[test]
fn float_samples_are_left_out() {
    let context = testing_context();
    let natural = args(json!({ "mode": "natural" }));
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..grey8()
    };
    let float = StreamFormat::Frames(float_format);

    let accepted =
        SceneDetect::accepts_input(context.capabilities(), &natural, 0, float, &[]).expect("answers");

    assert!(accepted.is_empty());
}

#[test]
fn rgb_is_refused() {
    let context = testing_context();
    let natural = args(json!({ "mode": "natural" }));
    let rgb_format = FrameFormat {
        layout: FrameLayout::Rgb,
        ..grey8()
    };
    let rgb = StreamFormat::Frames(rgb_format);

    let accepted = SceneDetect::accepts_input(context.capabilities(), &natural, 0, rgb, &[]);

    assert!(accepted.is_err());
}

#[test]
fn natural_arguments_prepare() {
    let prepared = prepare(json!({ "mode": "natural", "speed": "fast" }));

    assert!(prepared.is_ok());
}

#[test]
fn bad_arguments_are_refused_at_prepare() {
    let above = prepare(json!({ "mode": "natural", "min_scene_frames": 30, "max_scene_frames": 20 }));
    let unknown = prepare(json!({ "mode": "natural", "bogus": 1 }));

    assert!(matches!(above, Err(BuildStageError::InvalidArgs { .. })));
    assert!(matches!(unknown, Err(BuildStageError::InvalidArgs { .. })));
}
