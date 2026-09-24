use av_toolbox_core::{
    AnyItem,
    BuildStageError,
    InputArgument,
    PipelineRegistry,
    PreparedStage,
    StageHandle,
    StageKind,
    StreamType,
    end_meta_transform,
    run_meta_transform,
};
use av_toolbox_gpu::{
    ColourInfo,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostStreamItem,
    SampleFormat,
    SceneScore,
    StreamItem,
    testing_context,
};
use serde_json::json;

use crate::split::SplitLimit;
use crate::split::stage::Split;

fn format() -> FrameFormat {
    FrameFormat {
        width: 16,
        height: 16,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    }
}

fn meta(index: u64, chunk: u64, score: Option<f32>) -> FrameMeta {
    let scene_score = score.map(SceneScore::new);

    FrameMeta {
        chunk,
        scene_score,
        ..FrameMeta::at(index)
    }
}

fn host_item(context: &FrameContext, meta: FrameMeta) -> AnyItem {
    let frame = context.alloc_host_frame(&format(), meta).freeze();

    AnyItem::from(HostStreamItem::Frame(frame))
}

fn device_item(context: &FrameContext, meta: FrameMeta) -> AnyItem {
    let frame = context.alloc_frame(&format(), meta);

    AnyItem::from(StreamItem::Frame(frame))
}

fn frames_limit(frames: usize) -> SplitLimit {
    SplitLimit::new(Some(frames), None).expect("a valid limit")
}

fn run(split: &mut Split, context: &mut FrameContext, items: Vec<AnyItem>) -> Vec<AnyItem> {
    let mut emitted = Vec::new();
    for item in items {
        let out = run_meta_transform(split, context, vec![Some(item)]).expect("runs");
        emitted.extend(out);
    }

    let flushed = end_meta_transform(split, context).expect("flushes");
    emitted.extend(flushed);

    emitted
}

fn chunks_of(items: &[AnyItem]) -> Vec<u64> {
    items.iter().map(|item| item.meta().chunk).collect()
}

fn prepare(value: serde_json::Value) -> Result<PreparedStage, BuildStageError> {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("registers");
    let input = InputArgument::positional(StageHandle::new(0), StreamType::Frames);

    registry.prepare(StageKind::Transform, "scene/split", &value, &[input])
}

#[test]
fn chunks_come_out_numbered_from_zero_without_gaps() {
    let mut context = testing_context();
    let incoming = [5, 5, 9, 9, 9];
    let items = incoming
        .iter()
        .enumerate()
        .map(|(index, &chunk)| host_item(&context, meta(index as u64, chunk, None)))
        .collect();
    let mut split = Split::new(frames_limit(10));

    let emitted = run(&mut split, &mut context, items);

    assert_eq!(chunks_of(&emitted), vec![0, 0, 1, 1, 1]);
}

#[test]
fn a_scene_within_the_limit_passes_through_as_one_chunk() {
    let mut context = testing_context();
    let items = (0..4)
        .map(|index| host_item(&context, meta(index, 3, Some(1.0))))
        .collect();
    let mut split = Split::new(frames_limit(10));

    let emitted = run(&mut split, &mut context, items);

    let indices: Vec<u64> = emitted.iter().map(|item| item.meta().index).collect();
    assert_eq!(indices, vec![0, 1, 2, 3]);
    assert_eq!(chunks_of(&emitted), vec![0, 0, 0, 0]);
}

#[test]
fn an_unscored_scene_splits_evenly() {
    let mut context = testing_context();
    let items = (0..25)
        .map(|index| host_item(&context, meta(index, 0, None)))
        .collect();
    let mut split = Split::new(frames_limit(10));

    let emitted = run(&mut split, &mut context, items);

    let mut expected = vec![0; 8];
    expected.extend([1; 8]);
    expected.extend([2; 9]);
    assert_eq!(chunks_of(&emitted), expected);
}

#[test]
fn a_scored_scene_splits_at_its_strongest_cut() {
    let mut context = testing_context();
    let items = (0..25)
        .map(|index| {
            let score = if index == 10 { 3.0 } else { 1.0 };
            host_item(&context, meta(index, 0, Some(score)))
        })
        .collect();
    let mut split = Split::new(frames_limit(10));

    let emitted = run(&mut split, &mut context, items);

    let mut expected = vec![0; 10];
    expected.extend([1; 7]);
    expected.extend([2; 8]);
    assert_eq!(chunks_of(&emitted), expected);
}

#[test]
fn a_scene_is_held_until_the_next_chunk_arrives() {
    let mut context = testing_context();
    let mut split = Split::new(frames_limit(10));

    let mut held = Vec::new();
    for index in 0..3 {
        let item = host_item(&context, meta(index, 0, None));
        let out = run_meta_transform(&mut split, &mut context, vec![Some(item)]).expect("runs");
        held.extend(out);
    }

    let next = host_item(&context, meta(3, 1, None));
    let released = run_meta_transform(&mut split, &mut context, vec![Some(next)]).expect("runs");

    assert!(held.is_empty());
    assert_eq!(released.len(), 3);
}

#[test]
fn items_keep_their_side_and_scores() {
    let mut context = testing_context();
    let items = vec![
        host_item(&context, meta(0, 0, Some(1.5))),
        device_item(&context, meta(1, 0, Some(2.5))),
        device_item(&context, meta(2, 1, None)),
    ];
    let mut split = Split::new(frames_limit(10));

    let emitted = run(&mut split, &mut context, items);

    let sides: Vec<bool> = emitted.iter().map(AnyItem::is_host).collect();
    let scores: Vec<Option<SceneScore>> = emitted.iter().map(|item| item.meta().scene_score).collect();
    assert_eq!(sides, vec![true, false, false]);
    assert_eq!(
        scores,
        vec![Some(SceneScore::new(1.5)), Some(SceneScore::new(2.5)), None]
    );
    assert_eq!(chunks_of(&emitted), vec![0, 0, 1]);
}

#[test]
fn split_arguments_prepare() {
    let defaults = prepare(json!({}));
    let frames = prepare(json!({ "max_frames": 48 }));

    assert!(defaults.is_ok());
    assert!(frames.is_ok());
}

#[test]
fn bad_split_arguments_are_refused_at_prepare() {
    let zero = prepare(json!({ "max_frames": 0 }));
    let unknown = prepare(json!({ "max_frame": 4 }));
    let both_null = prepare(json!({ "max_frames": null, "max_seconds": null }));

    assert!(matches!(zero, Err(BuildStageError::InvalidArgs { .. })));
    assert!(matches!(unknown, Err(BuildStageError::InvalidArgs { .. })));
    assert!(matches!(both_null, Err(BuildStageError::InvalidArgs { .. })));
}
