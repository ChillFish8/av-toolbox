use av_toolbox_core::{
    AnyDemand,
    DemandKind,
    InputArgument,
    Inputs,
    LayerDemand,
    PipelineRegistry,
    Provider,
    ProviderDefinition,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{
    Capabilities,
    FrameContext,
    FrameFormat,
    HostFrame,
    HostSide,
    HostStreamItem,
    LayerId,
    SampleFormat,
    StreamFormat,
    default_accelerators,
    testing_context,
};

use crate::scene::score::{SceneScoreProvider, ServeScores};
use crate::scene::{SceneCut, SceneScores, SceneSpeed, ScoreRequest};
use crate::tests::clip::{THREE_SCENES, clip, grey8};

fn score_demands(layers: u32) -> Vec<LayerDemand> {
    let request = ScoreRequest {
        speed: SceneSpeed::Standard,
        after: None,
    };

    (0..layers)
        .map(|layer| LayerDemand {
            layer: LayerId(layer),
            demand: AnyDemand::new::<SceneScores>(request.clone()),
        })
        .collect()
}

fn frames_of(items: Vec<HostStreamItem>) -> Vec<HostFrame> {
    items
        .into_iter()
        .map(|item| match item {
            HostStreamItem::Frame(frame) => frame,
            HostStreamItem::Mask(_) => panic!("the score provider only emits frames"),
        })
        .collect()
}

/// Runs `frames` through a score provider serving `layers` layers, returning everything it emits.
fn score(context: &mut FrameContext, layers: u32, frames: Vec<HostFrame>) -> Vec<HostFrame> {
    let args = score_demands(layers);
    let mut provider = ServeScores::new(&args, &grey8()).expect("builds");

    let mut emitted = Vec::new();
    for frame in frames {
        let slot = HostStreamItem::Frame(frame);
        let inputs = Inputs::<HostSide>::from_slots(vec![Some(slot)]);
        let out = provider.process(context, inputs).expect("processes");
        emitted.extend(frames_of(out));
    }

    let rest = provider.end_of_stream(context).expect("finishes");
    emitted.extend(frames_of(rest));

    emitted
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(16, 16)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

#[test]
fn every_layer_gets_the_same_cut() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);

    let emitted = score(&mut context, 2, frames);

    assert_eq!(emitted.len(), 90);

    let mut starts = Vec::new();
    for (position, frame) in emitted.iter().enumerate() {
        let provided = &frame.meta().provided;
        let first = provided.get::<SceneCut>(LayerId(0)).expect("scored");
        let second = provided.get::<SceneCut>(LayerId(1)).expect("scored");
        assert_eq!(first, second);

        if first.starts_scene {
            starts.push(position);
        }
    }

    assert_eq!(starts, [0, 30, 60]);
}

#[test]
fn half_floats_are_refused() {
    let capabilities = capabilities();
    let args = score_demands(1);
    let half = FrameFormat {
        sample: SampleFormat::F16,
        ..grey8()
    };

    let half_offered = StreamFormat::Frames(half);
    let byte_offered = StreamFormat::Frames(grey8());
    let half_accepted = SceneScoreProvider::accepts_input(&capabilities, &args, half_offered);
    let byte_accepted = SceneScoreProvider::accepts_input(&capabilities, &args, byte_offered);

    assert!(half_accepted.expect("answers").is_empty());
    assert!(byte_accepted.expect("answers").accepts(SampleFormat::U8));
}

#[test]
fn the_score_provider_is_registered_for_scene_scores() {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("registers");

    let found = registry.provider_for(DemandKind::of::<SceneScores>());
    let identifier = found.map(|(identifier, _)| identifier);
    assert_eq!(identifier, Some("provider/scene-score"));

    let args = score_demands(1);
    let args = serde_json::to_value(args).expect("serialises");
    let input = InputArgument::named("input", StageHandle::new(0), StreamType::Frames);
    let prepared = registry.prepare(StageKind::Provider, "provider/scene-score", &args, &[input]);

    assert!(prepared.is_ok());
}
