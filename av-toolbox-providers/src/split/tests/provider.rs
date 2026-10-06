use av_toolbox_core::{
    AnyDemand,
    DemandDefinition,
    DemandKind,
    InputArgument,
    LayerDemand,
    PipelineRegistry,
    StageDefinition,
    StageHandle,
    StageKind,
    StreamType,
    run_meta_provider,
};
use av_toolbox_gpu::{AnyItem, Channel, FrameMeta, HostStreamItem, LayerId, testing_context};
use serde_json::json;

use super::{SCORE_LAYER, chunks_in, demands, limited, scores_provision, serve};
use crate::scene::{SceneRequest, SceneScores, SceneSpeed, Scenes, ScoreRequest};
use crate::split::provider::{SceneSplitProvider, ServeSplits};
use crate::tests::clip::{THREE_SCENES, clip, grey8};

fn after(name: &str, speed: SceneSpeed) -> SceneRequest {
    SceneRequest {
        speed,
        after: Some(name.to_string()),
        ..SceneRequest::natural()
    }
}

#[test]
fn two_layers_are_served_by_one_detection() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let requests = [SceneRequest::natural(), limited(0, Some(12))];

    let emitted = serve(&mut context, &grey8(), &requests, frames);

    assert_eq!(emitted.len(), 90);
    for frame in &emitted {
        let layers: Vec<LayerId> = frame.meta().provided.layers().collect();
        assert_eq!(layers, vec![LayerId(0), LayerId(1), SCORE_LAYER]);
    }

    let natural: Vec<u64> = (0..90).map(|index| index / 30).collect();
    assert_eq!(chunks_in(&emitted, 0), natural);

    let split = chunks_in(&emitted, 1);
    assert!(split.windows(2).all(|pair| pair[1] >= pair[0]), "{split:?}");
    assert!(split[89] > natural[89], "{split:?}");
}

#[test]
fn frames_pass_through_unchanged() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let originals = frames.clone();

    let emitted = serve(&mut context, &grey8(), &[limited(0, Some(12))], frames);

    assert_eq!(emitted.len(), originals.len());
    for (original, passed) in originals.iter().zip(&emitted) {
        assert_eq!(passed.meta().index, original.meta().index);

        let original_luma = original.plane(Channel::Y).expect("luma").bytes();
        let passed_luma = passed.plane(Channel::Y).expect("luma").bytes();
        assert_eq!(passed_luma, original_luma);
    }
}

#[test]
fn the_split_provider_is_registered_for_scenes() {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("registers");

    let found = registry.provider_for(DemandKind::of::<Scenes>());
    let identifier = found.map(|(identifier, _)| identifier);
    assert_eq!(identifier, Some("provider/scene-split"));

    let args = demands(&[SceneRequest::natural()]);
    let args = serde_json::to_value(args).expect("serialises");
    let input = InputArgument::named("input", StageHandle::new(0), StreamType::Frames)
        .with_demand(DemandKind::of::<SceneScores>(), SCORE_LAYER);
    let prepared = registry.prepare(StageKind::Provider, "provider/scene-split", &args, &[input]);

    assert!(prepared.is_ok());
}

#[test]
fn the_split_demands_one_score_at_the_slowest_speed_after_the_first_layer() {
    let requests = [
        after("denoise", SceneSpeed::Fast),
        after("other", SceneSpeed::Standard),
    ];
    let args = demands(&requests);

    let demanded = SceneSplitProvider::demands(&args);

    assert_eq!(demanded.len(), 1);
    assert_eq!(demanded[0].input, "input");

    let request = demanded[0].demand.request::<SceneScores>();
    let expected = ScoreRequest {
        speed: SceneSpeed::Standard,
        after: Some("denoise".to_string()),
    };
    assert_eq!(request, Some(expected));
}

#[test]
fn split_refuses_an_unscored_frame() {
    let context = testing_context();
    let args = demands(&[SceneRequest::natural()]);
    let mut split = ServeSplits::new(&args, None, scores_provision()).expect("builds");
    let frame = context.alloc_host_frame(&grey8(), FrameMeta::at(0)).freeze();
    let item = AnyItem::Host(HostStreamItem::from(frame));
    let mut stage_context = testing_context();

    let Err(error) = run_meta_provider(&mut split, &mut stage_context, item) else {
        panic!("an unscored frame should be refused");
    };

    assert_eq!(
        error.to_string(),
        "scene-split was given a frame scene-score never scored"
    );
}

#[test]
fn a_demand_of_another_kind_is_refused() {
    struct Other;

    impl DemandDefinition for Other {
        type Request = ();
        type Output = ();

        fn kind() -> &'static str {
            "other"
        }
    }

    let args = vec![LayerDemand {
        layer: LayerId(0),
        demand: AnyDemand::new::<Other>(()),
    }];

    let built = ServeSplits::new(&args, None, scores_provision());

    assert!(built.is_err());
}

#[test]
fn a_malformed_scene_request_is_refused_as_malformed() {
    let value = json!({ "kind": "scenes", "request": { "breaks": { "mode": "sideways" } } });
    let demand: AnyDemand = serde_json::from_value(value).expect("deserialises");
    let args = vec![LayerDemand {
        layer: LayerId(0),
        demand,
    }];

    let Err(error) = ServeSplits::new(&args, None, scores_provision()) else {
        panic!("a malformed scenes request should be refused");
    };

    let message = error.to_string();
    assert!(message.contains("malformed"), "{message}");
    assert!(message.contains("sideways"), "{message}");
    assert!(!message.contains("asks for a"), "{message}");
}
