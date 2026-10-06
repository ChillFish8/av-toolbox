use av_toolbox_core::{
    AnyDemand,
    DemandDefinition,
    DemandKind,
    InputArgument,
    PipelineRegistry,
    Placement,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{Channel, LayerId, testing_context};
use serde_json::json;

use super::{chunks_in, demands, limited, serve};
use crate::provider::{SceneBreaks, SceneRequest, SceneSpeed, Scenes};
use crate::tests::clip::{THREE_SCENES, clip, grey8};

#[test]
fn two_layers_are_served_by_one_detection() {
    let mut context = testing_context();
    let frames = clip(&context, &grey8(), THREE_SCENES);
    let requests = [SceneRequest::natural(), limited(0, Some(12))];

    let emitted = serve(&mut context, &grey8(), &requests, frames);

    assert_eq!(emitted.len(), 90);
    for frame in &emitted {
        let layers: Vec<LayerId> = frame.meta().chunks.iter().map(|(layer, _)| layer).collect();
        assert_eq!(layers, vec![LayerId(0), LayerId(1)]);
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
fn scene_requests_describe_themselves() {
    let seconds = SceneRequest::limited_to_seconds(10.0);
    let frames = SceneRequest {
        breaks: SceneBreaks::Limited {
            min_frames: 24,
            max_frames: Some(12),
            max_seconds: None,
        },
        ..SceneRequest::natural()
    };

    assert_eq!(Scenes::describe(&SceneRequest::natural()), "natural");
    assert_eq!(Scenes::describe(&seconds), "limited ≤10 s");
    assert_eq!(Scenes::describe(&frames), "limited ≥24 frames ≤12 frames");
}

#[test]
fn a_scene_request_reads_from_toml_shaped_json() {
    let value = json!({ "breaks": { "mode": "limited", "max_seconds": 10.0 } });

    let request: SceneRequest = serde_json::from_value(value).expect("parses");

    assert_eq!(request, SceneRequest::limited_to_seconds(10.0));
    assert_eq!(request.speed, SceneSpeed::Standard);
    assert_eq!(request.after, None);
}

#[test]
fn the_scene_provider_is_registered_for_scenes() {
    let mut registry = PipelineRegistry::new();
    crate::register(&mut registry).expect("registers");

    let found = registry.provider_for(DemandKind::of::<Scenes>());
    assert_eq!(found, Some(("provider/scenes", Placement::AfterSource)));

    let args = demands(&[SceneRequest::natural()]);
    let args = serde_json::to_value(args).expect("serialises");
    let input = InputArgument::named("input", StageHandle::new(0), StreamType::Frames);
    let prepared = registry.prepare(StageKind::Provider, "provider/scenes", &args, &[input]);

    assert!(prepared.is_ok());
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

    let args = vec![av_toolbox_core::LayerDemand {
        layer: LayerId(0),
        demand: AnyDemand::new::<Other>(()),
    }];

    let built = crate::provider::ServeScenes::new(&args, &grey8());

    assert!(built.is_err());
}

#[test]
fn a_malformed_scene_request_is_refused_as_malformed() {
    let value = json!({ "kind": "scenes", "request": { "breaks": { "mode": "sideways" } } });
    let demand: AnyDemand = serde_json::from_value(value).expect("deserialises");
    let args = vec![av_toolbox_core::LayerDemand {
        layer: LayerId(0),
        demand,
    }];

    let Err(error) = crate::provider::ServeScenes::new(&args, &grey8()) else {
        panic!("a malformed scenes request should be refused");
    };

    let message = error.to_string();
    assert!(message.contains("malformed"), "{message}");
    assert!(message.contains("sideways"), "{message}");
    assert!(!message.contains("asks for a"), "{message}");
}

#[test]
fn a_seconds_limit_that_is_not_finite_is_refused() {
    for seconds in [f64::INFINITY, f64::NAN] {
        let request = SceneRequest::limited_to_seconds(seconds);

        let error = request.validate().expect_err("refused").to_string();

        assert!(error.contains("finite"), "{error}");
    }

    assert!(SceneRequest::limited_to_seconds(10.0).validate().is_ok());
    assert!(SceneRequest::natural().validate().is_ok());
}
