use std::hash::{DefaultHasher, Hash, Hasher};

use av_toolbox_core::DemandDefinition;
use serde_json::json;

use crate::scene::{SceneBreaks, SceneRequest, SceneScore, SceneSpeed, Scenes};

fn hash_of(score: SceneScore) -> u64 {
    let mut hasher = DefaultHasher::new();
    score.hash(&mut hasher);

    hasher.finish()
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
fn a_seconds_limit_that_is_not_finite_is_refused() {
    for seconds in [f64::INFINITY, f64::NAN] {
        let request = SceneRequest::limited_to_seconds(seconds);

        let error = request.validate().expect_err("refused").to_string();

        assert!(error.contains("finite"), "{error}");
    }

    assert!(SceneRequest::limited_to_seconds(10.0).validate().is_ok());
    assert!(SceneRequest::natural().validate().is_ok());
}

#[test]
fn the_slowest_speed_is_standard_when_any_asks_for_it() {
    let mixed = SceneSpeed::slowest([SceneSpeed::Fast, SceneSpeed::Standard]);
    let fast = SceneSpeed::slowest([SceneSpeed::Fast]);

    assert_eq!(mixed, SceneSpeed::Standard);
    assert_eq!(fast, SceneSpeed::Fast);
}

#[test]
fn scene_scores_with_the_same_bits_are_equal_and_hash_equal() {
    let first = SceneScore::new(2.5);
    let second = SceneScore::new(2.5);

    assert_eq!(first, second);
    assert_eq!(hash_of(first), hash_of(second));
}

#[test]
fn scene_scores_compare_by_their_bits() {
    let positive_zero = SceneScore::new(0.0);
    let negative_zero = SceneScore::new(-0.0);

    assert_ne!(positive_zero, negative_zero);
}

#[test]
fn a_nan_scene_score_equals_itself() {
    let score = SceneScore::new(f32::NAN);

    assert_eq!(score, score);
}
