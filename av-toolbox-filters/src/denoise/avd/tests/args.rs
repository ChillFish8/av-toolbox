use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    MotionCompensationMode,
    MotionSearch,
    NlmeansAlgorithm,
    PrefilterMode,
};
use av_toolbox_core::{Parallelism, StageDefinition};
use av_toolbox_providers::{SceneRequest, Scenes};
use serde_json::json;

use crate::denoise::avd::Denoise;
use crate::denoise::avd::args::{AvdArgs, EngineOptions, Nl4dArgs, NlmeansArgs, NlmeansHqArgs};

fn parsed<A: AvdArgs>(value: serde_json::Value) -> A {
    serde_json::from_value(value).expect("the args should parse")
}

fn nlmeans_algorithm<A: AvdArgs>(args: &A, channels: ChannelMode) -> NlmeansAlgorithm {
    let options = args.options(channels).expect("the options should resolve");
    let EngineOptions::Nlmeans(algorithm) = options else {
        panic!("expected nlmeans options");
    };

    algorithm
}

fn nl4d_options(args: &Nl4dArgs, channels: ChannelMode) -> av_denoise_core::Nl4dOptions {
    let options = args.options(channels).expect("the options should resolve");
    let EngineOptions::Nl4d(options) = options else {
        panic!("expected nl4d options");
    };

    options
}

#[test]
fn nlmeans_defaults_to_the_base_preset() {
    let args: NlmeansArgs = parsed(json!({}));

    let NlmeansAlgorithm::Fast(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };

    assert_eq!(options.mode, DenoisingMode::Temporal { radius: 2 });
    assert_eq!(options.tuning.search_radius, Some(2));
    assert_eq!(options.tuning.patch_radius, None);
    assert_eq!(options.tuning.strength, None);
    assert_eq!(options.prefilter, PrefilterMode::None);
    assert_eq!(options.motion_compensation, MotionCompensationMode::None);
    assert_eq!(args.held_frames(), 2);
}

#[test]
fn veryfast_nlmeans_is_spatial() {
    let args: NlmeansArgs = parsed(json!({ "preset": "veryfast" }));

    let NlmeansAlgorithm::Fast(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };

    assert_eq!(options.mode, DenoisingMode::Spacial);
    assert_eq!(args.held_frames(), 0);
}

#[test]
fn explicit_radii_win_over_the_preset() {
    let args: NlmeansArgs = parsed(json!({
        "preset": "veryslow",
        "temporal_radius": 1,
        "search_radius": 3,
        "patch_radius": 2,
    }));

    let NlmeansAlgorithm::Fast(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };

    assert_eq!(options.mode, DenoisingMode::Temporal { radius: 1 });
    assert_eq!(options.tuning.search_radius, Some(3));
    assert_eq!(options.tuning.patch_radius, Some(2));
    assert_eq!(args.held_frames(), 1);
}

#[test]
fn per_plane_strength_wins_over_strength() {
    let args: NlmeansArgs = parsed(json!({ "strength": 1.5, "chroma_strength": 0.5 }));

    let NlmeansAlgorithm::Fast(luma) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };
    let NlmeansAlgorithm::Fast(chroma) = nlmeans_algorithm(&args, ChannelMode::Chroma) else {
        panic!("nlmeans should run the fast variant");
    };
    let NlmeansAlgorithm::Fast(fused) = nlmeans_algorithm(&args, ChannelMode::Yuv) else {
        panic!("nlmeans should run the fast variant");
    };

    assert_eq!(luma.tuning.strength, Some(1.5));
    assert_eq!(chroma.tuning.strength, Some(0.5));
    assert_eq!(fused.tuning.strength, Some(1.5));
}

#[test]
fn motion_compensation_uses_the_default_search() {
    let args: NlmeansArgs = parsed(json!({ "motion_compensation": true }));

    let NlmeansAlgorithm::Fast(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };

    let expected = MotionCompensationMode::from(MotionSearch::default());
    assert_eq!(options.motion_compensation, expected);
}

#[test]
fn a_prefilter_is_parsed_with_av_denoise_syntax() {
    let args: NlmeansArgs = parsed(json!({ "prefilter": "bilateral:2.0,0.1" }));

    let NlmeansAlgorithm::Fast(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans should run the fast variant");
    };

    let expected = av_denoise_core::parse_prefilter("bilateral:2.0,0.1").expect("a valid prefilter");
    assert_eq!(options.prefilter, expected);
}

#[test]
fn a_bad_prefilter_is_refused_naming_the_stage() {
    let args: NlmeansArgs = parsed(json!({ "prefilter": "sharpen" }));

    let error = args.options(ChannelMode::Luma).expect_err("should refuse");

    assert!(error.to_string().contains("av-denoise/nlmeans"), "{error}");
}

#[test]
fn nlmeans_hq_carries_sigma() {
    let args: NlmeansHqArgs = parsed(json!({ "sigma": 0.02, "sigma_scale": 1.3 }));

    let NlmeansAlgorithm::Hq(options) = nlmeans_algorithm(&args, ChannelMode::Luma) else {
        panic!("nlmeans-hq should run the hq variant");
    };

    assert_eq!(options.hq.sigma_override, Some(0.02));
    assert_eq!(options.hq.sigma_scale, 1.3);
    assert!(!options.hq.windowed_noise_estimation);
    assert_eq!(options.nlm.mode, DenoisingMode::Temporal { radius: 2 });
}

#[test]
fn nlmeans_refuses_sigma() {
    let parsed: Result<NlmeansArgs, _> = serde_json::from_value(json!({ "sigma": 0.02 }));

    assert!(parsed.is_err());
}

#[test]
fn nl4d_follows_its_preset() {
    let args: Nl4dArgs = parsed(json!({ "preset": "veryfast" }));

    let options = nl4d_options(&args, ChannelMode::Luma);

    assert_eq!(options.temporal_radius, 1);
    assert_eq!(options.spatial_radius, 6);
    assert!(!options.grain_export);
    assert!(!options.windowed_noise_estimation);
    assert_eq!(args.held_frames(), 2);
}

#[test]
fn nl4d_holds_twice_its_radius() {
    let args: Nl4dArgs = parsed(json!({ "temporal_radius": 3 }));

    assert_eq!(args.held_frames(), 6);
}

#[test]
fn per_plane_lambda_wins_over_lambda() {
    let args: Nl4dArgs = parsed(json!({ "lambda_ht": 3.0, "luma_lambda_ht": 5.0 }));

    let luma = nl4d_options(&args, ChannelMode::Luma);
    let chroma = nl4d_options(&args, ChannelMode::Chroma);

    assert_eq!(luma.lambda_ht, Some(5.0));
    assert_eq!(chroma.lambda_ht, Some(3.0));
}

#[test]
fn an_unset_lambda_stays_unset_for_av_denoise_to_resolve() {
    let args: Nl4dArgs = parsed(json!({}));

    let chroma = nl4d_options(&args, ChannelMode::Chroma);

    assert_eq!(chroma.lambda_ht, None);
}

#[test]
fn nl4d_refuses_strength() {
    let parsed: Result<Nl4dArgs, _> = serde_json::from_value(json!({ "strength": 1.0 }));

    assert!(parsed.is_err());
}

#[test]
fn an_unknown_preset_is_refused() {
    let parsed: Result<NlmeansArgs, _> = serde_json::from_value(json!({ "preset": "ultra" }));

    assert!(parsed.is_err());
}

#[test]
fn every_engine_chunks_on_natural_scenes() {
    let chunked = Parallelism::chunk::<Scenes>(SceneRequest::natural());

    let nlmeans: NlmeansArgs = parsed(json!({}));
    let nlmeans_hq: NlmeansHqArgs = parsed(json!({}));
    let nl4d: Nl4dArgs = parsed(json!({}));

    assert_eq!(Denoise::<NlmeansArgs>::parallelism(&nlmeans), chunked);
    assert_eq!(Denoise::<NlmeansHqArgs>::parallelism(&nlmeans_hq), chunked);
    assert_eq!(Denoise::<Nl4dArgs>::parallelism(&nl4d), chunked);
}

#[test]
fn every_engine_asks_for_two_workers() {
    let nlmeans: NlmeansArgs = parsed(json!({}));
    let nlmeans_hq: NlmeansHqArgs = parsed(json!({}));
    let nl4d: Nl4dArgs = parsed(json!({}));

    assert_eq!(Denoise::<NlmeansArgs>::workers(&nlmeans), Some(2));
    assert_eq!(Denoise::<NlmeansHqArgs>::workers(&nlmeans_hq), Some(2));
    assert_eq!(Denoise::<Nl4dArgs>::workers(&nl4d), Some(2));
}
