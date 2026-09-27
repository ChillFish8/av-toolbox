use av_toolbox_gpu::{
    Channel,
    ColourInfo,
    ColourRange,
    Depth as BitDepth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::deband::placebo::{PerPlane, PlaceboArgs, neutral, plane_settings};

fn args(value: serde_json::Value) -> PlaceboArgs {
    serde_json::from_value(value).expect("parses")
}

fn format(layout: FrameLayout, sample: SampleFormat, range: ColourRange) -> FrameFormat {
    let colour = ColourInfo {
        range,
        ..ColourInfo::default()
    };

    FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour,
        frame_rate: None,
    }
}

fn yuv420() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(layout, SampleFormat::U8, ColourRange::Limited)
}

fn refusal(value: serde_json::Value, input: &FrameFormat) -> String {
    let parsed = args(value);
    let error = plane_settings(&parsed, input).expect_err("should be refused");

    error.to_string()
}

#[test]
fn the_defaults_are_libplacebos() {
    let parsed = args(json!({}));

    assert_eq!(parsed.iterations, 1);
    assert_eq!(parsed.threshold, PerPlane::Every(3.0));
    assert_eq!(parsed.radius, 16.0);
    assert_eq!(parsed.grain, PerPlane::Every(4.0));
    assert_eq!(parsed.seed, 0);
}

#[test]
fn an_array_parses_per_plane() {
    let parsed = args(json!({ "threshold": [4.0, 3.0, 3.0], "grain": [6, 0, 0] }));

    assert_eq!(parsed.threshold, PerPlane::Each(vec![4.0, 3.0, 3.0]));
    assert_eq!(parsed.grain, PerPlane::Each(vec![6.0, 0.0, 0.0]));
}

#[test]
fn an_unknown_argument_is_refused() {
    let parsed = serde_json::from_value::<PlaceboArgs>(json!({ "strength": 2 }));

    assert!(parsed.is_err());
}

#[test]
fn a_scalar_matches_the_array_that_repeats_it() {
    let input = yuv420();

    let scalar_args = args(json!({ "threshold": 5.0, "grain": 2.0 }));
    let scalar = plane_settings(&scalar_args, &input).expect("valid");

    let array_args = args(json!({ "threshold": [5.0, 5.0, 5.0], "grain": [2.0, 2.0, 2.0] }));
    let array = plane_settings(&array_args, &input).expect("valid");

    assert_eq!(scalar, array);
}

#[test]
fn settings_follow_the_layouts_channel_order() {
    let input = yuv420();
    let parsed = args(json!({ "threshold": [4.0, 3.0, 2.0] }));
    let settings = plane_settings(&parsed, &input).expect("valid");
    let channels: Vec<Channel> = settings.iter().map(|plane| plane.channel).collect();
    let thresholds: Vec<f32> = settings.iter().map(|plane| plane.threshold).collect();

    assert_eq!(channels, vec![Channel::Y, Channel::U, Channel::V]);
    assert_eq!(thresholds, vec![4.0, 3.0, 2.0]);
}

#[test]
fn an_array_of_the_wrong_length_is_refused() {
    let input = yuv420();
    let value = json!({ "threshold": [4.0, 3.0] });
    let message = refusal(value, &input);

    assert!(message.contains("threshold"), "{message}");
    assert!(message.contains('2'), "{message}");
    assert!(message.contains('3'), "{message}");
}

#[test]
fn a_grey_clip_takes_a_one_element_array() {
    let grey = format(FrameLayout::Grey, SampleFormat::U8, ColourRange::Limited);
    let parsed = args(json!({ "grain": [6.0] }));
    let settings = plane_settings(&parsed, &grey).expect("valid");

    assert_eq!(settings.len(), 1);
    assert_eq!(settings[0].grain, 6.0);
}

#[test]
fn negative_values_are_refused() {
    let input = yuv420();

    let threshold_value = json!({ "threshold": -1.0 });
    let threshold = refusal(threshold_value, &input);

    let grain_value = json!({ "grain": [1.0, -2.0, 1.0] });
    let grain = refusal(grain_value, &input);

    assert!(
        threshold.contains("threshold") && threshold.contains("-1"),
        "{threshold}"
    );
    assert!(grain.contains("grain") && grain.contains("-2"), "{grain}");
}

#[test]
fn too_many_iterations_and_a_bad_radius_are_refused() {
    let input = yuv420();

    let iterations_value = json!({ "iterations": 17 });
    let iterations = refusal(iterations_value, &input);

    let zero_value = json!({ "radius": 0.0 });
    let zero = refusal(zero_value, &input);

    let negative_value = json!({ "radius": -4.0 });
    let negative = refusal(negative_value, &input);

    assert!(
        iterations.contains("iterations") && iterations.contains("17"),
        "{iterations}"
    );
    assert!(zero.contains("radius"), "{zero}");
    assert!(
        negative.contains("radius") && negative.contains("-4"),
        "{negative}"
    );
}

#[test]
fn sixteen_iterations_are_accepted() {
    let input = yuv420();
    let parsed = args(json!({ "iterations": 16 }));
    let settings = plane_settings(&parsed, &input);

    assert!(settings.is_ok());
}

#[test]
fn the_neutral_follows_libplacebos_renderer() {
    assert_eq!(neutral(Channel::Y, ColourRange::Limited, 8), 16.0);
    assert_eq!(neutral(Channel::Y, ColourRange::Unspecified, 8), 16.0);
    assert_eq!(neutral(Channel::Y, ColourRange::Full, 8), 0.0);
    assert_eq!(neutral(Channel::Y, ColourRange::Limited, 10), 64.0);
    assert_eq!(neutral(Channel::R, ColourRange::Limited, 8), 0.0);
    assert_eq!(neutral(Channel::G, ColourRange::Full, 10), 0.0);
    assert_eq!(neutral(Channel::U, ColourRange::Limited, 8), 128.0);
    assert_eq!(neutral(Channel::V, ColourRange::Full, 10), 512.0);
}

#[test]
fn a_chroma_clip_gets_the_chroma_neutral_on_both_planes() {
    let layout = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let input = format(layout, ten_bit, ColourRange::Limited);
    let parsed = args(json!({}));
    let settings = plane_settings(&parsed, &input).expect("valid");
    let neutrals: Vec<f32> = settings.iter().map(|plane| plane.neutral).collect();

    assert_eq!(neutrals, vec![512.0, 512.0]);
}

#[test]
fn a_plane_is_skipped_only_when_it_would_not_change() {
    let input = yuv420();
    let cases = [
        (json!({ "threshold": 0.0, "grain": 0.0 }), true),
        (json!({ "threshold": 3.0, "grain": 0.0 }), false),
        (json!({ "threshold": 0.0, "grain": 4.0 }), false),
        (json!({ "iterations": 0, "threshold": 3.0, "grain": 0.0 }), true),
        (json!({ "iterations": 0, "threshold": 3.0, "grain": 4.0 }), false),
    ];

    for (value, skipped) in cases {
        let parsed = args(value.clone());
        let settings = plane_settings(&parsed, &input).expect("valid");

        assert!(settings.iter().all(|plane| plane.skipped == skipped), "{value}");
    }
}

#[test]
fn only_the_planes_with_nothing_to_do_are_skipped() {
    let input = yuv420();
    let value = json!({ "threshold": [0.0, 3.0, 3.0], "grain": [0.0, 0.0, 0.0] });
    let parsed = args(value);
    let settings = plane_settings(&parsed, &input).expect("valid");
    let skipped: Vec<bool> = settings.iter().map(|plane| plane.skipped).collect();

    assert_eq!(skipped, vec![true, false, false]);
}
