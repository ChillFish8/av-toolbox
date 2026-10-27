use av_toolbox_core::FilterDefinition;
use av_toolbox_gpu::{
    Capabilities,
    Channel,
    ColourInfo,
    ColourRange,
    Depth as BitDepth,
    FormatSet,
    FrameFormat,
    FrameLayout,
    MaskFormat,
    Offered,
    SampleFormat,
    StreamFormat,
    Subsampling,
    default_accelerators,
};
use serde_json::json;

use crate::deband::Placebo;
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
        field_order: None,
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
    assert!(message.contains("got 2 for 3 planes"), "{message}");
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
fn a_non_finite_threshold_is_refused() {
    let input = yuv420();
    let args = PlaceboArgs {
        iterations: 1,
        threshold: PerPlane::Every(f32::NAN),
        radius: 16.0,
        grain: PerPlane::Every(4.0),
        seed: 0,
    };
    let error = plane_settings(&args, &input).expect_err("should be refused");

    assert!(error.to_string().contains("threshold"), "{error}");
}

#[test]
fn a_non_finite_grain_element_is_refused() {
    let input = yuv420();
    let args = PlaceboArgs {
        iterations: 1,
        threshold: PerPlane::Every(3.0),
        radius: 16.0,
        grain: PerPlane::Each(vec![4.0, f32::INFINITY, 4.0]),
        seed: 0,
    };
    let error = plane_settings(&args, &input).expect_err("should be refused");

    assert!(error.to_string().contains("grain"), "{error}");
}

#[test]
fn a_non_finite_radius_is_refused() {
    let input = yuv420();
    let args = PlaceboArgs {
        iterations: 1,
        threshold: PerPlane::Every(3.0),
        radius: f32::INFINITY,
        grain: PerPlane::Every(4.0),
        seed: 0,
    };
    let error = plane_settings(&args, &input).expect_err("should be refused");

    assert!(error.to_string().contains("radius"), "{error}");
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
        let cloned_value = value.clone();
        let parsed = args(cloned_value);
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

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn produced(
    input: FrameFormat,
    value: serde_json::Value,
) -> Result<StreamFormat, av_toolbox_core::StageError> {
    let capabilities = capabilities();
    let arguments = args(value);
    let chosen = [Some(StreamFormat::Frames(input))];
    let outputs = Placebo::produces(&capabilities, &arguments, &chosen)?;
    let sample = outputs.samples.preferences()[0];

    Ok(outputs.template.with_sample(sample))
}

#[test]
fn the_output_format_is_the_input_format_in_f32() {
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let inputs = [
        yuv420(),
        format(FrameLayout::Rgb, ten_bit, ColourRange::Full),
        format(FrameLayout::Grey, SampleFormat::U8, ColourRange::Limited),
        format(chroma, ten_bit, ColourRange::Limited),
    ];

    for input in inputs {
        let output = produced(input, json!({})).expect("valid");
        let expected = FrameFormat {
            sample: SampleFormat::F32,
            ..input
        };

        assert_eq!(output, StreamFormat::Frames(expected));
    }
}

#[test]
fn the_output_is_f32_for_every_input() {
    let capabilities = capabilities();
    let arguments = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let samples = [SampleFormat::U8, ten_bit, SampleFormat::F32];

    for sample in samples {
        let input = format(FrameLayout::Grey, sample, ColourRange::Limited);
        let chosen = [Some(StreamFormat::Frames(input))];

        let outputs = Placebo::produces(&capabilities, &arguments, &chosen).expect("valid");

        assert_eq!(outputs.samples, FormatSet::of([SampleFormat::F32]), "{sample}");
    }
}

#[test]
fn every_plane_skipped_keeps_the_input_sample() {
    let capabilities = capabilities();
    let arguments = args(json!({ "threshold": 0.0, "grain": 0.0 }));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let samples = [SampleFormat::U8, ten_bit, SampleFormat::F32];

    for sample in samples {
        let input = format(FrameLayout::Grey, sample, ColourRange::Limited);
        let chosen = [Some(StreamFormat::Frames(input))];

        let outputs = Placebo::produces(&capabilities, &arguments, &chosen).expect("valid");

        assert_eq!(outputs.samples, FormatSet::of([sample]), "{sample}");
    }
}

#[test]
fn a_float_clip_gets_its_labels_neutral_at_sixteen_bits() {
    let limited = format(yuv420().layout, SampleFormat::F32, ColourRange::Limited);
    let parsed = args(json!({}));
    let settings = plane_settings(&parsed, &limited).expect("valid");
    let neutrals: Vec<f32> = settings.iter().map(|plane| plane.neutral).collect();

    assert_eq!(neutrals, vec![4096.0, 32768.0, 32768.0]);
}

#[test]
fn produces_refuses_bad_arguments() {
    let input = yuv420();
    let refused = produced(input, json!({ "threshold": [1.0] }));

    assert!(refused.is_err());
}

#[test]
fn floats_are_accepted_and_masks_refused() {
    let capabilities = capabilities();
    let arguments = args(json!({}));
    let inputs = [Offered::Unknown];
    let input = yuv420();
    let float = StreamFormat::Frames(FrameFormat {
        sample: SampleFormat::F32,
        ..input
    });
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted =
        Placebo::accepts_input(&capabilities, &arguments, 0, float, &inputs).expect("frames are readable");
    let refused = Placebo::accepts_input(&capabilities, &arguments, 0, mask, &inputs);

    assert_eq!(accepted, FormatSet::of([SampleFormat::F32]));
    assert!(refused.is_err());
}
