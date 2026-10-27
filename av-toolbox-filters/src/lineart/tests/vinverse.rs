use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::testing::reference_vinverse;
use av_toolbox_gpu::{
    Capabilities,
    ColourInfo,
    Depth as BitDepth,
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

use crate::lineart::Vinverse;
use crate::lineart::vinverse::{VinverseArgs, vinverse_params};
use crate::sample_checks::assert_integers_and_floats_accepted_and_mask_refused;

const STAGE: &str = "lineart/vinverse";

fn args(value: serde_json::Value) -> VinverseArgs {
    serde_json::from_value(value).expect("parses")
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = vinverse_params(&parsed, SampleFormat::U8).expect_err("should be refused");

    error.to_string()
}

fn format(layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width: 16,
        height: 8,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
        field_order: None,
    }
}

fn yuv420() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    format(layout, SampleFormat::U8)
}

fn capabilities() -> Capabilities {
    let accelerator = default_accelerators()[0];

    Capabilities::builder(accelerator)
        .plane_size(32, 32)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

fn mask_format(width: u32, height: u32, sample: SampleFormat) -> StreamFormat {
    StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width,
        height,
        sample,
        chroma: None,
    })
}

#[test]
fn the_defaults_are_vs_jetpacks() {
    let parsed = args(json!({}));
    let params = vinverse_params(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!(params.contra_str, 2.7);
    assert_eq!(params.scl, 0.25);
    assert_eq!(params.amnt, None);
    assert_eq!(parsed.planes, None);
}

#[test]
fn amnt_is_scaled_to_the_bit_depth() {
    let parsed = args(json!({ "amnt": 5 }));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };

    let eight = vinverse_params(&parsed, SampleFormat::U8).expect("valid");
    let ten = vinverse_params(&parsed, ten_bit).expect("valid");

    assert_eq!(eight.amnt, Some(5.0));
    assert_eq!(ten.amnt, Some(20.0));
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "contra_str": -0.5 }), "contra_str", "-0.5"),
        (json!({ "scl": 1.5 }), "scl", "1.5"),
        (json!({ "scl": -0.1 }), "scl", "-0.1"),
        (json!({ "amnt": 256 }), "amnt", "256"),
        (json!({ "amnt": -1 }), "amnt", "-1"),
    ];

    for (value, name, expected) in cases {
        let message = refusal(value);

        assert!(
            message.contains(STAGE) && message.contains(name) && message.contains(expected),
            "{message}"
        );
    }
}

#[test]
fn nan_or_infinite_arguments_are_refused() {
    let defaults = || args(json!({}));
    let cases = [
        (
            VinverseArgs {
                contra_str: f32::NAN,
                ..defaults()
            },
            "contra_str",
            "NaN",
        ),
        (
            VinverseArgs {
                contra_str: f32::INFINITY,
                ..defaults()
            },
            "contra_str",
            "inf",
        ),
        (
            VinverseArgs {
                scl: f32::NAN,
                ..defaults()
            },
            "scl",
            "NaN",
        ),
        (
            VinverseArgs {
                amnt: Some(f32::NAN),
                ..defaults()
            },
            "amnt",
            "NaN",
        ),
    ];

    for (parsed, name, expected) in cases {
        let error = vinverse_params(&parsed, SampleFormat::U8).expect_err("should be refused");
        let message = error.to_string();

        assert!(
            message.contains(STAGE) && message.contains(name) && message.contains(expected),
            "{message}"
        );
    }
}

#[test]
fn the_bounds_themselves_are_accepted() {
    let values = [
        json!({ "contra_str": 0, "scl": 0, "amnt": 0 }),
        json!({ "scl": 1, "amnt": 255 }),
    ];

    for value in values {
        let parsed = args(value);

        assert!(vinverse_params(&parsed, SampleFormat::U8).is_ok());
    }
}

#[test]
fn a_plane_listed_twice_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u", "u"] }));
    let chosen = [Some(StreamFormat::Frames(yuv420()))];

    let error = Vinverse::produces(&capabilities, &parsed, &chosen).expect_err("u is listed twice");

    assert!(error.to_string().contains(STAGE));
}

#[test]
fn the_output_is_f32_for_every_input() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let inputs = [
        yuv420(),
        format(FrameLayout::Grey, ten_bit),
        format(chroma, ten_bit),
        format(FrameLayout::Rgb, SampleFormat::U8),
        format(FrameLayout::Grey, SampleFormat::F32),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered), None];
        let outputs = Vinverse::produces(&capabilities, &parsed, &chosen).expect("valid");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        let float = FrameFormat {
            sample: SampleFormat::F32,
            ..input
        };

        assert_eq!(output, StreamFormat::Frames(float));
    }
}

#[test]
fn floats_are_accepted_and_a_mask_on_the_input_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let accepted = |offered| Vinverse::accepts_input(&capabilities, &parsed, 0, offered, &inputs);

    assert_integers_and_floats_accepted_and_mask_refused("lineart/vinverse", layout, format, accepted);
}

#[test]
fn a_mask_of_another_size_or_depth_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420()));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let small = Some(mask_format(8, 4, SampleFormat::U8));
    let wide = Some(mask_format(16, 8, ten_bit));

    let small_error = Vinverse::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let wide_error = Vinverse::produces(&capabilities, &parsed, &[frames, wide]).expect_err("not U8");

    assert!(small_error.to_string().contains(STAGE));
    assert!(small_error.to_string().contains("8x4"));
    assert!(wide_error.to_string().contains(STAGE));
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Vinverse::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains(STAGE));
}

/// A 4x8 plane alternating 100 and 140 by row.
fn comb() -> Vec<f32> {
    (0..32)
        .map(|index| if (index / 4) % 2 == 0 { 100.0 } else { 140.0 })
        .collect()
}

#[test]
fn vinverse_removes_a_one_row_comb() {
    let input = comb();

    let output = reference_vinverse(&input, 4, 8, 2.7, 0.25, None);

    for row in 1..6 {
        let this = output[row * 4];
        let next = output[(row + 1) * 4];
        assert!((this - next).abs() < 2.0, "rows {row} and {} differ", row + 1);
    }
}

/// The comb column, where `p * q` is 0 and `out` is a flat 120.
#[test]
fn the_comb_sample_works_through_all_four_steps() {
    let input = comb();

    let output = reference_vinverse(&input, 4, 8, 2.7, 0.25, None);

    assert_eq!(output[3 * 4], 120.0);
    assert_eq!(output[2 * 4], 120.0);
}

/// The column `0 16 8 16 0`, where the middle sample takes the `scl` branch.
#[test]
fn a_correction_against_the_input_is_scaled_by_scl() {
    let input = [0.0, 16.0, 8.0, 16.0, 0.0];

    let quarter = reference_vinverse(&input, 1, 5, 2.7, 0.25, None);
    let full = reference_vinverse(&input, 1, 5, 2.7, 1.0, None);
    let limited = reference_vinverse(&input, 1, 5, 2.7, 0.25, Some(3.0));

    assert!((quarter[2] - 12.675).abs() < 1e-5, "{}", quarter[2]);
    assert!((full[2] - 14.7).abs() < 1e-5, "{}", full[2]);
    assert_eq!(limited[2], 11.0);
}

/// The column `10 10 90 10 10`, where the middle sample keeps its peak.
#[test]
fn a_spike_keeps_its_peak_and_softens_its_shoulder() {
    let input = [10.0, 10.0, 90.0, 10.0, 10.0];

    let output = reference_vinverse(&input, 1, 5, 2.7, 0.25, None);

    assert!((output[2] - 90.0).abs() < 1e-5, "{}", output[2]);
    assert!((output[1] - 23.25).abs() < 1e-5, "{}", output[1]);
    assert!((output[0] - 10.0).abs() < 1e-5, "{}", output[0]);
}

#[test]
fn amnt_limits_the_change() {
    let input = comb();

    let output = reference_vinverse(&input, 4, 8, 2.7, 0.25, Some(5.0));

    let largest_move = output
        .iter()
        .zip(&input)
        .map(|(result, value)| (result - value).abs())
        .fold(0.0f32, f32::max);
    println!("largest move under amnt 5: {largest_move}");
    assert!(largest_move <= 5.0);
    assert_eq!(largest_move, 5.0);
}

#[test]
fn a_flat_plane_is_unchanged() {
    let input = vec![77.0f32; 12];

    let output = reference_vinverse(&input, 3, 4, 2.7, 0.25, None);

    assert_eq!(output, input);
}
