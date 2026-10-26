use av_toolbox_core::FilterDefinition;
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

use crate::dehalo::Alpha;
use crate::dehalo::alpha::{AlphaArgs, alpha_params};

fn args(value: serde_json::Value) -> AlphaArgs {
    serde_json::from_value(value).expect("parses")
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = alpha_params(&parsed, SampleFormat::U8).expect_err("should be refused");

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

#[test]
fn the_defaults_are_havsfuncs() {
    let parsed = args(json!({}));
    let params = alpha_params(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!((params.rx, params.ry), (2.0, 2.0));
    assert_eq!((params.darkstr, params.brightstr), (1.0, 1.0));
    assert_eq!((params.lowsens, params.highsens), (50.0, 50.0));
    assert_eq!(params.ss, 1.5);
    assert_eq!(params.bits, 8);
    assert_eq!(parsed.planes, None);
}

#[test]
fn the_depth_comes_from_the_sample() {
    let parsed = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };

    let params = alpha_params(&parsed, ten_bit).expect("valid");

    assert_eq!(params.bits, 10);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "rx": 0.5 }), "rx", "0.5"),
        (json!({ "ry": 0.9 }), "ry", "0.9"),
        (json!({ "darkstr": -0.1 }), "darkstr", "-0.1"),
        (json!({ "brightstr": -1.0 }), "brightstr", "-1"),
        (json!({ "lowsens": 101.0 }), "lowsens", "101"),
        (json!({ "highsens": -1.0 }), "highsens", "-1"),
        (json!({ "ss": 0.5 }), "ss", "0.5"),
    ];

    for (value, name, expected) in cases {
        let message = refusal(value);

        assert!(
            message.contains("dehalo/alpha") && message.contains(name) && message.contains(expected),
            "{message}"
        );
    }
}

#[test]
fn non_finite_arguments_are_refused() {
    let mut nan_rx = args(json!({}));
    nan_rx.rx = f32::NAN;
    let nan_error = alpha_params(&nan_rx, SampleFormat::U8).expect_err("should be refused");
    let nan_message = nan_error.to_string();

    let mut infinite_lowsens = args(json!({}));
    infinite_lowsens.lowsens = f32::INFINITY;
    let infinite_error = alpha_params(&infinite_lowsens, SampleFormat::U8).expect_err("should be refused");
    let infinite_message = infinite_error.to_string();

    assert!(
        nan_message.contains("dehalo/alpha") && nan_message.contains("rx"),
        "{nan_message}"
    );
    assert!(
        infinite_message.contains("dehalo/alpha") && infinite_message.contains("lowsens"),
        "{infinite_message}"
    );
}

#[test]
fn the_output_format_is_the_input_format() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let ten_bit = SampleFormat::U16 { depth: BitDepth::TEN };
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let inputs = [
        yuv420(),
        format(FrameLayout::Grey, ten_bit),
        format(FrameLayout::Rgb, SampleFormat::U8),
        format(chroma, ten_bit),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered)];
        let outputs = Alpha::produces(&capabilities, &parsed, &chosen).expect("valid");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        assert_eq!(output, offered);
    }
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Alpha::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains("dehalo/alpha"));
}

#[test]
fn a_huge_supersampling_factor_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "ss": 2000.0 }));
    let input = yuv420();
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Alpha::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");
    let message = error.to_string();

    assert!(
        message.contains("dehalo/alpha") && message.contains("16x8"),
        "{message}"
    );
}

#[test]
fn floats_are_left_out_and_masks_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420()
    };
    let float = StreamFormat::Frames(float_format);
    let mask = StreamFormat::Mask(MaskFormat {
        frame_rate: None,
        width: 16,
        height: 8,
        sample: SampleFormat::U8,
        chroma: None,
    });

    let accepted =
        Alpha::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("frames are readable");
    let refused = Alpha::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}
