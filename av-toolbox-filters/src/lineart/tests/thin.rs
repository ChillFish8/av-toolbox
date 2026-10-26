use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::neighbourhood::{Extreme, Shape, Spread, Step};
use av_toolbox_cubek::warp::BlurKind;
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

use crate::lineart::Thin;
use crate::lineart::thin::{ThinArgs, mask_steps, thin_settings};

const STAGE: &str = "lineart/thin";

fn args(value: serde_json::Value) -> ThinArgs {
    serde_json::from_value(value).expect("parses")
}

fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: BitDepth::TEN }
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = thin_settings(&parsed, SampleFormat::U8).expect_err("should be refused");

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
fn the_defaults_are_the_specs() {
    let parsed = args(json!({}));
    let settings = thin_settings(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!(settings.warp.depth, 16);
    assert_eq!(settings.warp.passes, 2);
    assert_eq!(settings.warp.thresh, 128);
    assert_eq!(settings.warp.blur, BlurKind::R6);
    assert_eq!(settings.warp.bits, 8);
    assert_eq!(settings.threshold, 20.0);
    assert_eq!(settings.radius, 1);
    assert_eq!(settings.peak, 255.0);
    assert_eq!(parsed.planes, None);
}

/// `0.08 * 1023 = 81.84` rounds to 82. The warp scales `thresh` itself, so 128 stays 128.
#[test]
fn the_mask_threshold_follows_the_bit_depth() {
    let parsed = args(json!({ "blur": 5, "depth": -7, "mrad": 3 }));

    let ten = ten_bit();
    let settings = thin_settings(&parsed, ten).expect("valid");

    assert_eq!(settings.threshold, 82.0);
    assert_eq!(settings.peak, 1023.0);
    assert_eq!(settings.warp.bits, 10);
    assert_eq!(settings.warp.thresh, 128);
    assert_eq!(settings.warp.scaled_thresh(), 512);
    assert_eq!(settings.warp.passes, 5);
    assert_eq!(settings.warp.depth, -7);
    assert_eq!(settings.radius, 3);
}

/// `0.5 * 255 = 127.5` rounds up to 128, `0.1 * 255 = 25.5` to 26 and `0.04 * 255 = 10.2` down to 10.
#[test]
fn the_mask_threshold_is_rounded_to_the_nearest_code() {
    let cases = [(0.0, 0.0), (0.04, 10.0), (0.5, 128.0), (1.0, 255.0)];

    for (mthr, expected) in cases {
        let parsed = args(json!({ "mthr": mthr }));

        let settings = thin_settings(&parsed, SampleFormat::U8).expect("valid");

        assert_eq!(settings.threshold, expected, "mthr {mthr}");
    }
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "depth": 128 }), "depth", "128"),
        (json!({ "depth": -129 }), "depth", "-129"),
        (json!({ "blur": 101 }), "blur", "101"),
        (json!({ "thresh": 256 }), "thresh", "256"),
        (json!({ "mthr": 1.5 }), "mthr", "1.5"),
        (json!({ "mthr": -0.25 }), "mthr", "-0.25"),
        (json!({ "mrad": 5 }), "mrad", "5"),
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
fn a_nan_mthr_is_refused() {
    let parsed = ThinArgs {
        mthr: f32::NAN,
        ..args(json!({}))
    };

    let error = thin_settings(&parsed, SampleFormat::U8).expect_err("should be refused");
    let message = error.to_string();

    assert!(
        message.contains(STAGE) && message.contains("mthr") && message.contains("NaN"),
        "{message}"
    );
}

#[test]
fn the_bounds_themselves_are_accepted() {
    let values = [
        json!({ "depth": -128, "blur": 0, "thresh": 0, "mthr": 0, "mrad": 0 }),
        json!({ "depth": 127, "blur": 100, "thresh": 255, "mthr": 1, "mrad": 4 }),
    ];

    for value in values {
        let parsed = args(value);

        assert!(thin_settings(&parsed, SampleFormat::U8).is_ok());
    }
}

#[test]
fn mrad_zero_skips_the_maximum() {
    let inflate = Step::Spread(Spread::Inflate);

    let steps = mask_steps(0);

    assert_eq!(steps, vec![inflate]);
}

#[test]
fn each_mrad_adds_a_square_maximum_before_the_inflate() {
    let maximum = Step::Extreme(Extreme::Maximum, Shape::Square);
    let inflate = Step::Spread(Spread::Inflate);

    for radius in 1..=4u32 {
        let steps = mask_steps(radius);

        let (last, maxima) = steps.split_last().expect("the inflate is always there");
        assert_eq!(*last, inflate, "radius {radius}");
        assert_eq!(maxima.len(), radius as usize, "radius {radius}");
        assert!(maxima.iter().all(|step| *step == maximum), "radius {radius}");
    }
}

#[test]
fn a_plane_listed_twice_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u", "u"] }));
    let input = yuv420();
    let frames = StreamFormat::Frames(input);
    let chosen = [Some(frames)];

    let error = Thin::produces(&capabilities, &parsed, &chosen).expect_err("u is listed twice");

    assert!(error.to_string().contains(STAGE));
}

#[test]
fn a_bad_argument_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "mrad": 9 }));
    let input = yuv420();
    let frames = StreamFormat::Frames(input);
    let chosen = [Some(frames)];

    let error = Thin::produces(&capabilities, &parsed, &chosen).expect_err("mrad is too large");

    assert!(error.to_string().contains(STAGE));
    assert!(error.to_string().contains("mrad"));
}

#[test]
fn the_output_format_is_the_input_format() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let ten = ten_bit();
    let inputs = [
        yuv420(),
        format(FrameLayout::Grey, ten),
        format(chroma, ten),
        format(FrameLayout::Rgb, SampleFormat::U8),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered), None];
        let outputs = Thin::produces(&capabilities, &parsed, &chosen).expect("valid");
        let sample = outputs.samples.preferences()[0];
        let output = outputs.template.with_sample(sample);

        assert_eq!(output, offered);
    }
}

#[test]
fn floats_are_left_out_and_a_mask_on_the_input_refused() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let inputs = [Offered::Unknown];
    let float_format = FrameFormat {
        sample: SampleFormat::F32,
        ..yuv420()
    };
    let float = StreamFormat::Frames(float_format);
    let mask = mask_format(16, 8, SampleFormat::U8);

    let accepted = Thin::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("readable");
    let refused = Thin::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn a_mask_of_another_size_or_depth_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let input = yuv420();
    let frames = StreamFormat::Frames(input);
    let small = mask_format(8, 4, SampleFormat::U8);
    let ten = ten_bit();
    let wide = mask_format(16, 8, ten);
    let frames = Some(frames);
    let small = Some(small);
    let wide = Some(wide);

    let small_error = Thin::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let wide_error = Thin::produces(&capabilities, &parsed, &[frames, wide]).expect_err("not U8");

    assert!(small_error.to_string().contains(STAGE));
    assert!(small_error.to_string().contains("8x4"));
    assert!(wide_error.to_string().contains(STAGE));
}
