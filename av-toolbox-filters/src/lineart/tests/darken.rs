use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::testing::reference_darken;
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

use crate::lineart::Darken;
use crate::lineart::darken::{DarkenArgs, darken_settings};

const STAGE: &str = "lineart/darken";

const SIZE: u32 = 9;

fn args(value: serde_json::Value) -> DarkenArgs {
    serde_json::from_value(value).expect("parses")
}

fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: BitDepth::TEN }
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = darken_settings(&parsed, SampleFormat::U8).expect_err("should be refused");

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
        width,
        height,
        sample,
        chroma: None,
    })
}

/// A 9x9 plane of `background` with a 1-wide vertical line of `line` down column 4.
fn line_plane(background: f32, line: f32) -> Vec<f32> {
    (0..SIZE * SIZE)
        .map(|index| if index % SIZE == 4 { line } else { background })
        .collect()
}

/// Darkens `plane` with the settings `value` gives at `sample`.
fn darkened(value: serde_json::Value, sample: SampleFormat, plane: &[f32]) -> Vec<f32> {
    let parsed = args(value);
    let settings = darken_settings(&parsed, sample).expect("valid");

    reference_darken(plane, SIZE, SIZE, &settings)
}

/// The sample of a 9x9 plane at `column` on the middle row.
fn at_column(plane: &[f32], column: u32) -> f32 {
    plane[(4 * SIZE + column) as usize]
}

#[test]
fn the_defaults_are_vs_jetpacks() {
    let parsed = args(json!({}));
    let settings = darken_settings(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!(settings.s, 0.375);
    assert_eq!(settings.limit, 42.5);
    assert_eq!(settings.cap, 191.0);
    assert_eq!(settings.thr, 4.0);
    assert_eq!(settings.t, 0.0);
    assert_eq!(settings.n, 128.0);
    assert_eq!(settings.peak, 255.0);
    assert_eq!(parsed.planes, None);
}

#[test]
fn the_thresholds_are_scaled_to_the_bit_depth() {
    let parsed = args(json!({ "thinning": 64 }));

    let settings = darken_settings(&parsed, ten_bit()).expect("valid");

    assert_eq!(settings.cap, 764.0);
    assert_eq!(settings.thr, 16.0);
    assert_eq!(settings.n, 512.0);
    assert_eq!(settings.peak, 1023.0);
    assert_eq!(settings.limit, 170.5);
    assert_eq!(settings.t, 4.0);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "strength": 257 }), "strength", "257"),
        (json!({ "strength": -1 }), "strength", "-1"),
        (json!({ "protection": 51 }), "protection", "51"),
        (json!({ "protection": -1 }), "protection", "-1"),
        (json!({ "luma_cap": 256 }), "luma_cap", "256"),
        (json!({ "luma_cap": -1 }), "luma_cap", "-1"),
        (json!({ "threshold": 256 }), "threshold", "256"),
        (json!({ "threshold": -1 }), "threshold", "-1"),
        (json!({ "thinning": 257 }), "thinning", "257"),
        (json!({ "thinning": -1 }), "thinning", "-1"),
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
fn nan_arguments_are_refused() {
    let defaults = || args(json!({}));
    let cases = [
        (
            DarkenArgs {
                strength: f32::NAN,
                ..defaults()
            },
            "strength",
        ),
        (
            DarkenArgs {
                protection: f32::NAN,
                ..defaults()
            },
            "protection",
        ),
        (
            DarkenArgs {
                luma_cap: f32::NAN,
                ..defaults()
            },
            "luma_cap",
        ),
        (
            DarkenArgs {
                threshold: f32::NAN,
                ..defaults()
            },
            "threshold",
        ),
        (
            DarkenArgs {
                thinning: f32::NAN,
                ..defaults()
            },
            "thinning",
        ),
    ];

    for (parsed, name) in cases {
        let error = darken_settings(&parsed, SampleFormat::U8).expect_err("should be refused");
        let message = error.to_string();

        assert!(
            message.contains(STAGE) && message.contains(name) && message.contains("NaN"),
            "{message}"
        );
    }
}

#[test]
fn the_bounds_themselves_are_accepted() {
    let values = [
        json!({ "strength": 0, "protection": 0, "luma_cap": 0, "threshold": 0, "thinning": 0 }),
        json!({ "strength": 256, "protection": 50, "luma_cap": 255, "threshold": 255, "thinning": 256 }),
    ];

    for value in values {
        let parsed = args(value);

        assert!(darken_settings(&parsed, SampleFormat::U8).is_ok());
    }
}

#[test]
fn a_plane_listed_twice_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u", "u"] }));
    let chosen = [Some(StreamFormat::Frames(yuv420()))];

    let error = Darken::produces(&capabilities, &parsed, &chosen).expect_err("u is listed twice");

    assert!(error.to_string().contains(STAGE));
}

#[test]
fn a_plane_the_layout_lacks_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u"] }));
    let input = format(FrameLayout::Grey, SampleFormat::U8);
    let chosen = [Some(StreamFormat::Frames(input))];

    let error = Darken::produces(&capabilities, &parsed, &chosen).expect_err("should be refused");

    assert!(error.to_string().contains(STAGE));
}

#[test]
fn the_output_format_is_the_input_format() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let chroma = FrameLayout::Chroma {
        subsampling: Subsampling::Yuv420,
    };
    let inputs = [
        yuv420(),
        format(FrameLayout::Grey, ten_bit()),
        format(chroma, ten_bit()),
        format(FrameLayout::Rgb, SampleFormat::U8),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered), None];
        let outputs = Darken::produces(&capabilities, &parsed, &chosen).expect("valid");
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

    let accepted = Darken::accepts_input(&capabilities, &parsed, 0, float, &inputs).expect("readable");
    let refused = Darken::accepts_input(&capabilities, &parsed, 0, mask, &inputs);

    assert!(accepted.is_empty());
    assert!(refused.is_err());
}

#[test]
fn a_mask_of_another_size_or_depth_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({}));
    let frames = Some(StreamFormat::Frames(yuv420()));
    let small = Some(mask_format(8, 4, SampleFormat::U8));
    let wide = Some(mask_format(16, 8, ten_bit()));

    let small_error = Darken::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let wide_error = Darken::produces(&capabilities, &parsed, &[frames, wide]).expect_err("not U8");

    assert!(small_error.to_string().contains(STAGE));
    assert!(small_error.to_string().contains("8x4"));
    assert!(wide_error.to_string().contains(STAGE));
}

/// The closing is `min(192.5, 191) = 191`, so the line becomes `150 + (150 - 191) * 0.375 = 134.625`.
#[test]
fn darken_deepens_a_faded_line() {
    let plane = line_plane(200.0, 150.0);

    let output = darkened(json!({}), SampleFormat::U8, &plane);

    assert!(
        (at_column(&output, 4) - 134.625).abs() < 1e-4,
        "{}",
        at_column(&output, 4)
    );
}

/// Beside the line the closing is 191, which is not above `200 + 4`.
#[test]
fn the_background_beside_a_line_is_left_alone() {
    let plane = line_plane(200.0, 150.0);

    let output = darkened(json!({}), SampleFormat::U8, &plane);

    for column in [0, 1, 2, 3, 5, 6, 7, 8] {
        assert_eq!(at_column(&output, column), 200.0, "column {column}");
    }
}

/// A cap of 160 gives `150 + (150 - 160) * 0.375 = 146.25`, and a cap of 255 gives 134.0625.
#[test]
fn the_luma_cap_limits_the_closing() {
    let plane = line_plane(200.0, 150.0);

    let capped = darkened(json!({ "luma_cap": 160 }), SampleFormat::U8, &plane);
    let uncapped = darkened(json!({ "luma_cap": 255 }), SampleFormat::U8, &plane);

    assert!(
        (at_column(&capped, 4) - 146.25).abs() < 1e-4,
        "{}",
        at_column(&capped, 4)
    );
    assert!(
        (at_column(&uncapped, 4) - 134.0625).abs() < 1e-4,
        "{}",
        at_column(&uncapped, 4)
    );
}

/// A closing 1 code below, on and 1 code above `x + thr = 154`. Only the last is a line.
#[test]
fn a_closing_is_a_line_only_when_above_the_threshold() {
    let plane = line_plane(200.0, 150.0);

    let below = darkened(json!({ "luma_cap": 153 }), SampleFormat::U8, &plane);
    let on = darkened(json!({ "luma_cap": 154 }), SampleFormat::U8, &plane);
    let above = darkened(json!({ "luma_cap": 155 }), SampleFormat::U8, &plane);

    assert_eq!(at_column(&below, 4), 150.0);
    assert_eq!(at_column(&on, 4), 150.0);
    assert!(
        (at_column(&above, 4) - 148.125).abs() < 1e-4,
        "{}",
        at_column(&above, 4)
    );
}

/// With `protection = 50`, `limit = 5` and the closing is 155, so `150 + (150 - 155) * 0.375 = 148.125`.
#[test]
fn protection_limits_the_expansion() {
    let plane = line_plane(200.0, 150.0);

    let output = darkened(json!({ "protection": 50 }), SampleFormat::U8, &plane);

    assert!(
        (at_column(&output, 4) - 148.125).abs() < 1e-4,
        "{}",
        at_column(&output, 4)
    );
}

/// With `protection = 50` the closing of 155 is 1 above `x + thr` at a threshold of 4, and on it at 5.
#[test]
fn protection_straddles_the_threshold() {
    let plane = line_plane(200.0, 150.0);

    let above = darkened(
        json!({ "protection": 50, "threshold": 4 }),
        SampleFormat::U8,
        &plane,
    );
    let on = darkened(
        json!({ "protection": 50, "threshold": 5 }),
        SampleFormat::U8,
        &plane,
    );

    assert!(at_column(&above, 4) < 150.0);
    assert_eq!(at_column(&on, 4), 150.0);
}

/// At 10 bits `thr = 16`. A line at 747 sits 17 below `cap = 764` and a line at 748 sits 16 below it.
#[test]
fn the_threshold_is_scaled_at_ten_bits() {
    let above = line_plane(800.0, 747.0);
    let on = line_plane(800.0, 748.0);

    let deepened = darkened(json!({}), ten_bit(), &above);
    let kept = darkened(json!({}), ten_bit(), &on);

    assert!(
        (at_column(&deepened, 4) - 740.625).abs() < 1e-3,
        "{}",
        at_column(&deepened, 4)
    );
    assert_eq!(at_column(&kept, 4), 748.0);
}

/// With `thin` 143.625 and `linemask` 91, `out = 143.625 + (134.625 - 143.625) * 91 / 255`.
#[test]
fn thinning_takes_the_thin_branch() {
    let plane = line_plane(200.0, 150.0);

    let sharp = darkened(json!({}), SampleFormat::U8, &plane);
    let thinned = darkened(json!({ "thinning": 64 }), SampleFormat::U8, &plane);

    assert!((at_column(&sharp, 4) - 134.625).abs() < 1e-4);
    assert!(
        (at_column(&thinned, 4) - 140.413_24).abs() < 1e-4,
        "{}",
        at_column(&thinned, 4)
    );
}

#[test]
fn thinning_leaves_the_background_beside_a_line_alone() {
    let plane = line_plane(200.0, 150.0);

    let thinned = darkened(json!({ "thinning": 64 }), SampleFormat::U8, &plane);

    for column in [0, 1, 2, 3, 5, 6, 7, 8] {
        assert_eq!(at_column(&thinned, column), 200.0, "column {column}");
    }
}

#[test]
fn a_flat_plane_is_unchanged() {
    let plane = vec![77.0f32; (SIZE * SIZE) as usize];

    let plain = darkened(json!({}), SampleFormat::U8, &plane);
    let thinned = darkened(json!({ "thinning": 64 }), SampleFormat::U8, &plane);

    assert_eq!(plain, plane);
    assert_eq!(thinned, plane);
}
