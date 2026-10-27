use av_toolbox_core::FilterDefinition;
use av_toolbox_cubek::lineart::ToonSettings;
use av_toolbox_cubek::testing::reference_toon;
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

use crate::lineart::Toon;
use crate::lineart::toon::{ToonArgs, toon_settings};
use crate::sample_checks::assert_integers_and_floats_accepted_and_mask_refused;

const STAGE: &str = "lineart/toon";

/// The distance between two lines in `lines_plane`, which keeps their 3x3 windows apart.
const SPACING: u32 = 3;

const ROWS: u32 = 3;

fn args(value: serde_json::Value) -> ToonArgs {
    serde_json::from_value(value).expect("parses")
}

fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: BitDepth::TEN }
}

fn refusal(value: serde_json::Value) -> String {
    let parsed = args(value);
    let error = toon_settings(&parsed, SampleFormat::U8).expect_err("should be refused");

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

/// A plane of `background` with a 1-wide vertical line down column `SPACING * k + 1` for each value
/// `k` of `lines`, `ROWS` rows tall.
fn lines_plane(background: f32, lines: &[f32]) -> (Vec<f32>, u32) {
    let width = SPACING * lines.len() as u32;
    let plane = (0..width * ROWS)
        .map(|index| {
            let column = index % width;
            if column % SPACING != 1 {
                return background;
            }

            lines[(column / SPACING) as usize]
        })
        .collect();

    (plane, width)
}

/// Deepens `plane` with the settings `value` gives at `sample`, on the reference.
fn toned(value: serde_json::Value, sample: SampleFormat, plane: &[f32], width: u32) -> Vec<f32> {
    let parsed = args(value);
    let settings = toon_settings(&parsed, sample).expect("valid");

    reference_toon(plane, width, ROWS, &settings)
}

fn settings_at_depth(depth: i32) -> ToonSettings {
    let parsed = args(json!({ "depth": depth }));

    toon_settings(&parsed, SampleFormat::U8).expect("valid")
}

/// The middle row of a plane, at the columns of its lines.
fn line_samples(plane: &[f32], width: u32) -> Vec<f32> {
    let row = (ROWS / 2 * width) as usize;

    (0..width / SPACING)
        .map(|line| plane[row + (SPACING * line + 1) as usize])
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());

    for (index, (found, wanted)) in actual.iter().zip(expected).enumerate() {
        assert!((found - wanted).abs() < 1e-3, "{index}: {found} is not {wanted}");
    }
}

#[test]
fn the_defaults_are_havsfuncs() {
    let parsed = args(json!({}));
    let settings = toon_settings(&parsed, SampleFormat::U8).expect("valid");

    assert_eq!(settings.strength, 1.0);
    assert_eq!(settings.l, 2.0);
    assert_eq!(settings.u, 12.0);
    assert_eq!(settings.n, 128.0);
    assert_eq!(settings.peak, 255.0);
    assert_eq!(settings.warp.thresh, 128);
    assert_eq!(settings.warp.blur, BlurKind::R6);
    assert_eq!(settings.warp.passes, 2);
    assert_eq!(settings.warp.depth, 32);
    assert_eq!(settings.warp.bits, 8);
    assert_eq!(parsed.planes, None);
}

#[test]
fn the_thresholds_are_scaled_to_the_bit_depth() {
    let parsed = args(json!({ "l_thr": 3, "u_thr": 20, "blur": 5, "depth": -7 }));

    let ten = ten_bit();
    let settings = toon_settings(&parsed, ten).expect("valid");

    assert_eq!(settings.l, 12.0);
    assert_eq!(settings.u, 80.0);
    assert_eq!(settings.n, 512.0);
    assert_eq!(settings.peak, 1023.0);
    assert_eq!(settings.warp.bits, 10);
    assert_eq!(settings.warp.passes, 5);
    assert_eq!(settings.warp.depth, -7);
}

#[test]
fn out_of_range_arguments_are_refused() {
    let cases = [
        (json!({ "strength": -1 }), "strength", "-1"),
        (json!({ "l_thr": 256, "u_thr": 255 }), "l_thr", "256"),
        (json!({ "l_thr": -1 }), "l_thr", "-1"),
        (json!({ "u_thr": 256 }), "u_thr", "256"),
        (json!({ "u_thr": -1 }), "u_thr", "-1"),
        (json!({ "blur": 101 }), "blur", "101"),
        (json!({ "depth": 128 }), "depth", "128"),
        (json!({ "depth": -129 }), "depth", "-129"),
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
fn thresholds_that_are_not_ordered_are_refused() {
    let cases = [
        (json!({ "l_thr": 12, "u_thr": 12 }), "12"),
        (json!({ "l_thr": 13, "u_thr": 12 }), "13"),
        (json!({ "l_thr": 0, "u_thr": 0 }), "0"),
    ];

    for (value, expected) in cases {
        let message = refusal(value);

        assert!(
            message.contains(STAGE)
                && message.contains("l_thr")
                && message.contains("u_thr")
                && message.contains(expected),
            "{message}"
        );
    }
}

#[test]
fn nan_arguments_are_refused() {
    let defaults = || args(json!({}));
    let cases = [
        (
            ToonArgs {
                strength: f32::NAN,
                ..defaults()
            },
            "strength",
        ),
        (
            ToonArgs {
                l_thr: f32::NAN,
                ..defaults()
            },
            "l_thr",
        ),
        (
            ToonArgs {
                u_thr: f32::NAN,
                ..defaults()
            },
            "u_thr",
        ),
    ];

    for (parsed, name) in cases {
        let error = toon_settings(&parsed, SampleFormat::U8).expect_err("should be refused");
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
        json!({ "strength": 0, "l_thr": 0, "u_thr": 1, "blur": 0, "depth": -128 }),
        json!({ "strength": 100, "l_thr": 254, "u_thr": 255, "blur": 100, "depth": 127 }),
    ];

    for value in values {
        let parsed = args(value);

        assert!(toon_settings(&parsed, SampleFormat::U8).is_ok());
    }
}

#[test]
fn a_plane_listed_twice_is_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "planes": ["u", "u"] }));
    let input = yuv420();
    let frames = StreamFormat::Frames(input);
    let chosen = [Some(frames)];

    let error = Toon::produces(&capabilities, &parsed, &chosen).expect_err("u is listed twice");

    assert!(error.to_string().contains(STAGE));
}

#[test]
fn unordered_thresholds_are_refused_through_produces() {
    let capabilities = capabilities();
    let parsed = args(json!({ "l_thr": 20 }));
    let input = yuv420();
    let frames = StreamFormat::Frames(input);
    let chosen = [Some(frames)];

    let error = Toon::produces(&capabilities, &parsed, &chosen).expect_err("l_thr is above u_thr");

    assert!(error.to_string().contains(STAGE));
    assert!(error.to_string().contains("l_thr"));
}

#[test]
fn the_output_is_f32_for_every_input() {
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
        format(FrameLayout::Grey, SampleFormat::F32),
    ];

    for input in inputs {
        let offered = StreamFormat::Frames(input);
        let chosen = [Some(offered), None];
        let outputs = Toon::produces(&capabilities, &parsed, &chosen).expect("valid");
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
    let accepted = |offered| Toon::accepts_input(&capabilities, &parsed, 0, offered, &inputs);

    assert_integers_and_floats_accepted_and_mask_refused("lineart/toon", layout, format, accepted);
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

    let small_error = Toon::produces(&capabilities, &parsed, &[frames, small]).expect_err("too small");
    let wide_error = Toon::produces(&capabilities, &parsed, &[frames, wide]).expect_err("not U8");

    assert!(small_error.to_string().contains(STAGE));
    assert!(small_error.to_string().contains("8x4"));
    assert!(wide_error.to_string().contains(STAGE));
}

/// At depth 0, `w = d`. Lines 1, 2, 3, 6, 11, 12 and 13 below 100 give 99, 98, 96.7, 91.6, 79.1, 76 and 74.
#[test]
fn toon_interpolates_between_the_thresholds() {
    let (plane, width) = lines_plane(100.0, &[99.0, 98.0, 97.0, 94.0, 89.0, 88.0, 87.0]);

    let output = toned(json!({ "depth": 0 }), SampleFormat::U8, &plane, width);

    let lines = line_samples(&output, width);
    assert_close(&lines, &[99.0, 98.0, 96.7, 91.6, 79.1, 76.0, 74.0]);
    for (index, value) in output.iter().enumerate() {
        let column = index as u32 % width;
        if column % SPACING != 1 {
            assert_eq!(*value, 100.0, "column {column}");
        }
    }
}

/// At 10 bits `l = 8`, `u = 48`. Lines 7, 8, 9, 40, 48, 49 below 800 give 793, 792, 790.775, 728, 704, 702.
#[test]
fn toon_scales_its_thresholds_at_ten_bits() {
    let (plane, width) = lines_plane(800.0, &[793.0, 792.0, 791.0, 760.0, 752.0, 751.0]);

    let ten = ten_bit();
    let output = toned(json!({ "depth": 0 }), ten, &plane, width);

    let lines = line_samples(&output, width);
    assert_close(&lines, &[793.0, 792.0, 790.775, 728.0, 704.0, 702.0]);
}

/// Lines of 97 and 90 side by side have `d` of 3 and 10 and share `m = 10`, giving 94.6 and 82.
#[test]
fn a_faint_sample_beside_a_dark_one_takes_the_darker_map() {
    let mut plane = vec![100.0f32; 9 * 3];
    for row in 0..3 {
        plane[row * 9 + 4] = 97.0;
        plane[row * 9 + 5] = 90.0;
    }

    let output = toned(json!({ "depth": 0 }), SampleFormat::U8, &plane, 9);

    let middle = 9 + 3;
    assert_close(&output[middle..middle + 4], &[100.0, 94.6, 82.0, 100.0]);
}

/// A line of 88 in 100 has `d = 12`, so 82 at a strength of 0.5 and 64 at 2.
#[test]
fn strength_scales_the_darkening() {
    let (plane, width) = lines_plane(100.0, &[88.0]);

    let half = toned(
        json!({ "depth": 0, "strength": 0.5 }),
        SampleFormat::U8,
        &plane,
        width,
    );
    let double = toned(
        json!({ "depth": 0, "strength": 2 }),
        SampleFormat::U8,
        &plane,
        width,
    );
    let none = toned(
        json!({ "depth": 0, "strength": 0 }),
        SampleFormat::U8,
        &plane,
        width,
    );

    let half_lines = line_samples(&half, width);
    let double_lines = line_samples(&double, width);
    assert_close(&half_lines, &[82.0]);
    assert_close(&double_lines, &[64.0]);
    assert_eq!(none, plane);
}

/// A 100 line in 250 has `d = 150`, held to `peak - n = 127`, so -27. At 10 bits 900 is held to 511, so -411.
#[test]
fn the_darkness_map_is_held_below_the_neutral_range() {
    let (eight, eight_width) = lines_plane(250.0, &[100.0]);
    let (ten, ten_width) = lines_plane(1000.0, &[100.0]);

    let eight_output = toned(json!({ "depth": 0 }), SampleFormat::U8, &eight, eight_width);
    let ten_bits = ten_bit();
    let ten_output = toned(json!({ "depth": 0 }), ten_bits, &ten, ten_width);

    let eight_lines = line_samples(&eight_output, eight_width);
    let ten_lines = line_samples(&ten_output, ten_width);
    assert_close(&eight_lines, &[-27.0]);
    assert_close(&ten_lines, &[-411.0]);
}

#[test]
fn a_flat_plane_is_unchanged() {
    let plane = vec![77.0f32; 9 * ROWS as usize];

    for depth in [0, 32] {
        let output = toned(json!({ "depth": depth }), SampleFormat::U8, &plane, 9);

        assert_eq!(output, plane, "depth {depth}");
    }
}

/// A hashed plane has lines in every direction, so the warp at depth 32 moves samples the flat warp does not.
#[test]
fn the_warp_changes_a_noisy_plane() {
    let width = 24u32;
    let plane: Vec<f32> = (0..width * 12)
        .map(|index| (60 + (index.wrapping_mul(2_654_435_761) >> 20) % 140) as f32)
        .collect();

    let warped_settings = settings_at_depth(32);
    let flat_settings = settings_at_depth(0);
    let warped = reference_toon(&plane, width, 12, &warped_settings);
    let flat = reference_toon(&plane, width, 12, &flat_settings);

    let differing = warped
        .iter()
        .zip(&flat)
        .filter(|(left, right)| left != right)
        .count();
    println!("{differing} of {} samples differ", plane.len());
    assert!(differing > 0);
}
