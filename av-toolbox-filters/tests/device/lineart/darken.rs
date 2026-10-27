use av_toolbox_cubek::lineart::DarkenSettings;
use av_toolbox_cubek::testing::reference_darken;
use av_toolbox_gpu::{ColourRange, FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    WIDTH,
    assert_f32_matches_sixteen_bit,
    assert_matches,
    assert_unchanged,
    changed,
    faded_line_values,
    flat_values,
    lined_scene,
    peak_of,
    run_codes,
    run_masked_codes,
    sixteen_bit,
    yuv420,
};
use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    encoded,
    float_planes,
    run_into_codes,
    shared_planes,
    ten_bit,
    uniform_mask,
    widened,
};

const DARKEN: &str = "lineart/darken";

/// The darken settings for `sample` at the defaults, with `thinning` given on its own scale.
///
/// Written out here so the device is checked against values the stage did not produce.
fn darken_settings_for(sample: SampleFormat, thinning: f32) -> DarkenSettings {
    let peak = peak_of(sample) as f32;
    let scale = (peak + 1.0) / 256.0;

    DarkenSettings {
        s: 48.0 / 128.0,
        limit: peak / 6.0,
        cap: 191.0 * scale,
        thr: 4.0 * scale,
        t: thinning / 16.0,
        n: (peak + 1.0) / 2.0,
        peak,
    }
}

/// Checks every plane of `found` matches the reference, and that each one darkened a sample.
fn assert_matches_darken(
    format: &FrameFormat,
    values: &[Vec<u16>],
    found: &[Vec<f32>],
    settings: &DarkenSettings,
    label: &str,
) {
    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let codes = widened(&values[plane]);
        let expected = reference_darken(&codes, width, height, settings);
        let moved = changed(&found[plane], &values[plane]);
        let plane_label = format!("{label} {channel}");

        println!("{plane_label}: {moved} moved");
        assert!(moved > 0, "{plane_label} never darkened a sample");
        assert_matches(&found[plane], &expected, &plane_label);
    }
}

/// Checks the luma line of `found` holds `line` and every other sample `background`.
fn assert_faded_line(found: &[Vec<f32>], background: u16, line: f32, label: &str) {
    let column = (WIDTH / 2) as usize;
    let expected: Vec<f32> = (0..found[0].len())
        .map(|index| {
            if index % WIDTH as usize == column {
                line
            } else {
                f32::from(background)
            }
        })
        .collect();

    assert_codes_within(&found[0], &expected, ROUND_TRIP, label);
}

#[test]
fn darken_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0.0f32, 64.0] {
            let format = yuv420(sample);
            let values = lined_scene(&format);
            let planes = encoded(&format, &values);
            let settings = darken_settings_for(sample, thinning);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let found = run_codes(DARKEN, format, args, &planes);

            let label = format!("{sample} thinning {thinning}");
            assert_matches_darken(&format, &values, &found, &settings, &label);
        }
    }
}

/// A 150 line in 200 becomes 134.625, and with `protection = 50` 148.125. A 600 line in 800 becomes
/// 538.5.
#[test]
fn darken_deepens_a_faded_line_on_the_device() {
    let cases = [
        (SampleFormat::U8, 200u16, 150u16, json!({}), 134.625f32),
        (ten_bit(), 800, 600, json!({}), 538.5),
        (SampleFormat::U8, 200, 150, json!({ "protection": 50 }), 148.125),
    ];

    for (sample, background, line, args, expected) in cases {
        let format = yuv420(sample);
        let values = faded_line_values(&format, background, line);
        let planes = encoded(&format, &values);

        let found = run_codes(DARKEN, format, args.clone(), &planes);

        let label = format!("{sample} {args}");
        assert_faded_line(&found, background, expected, &label);
    }
}

/// Lines 1 code beyond `thr` below `cap` darken and lines on it or nearer do not.
#[test]
fn darken_switches_branches_at_the_threshold_on_the_device() {
    let cases = [
        (
            SampleFormat::U8,
            200u16,
            [(186u16, 184.125f32), (187, 187.0), (188, 188.0)],
        ),
        (ten_bit(), 800, [(747, 740.625), (748, 748.0), (749, 749.0)]),
    ];

    for (sample, background, lines) in cases {
        for (line, expected) in lines {
            let format = yuv420(sample);
            let values = faded_line_values(&format, background, line);
            let planes = encoded(&format, &values);

            let found = run_codes(DARKEN, format, json!({}), &planes);

            let label = format!("{sample} line {line}");
            assert_faded_line(&found, background, expected, &label);
        }
    }
}

/// The closing of a 150 line in 230 is held at the cap of 191, so the line becomes 134.625, all
/// times 256 in the codes of a limited F32 clip.
#[test]
fn darken_keeps_its_luma_cap_on_f32() {
    let mut format = yuv420(SampleFormat::F32);
    format.colour.range = ColourRange::Limited;
    let values = faded_line_values(&format, 230 * 256, 150 * 256);
    let planes = float_planes(&format, &values, 16);

    let found = run_into_codes(DARKEN, format, json!({}), &planes, 16);

    assert_faded_line(&found, 230 * 256, 134.625 * 256.0, "limited F32");
}

#[test]
fn a_flat_darken_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0, 64] {
            let format = yuv420(sample);
            let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
            let values = flat_values(&format, flat);
            let planes = encoded(&format, &values);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let found = run_codes(DARKEN, format, args, &planes);

            let label = format!("{sample} thinning {thinning}");
            assert_unchanged(&found, &values, &label);
        }
    }
}

#[test]
fn a_zero_mask_hands_back_the_darken_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0, 64] {
            let format = yuv420(sample);
            let values = lined_scene(&format);
            let planes = encoded(&format, &values);
            let mask = uniform_mask(&format, 0);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let found = run_masked_codes(DARKEN, format, args, &planes, mask);

            let label = format!("{sample} thinning {thinning}");
            assert_unchanged(&found, &values, &label);
        }
    }
}

#[test]
fn darken_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::F32);

    let shared = shared_planes(DARKEN, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn a_strength_of_zero_hands_back_the_plane() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = lined_scene(&format);
        let planes = encoded(&format, &values);

        let found = run_codes(DARKEN, format, json!({ "strength": 0 }), &planes);

        let label = format!("{sample}");
        assert_unchanged(&found, &values, &label);
    }
}

#[test]
fn darken_f32_matches_sixteen_bit() {
    for thinning in [0, 64] {
        let format = yuv420(sixteen_bit());
        let values = lined_scene(&format);
        let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

        assert_f32_matches_sixteen_bit(DARKEN, args, &values, 1.0, 0.0);
    }
}
