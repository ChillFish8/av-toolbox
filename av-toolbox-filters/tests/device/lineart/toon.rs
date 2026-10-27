use av_toolbox_cubek::lineart::ToonSettings;
use av_toolbox_cubek::testing::reference_toon;
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    WIDTH,
    assert_f32_matches_sixteen_bit,
    assert_matches,
    assert_unchanged,
    changed,
    faded_line_values,
    flat_values,
    peak_of,
    run_codes,
    run_masked_codes,
    sixteen_bit,
    yuv420,
};
use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    code_bits,
    element_of,
    encoded,
    shared_planes,
    ten_bit,
    uniform_mask,
    widened,
};

const TOON: &str = "lineart/toon";

/// The toon settings for `sample` at the defaults and a warp `depth`.
///
/// Written out here so the device is checked against values the stage did not produce.
fn toon_settings_for(sample: SampleFormat, depth: i32) -> ToonSettings {
    let peak = peak_of(sample) as f32;
    let scale = (peak + 1.0) / 256.0;
    let warp = WarpParams {
        element: element_of(sample),
        bits: code_bits(sample),
        thresh: 128,
        blur: BlurKind::R6,
        passes: 2,
        depth,
    };

    ToonSettings {
        strength: 1.0,
        l: 2.0 * scale,
        u: 12.0 * scale,
        n: (peak + 1.0) / 2.0,
        peak,
        warp,
    }
}

/// A scene of three bands of rows, scaled to the sample's depth.
///
/// - The top band is a flat 200 with 1-wide lines. Their depths sit 1 code below, on and 1 code above
///   `l` and `u`, with a depth between them and a deep one.
/// - The middle band is hashed noise, which puts lines of every depth in every direction.
/// - The bottom band is a flat 230 with 1-wide lines and pairs of side by side lines of different
///   depths, so a sample sits beside a darker one.
fn toon_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let sample = format.sample;
    let peak = peak_of(sample);
    let scale = (peak + 1) / 256;
    let lower = 2 * scale;
    let upper = 12 * scale;
    let depths = [
        lower - 1,
        lower,
        lower + 1,
        (lower + upper) / 2,
        upper - 1,
        upper,
        upper + 1,
        60 * scale,
    ];

    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    let band = row * 3 / height;
                    let value = match band {
                        0 if column % 4 == 2 => 200 * scale - depths[(column / 4) as usize % depths.len()],
                        0 => 200 * scale,
                        1 => {
                            let hash = (index + plane as u32 * 7).wrapping_mul(2_654_435_761) >> 20;
                            (60 + hash % 140) * scale
                        },
                        _ if column % 8 == 1 => 230 * scale - 3 * scale,
                        _ if column % 8 == 2 => 230 * scale - 9 * scale,
                        _ if column % 8 == 5 => 100 * scale,
                        _ => 230 * scale,
                    };

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// The reference's answer for one plane as codes.
fn toon_reference(values: &[u16], width: u32, height: u32, settings: &ToonSettings) -> Vec<f32> {
    let codes = widened(values);

    reference_toon(&codes, width, height, settings)
}

/// Checks every plane of `found` matches the reference, and that each one darkened a sample.
fn assert_matches_toon(
    format: &FrameFormat,
    values: &[Vec<u16>],
    found: &[Vec<f32>],
    settings: &ToonSettings,
    label: &str,
) {
    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let expected = toon_reference(&values[plane], width, height, settings);
        let moved = changed(&found[plane], &values[plane]);
        let plane_label = format!("{label} {channel}");

        println!("{plane_label}: {moved} moved");
        assert!(moved > 0, "{plane_label} never darkened a sample");
        assert_matches(&found[plane], &expected, &plane_label);
    }
}

/// The samples of the scene's planes the warp at `depth` changes the answer for.
fn warp_sensitive_samples(format: &FrameFormat, values: &[Vec<u16>], depth: i32) -> usize {
    let flat = toon_settings_for(format.sample, 0);
    let warped = toon_settings_for(format.sample, depth);
    let mut differing = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let unwarped = toon_reference(&values[plane], width, height, &flat);
        let moved = toon_reference(&values[plane], width, height, &warped);
        differing += moved
            .iter()
            .zip(&unwarped)
            .filter(|(left, right)| (*left - *right).abs() > ROUND_TRIP)
            .count();
    }

    differing
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
fn toon_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for depth in [0, 32] {
            let format = yuv420(sample);
            let values = toon_scene(&format);
            let planes = encoded(&format, &values);
            let settings = toon_settings_for(sample, depth);
            let args = json!({ "depth": depth, "planes": ["y", "u", "v"] });

            let found = run_codes(TOON, format, args, &planes);

            let label = format!("{sample} depth {depth}");
            assert_matches_toon(&format, &values, &found, &settings, &label);
            let sensitive = warp_sensitive_samples(&format, &values, depth);
            println!("{label}: {sensitive} moved by the warp");
            if depth != 0 {
                assert!(sensitive > 0, "{label} never reaches the warp");
            }
        }
    }
}

/// At depth 0, 150 and 190 in 200 give 100 and 182, and at 10 bits 600 and 760 in 800 give 400 and 728.
#[test]
fn toon_deepens_a_faded_line_on_the_device() {
    let cases = [
        (SampleFormat::U8, 200u16, 150u16, 100.0f32),
        (SampleFormat::U8, 200, 190, 182.0),
        (ten_bit(), 800, 600, 400.0),
        (ten_bit(), 800, 760, 728.0),
    ];

    for (sample, background, line, expected) in cases {
        let format = yuv420(sample);
        let values = faded_line_values(&format, background, line);
        let planes = encoded(&format, &values);

        let found = run_codes(TOON, format, json!({ "depth": 0 }), &planes);

        let label = format!("{sample} line {line}");
        assert_faded_line(&found, background, expected, &label);
    }
}

/// At depth 0, lines 1 code either side of `l` and `u` in 100, and of `l` and `u` scaled in 400 at 10 bits.
#[test]
fn toon_switches_branches_at_the_thresholds_on_the_device() {
    let cases = [
        (
            SampleFormat::U8,
            100u16,
            [
                (99u16, 99.0f32),
                (98, 98.0),
                (97, 96.7),
                (89, 79.1),
                (88, 76.0),
                (87, 74.0),
            ],
        ),
        (
            ten_bit(),
            400,
            [
                (393, 393.0),
                (392, 392.0),
                (391, 390.775),
                (353, 307.175),
                (352, 304.0),
                (351, 302.0),
            ],
        ),
    ];

    for (sample, background, lines) in cases {
        for (line, expected) in lines {
            let format = yuv420(sample);
            let values = faded_line_values(&format, background, line);
            let planes = encoded(&format, &values);

            let found = run_codes(TOON, format, json!({ "depth": 0 }), &planes);

            let label = format!("{sample} line {line}");
            assert_faded_line(&found, background, expected, &label);
        }
    }
}

#[test]
fn a_flat_toon_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for depth in [0, 32] {
            let format = yuv420(sample);
            let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
            let values = flat_values(&format, flat);
            let planes = encoded(&format, &values);
            let args = json!({ "depth": depth, "planes": ["y", "u", "v"] });

            let found = run_codes(TOON, format, args, &planes);

            let label = format!("{sample} depth {depth}");
            assert_unchanged(&found, &values, &label);
        }
    }
}

#[test]
fn a_zero_mask_hands_back_the_toon_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for depth in [0, 32] {
            let format = yuv420(sample);
            let values = toon_scene(&format);
            let planes = encoded(&format, &values);
            let mask = uniform_mask(&format, 0);
            let args = json!({ "depth": depth, "planes": ["y", "u", "v"] });

            let found = run_masked_codes(TOON, format, args, &planes, mask);

            let label = format!("{sample} depth {depth}");
            assert_unchanged(&found, &values, &label);
        }
    }
}

/// Blends `filtered` over `input` by a uniform mask of `weight`, as `blend/masked` does.
fn blended(input: &[u16], filtered: &[f32], weight: u8) -> Vec<f32> {
    let share = f32::from(weight) / 255.0;

    input
        .iter()
        .zip(filtered)
        .map(|(sample, target)| {
            let value = f32::from(*sample);

            value + (target - value) * share
        })
        .collect()
}

/// Toon alone takes a 60 line in 220 below 0, and a mask of 128 lands halfway between the input and
/// that unclamped output.
#[test]
fn a_masked_f32_merge_keeps_values_outside_the_range() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let scale = (peak_of(sample) as u16 + 1) / 256;
        let format = yuv420(sample);
        let values = faded_line_values(&format, 220 * scale, 60 * scale);
        let planes = encoded(&format, &values);
        let mask = uniform_mask(&format, 128);

        let alone = run_codes(TOON, format, json!({}), &planes);
        let masked = run_masked_codes(TOON, format, json!({}), &planes, mask);

        let expected = blended(&values[0], &alone[0], 128);
        let column = (WIDTH / 2) as usize;
        let label = format!("{sample}");

        println!(
            "{label}: toon alone gives {} on the line, masked {}, blended {}",
            alone[0][column], masked[0][column], expected[column],
        );
        assert!(
            alone[0][column] < 0.0,
            "{label} toon alone keeps the line at or above 0"
        );
        assert!(masked[0][column] < 0.0, "{label} the merge held the line at 0");
        assert_matches(&masked[0], &expected, &label);
    }
}

#[test]
fn toon_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::F32);

    let shared = shared_planes(TOON, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn toon_f32_matches_sixteen_bit() {
    let format = yuv420(sixteen_bit());
    let values = toon_scene(&format);
    let args = json!({ "planes": ["y", "u", "v"] });

    assert_f32_matches_sixteen_bit(TOON, args, &values, 1.0, 0.0);
}
