use av_toolbox_cubek::lineart::ToonSettings;
use av_toolbox_cubek::testing::{reference_narrow, reference_toon};
use av_toolbox_cubek::warp::{BlurKind, WarpParams};
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    Rounding,
    WIDTH,
    agreement,
    changed,
    element_of,
    encoded,
    faded_line_planes,
    luma_codes,
    peak_of,
    rounding_of,
    ten_bit,
    yuv420,
};
use crate::support::{from_bytes, run_masked, run_once, shared_planes, to_bytes, uniform_mask};

const TOON: &str = "lineart/toon";

/// The toon settings for `sample` at the defaults and a warp `depth`.
///
/// Written out here so the device is checked against values the stage did not produce.
fn toon_settings_for(sample: SampleFormat, depth: i32) -> ToonSettings {
    let peak = peak_of(sample) as f32;
    let scale = (peak + 1.0) / 256.0;
    let bits = if sample == SampleFormat::U8 { 8 } else { 10 };
    let warp = WarpParams {
        element: element_of(sample),
        bits,
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

/// The reference's answer for one plane as codes, and how close it sits to a rounding boundary.
fn toon_reference(values: &[u16], width: u32, height: u32, settings: &ToonSettings) -> (Vec<u16>, Rounding) {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let filtered = reference_toon(&widened, width, height, settings);
    let rounding = rounding_of(&filtered);
    let codes = reference_narrow(&filtered, settings.peak);

    (codes, rounding)
}

/// Checks every plane of `actual` is within a code of the reference.
///
/// The interpolation divides by `u - l`, so the reference lands on tenths at 8 bits and fortieths at
/// 10 bits. Only the samples within `RISK` of a rounding boundary may sit a code away, and every other
/// sample is exact. Returns the count of samples a code away across the frame.
fn assert_matches_toon(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    settings: &ToonSettings,
    label: &str,
) -> usize {
    let element = element_of(format.sample);
    let mut total_off_by_one = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let (expected, rounding) = toon_reference(&values[plane], width, height, settings);
        let result = agreement(&found, &expected);
        let moved = changed(&found, &values[plane]);
        let stray: Vec<usize> = (0..found.len())
            .filter(|index| found[*index] != expected[*index] && !rounding.at_risk.contains(index))
            .collect();
        total_off_by_one += result.off_by_one;

        println!(
            "{label} {channel}: largest {}, {} of {} off, {} at risk, {moved} moved, nearest margin {}",
            result.largest,
            result.off_by_one,
            found.len(),
            rounding.at_risk.len(),
            rounding.nearest,
        );
        assert!(moved > 0, "{label} {channel} never darkened a sample");
        assert!(
            result.largest <= 1,
            "{label} {channel} differs by {}",
            result.largest
        );
        assert!(
            stray.is_empty(),
            "{label} {channel} differs at {stray:?}, which do not sit near a boundary"
        );
    }

    total_off_by_one
}

/// The samples of the scene's planes the warp at `depth` changes the answer for.
fn warp_sensitive_samples(format: &FrameFormat, values: &[Vec<u16>], depth: i32) -> usize {
    let flat = toon_settings_for(format.sample, 0);
    let warped = toon_settings_for(format.sample, depth);
    let mut differing = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let (unwarped, _) = toon_reference(&values[plane], width, height, &flat);
        let (moved, _) = toon_reference(&values[plane], width, height, &warped);
        differing += changed(&moved, &unwarped);
    }

    differing
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

            let filtered = run_once(TOON, format, format, args, &planes);

            let label = format!("{sample} depth {depth}");
            let off_by_one = assert_matches_toon(&format, &values, &filtered, &settings, &label);
            let sensitive = warp_sensitive_samples(&format, &values, depth);
            println!("{label}: {off_by_one} samples a code away, {sensitive} moved by the warp");
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
        (SampleFormat::U8, 200u16, 150u16, 100u16),
        (SampleFormat::U8, 200, 190, 182),
        (ten_bit(), 800, 600, 400),
        (ten_bit(), 800, 760, 728),
    ];

    for (sample, background, line, expected) in cases {
        let format = yuv420(sample);
        let planes = faded_line_planes(&format, background, line);

        let filtered = run_once(TOON, format, format, json!({ "depth": 0 }), &planes);

        let luma = luma_codes(&filtered, sample);
        let column = (WIDTH / 2) as usize;
        for (index, value) in luma.iter().enumerate() {
            let want = if index % WIDTH as usize == column {
                expected
            } else {
                background
            };
            assert_eq!(*value, want, "{sample} line {line} at {index}");
        }
    }
}

/// At depth 0, lines 1 code either side of `l` and `u` in 100, and of `l` and `u` scaled in 400 at 10 bits.
#[test]
fn toon_switches_branches_at_the_thresholds_on_the_device() {
    let cases = [
        (
            SampleFormat::U8,
            100u16,
            [(99u16, 99u16), (98, 98), (97, 97), (89, 79), (88, 76), (87, 74)],
        ),
        (
            ten_bit(),
            400,
            [
                (393, 393),
                (392, 392),
                (391, 391),
                (353, 307),
                (352, 304),
                (351, 302),
            ],
        ),
    ];

    for (sample, background, lines) in cases {
        for (line, expected) in lines {
            let format = yuv420(sample);
            let planes = faded_line_planes(&format, background, line);

            let filtered = run_once(TOON, format, format, json!({ "depth": 0 }), &planes);

            let luma = luma_codes(&filtered, sample);
            let column = (WIDTH / 2) as usize;
            assert_eq!(luma[column], expected, "{sample} line {line}");
        }
    }
}

#[test]
fn a_flat_toon_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for depth in [0, 32] {
            let format = yuv420(sample);
            let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
            let element = element_of(sample);
            let planes: Vec<Vec<u8>> = format
                .channels()
                .iter()
                .map(|channel| {
                    let samples = format.channel_samples(*channel).expect("a listed channel");
                    to_bytes(&vec![flat; samples], element)
                })
                .collect();
            let args = json!({ "depth": depth, "planes": ["y", "u", "v"] });

            let filtered = run_once(TOON, format, format, args, &planes);

            assert_eq!(filtered, planes, "{sample} depth {depth}");
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
            let (mask, mask_chroma) = uniform_mask(&format, 0);
            let args = json!({ "depth": depth, "planes": ["y", "u", "v"] });

            let filtered = run_masked(TOON, format, args, &planes, mask, mask_chroma);

            assert_eq!(filtered, planes, "{sample} depth {depth}");
        }
    }
}

#[test]
fn toon_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(TOON, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}
