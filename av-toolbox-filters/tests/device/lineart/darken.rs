use av_toolbox_cubek::lineart::DarkenSettings;
use av_toolbox_cubek::testing::{reference_darken, reference_narrow};
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    Rounding,
    WIDTH,
    agreement,
    changed,
    faded_line_planes,
    lined_scene,
    luma_codes,
    rounding_of,
    yuv420,
};
use crate::support::{
    element_of,
    encoded,
    from_bytes,
    peak_of,
    run_masked,
    run_once,
    shared_planes,
    ten_bit,
    to_bytes,
    uniform_mask,
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

/// The reference's answer for one plane as codes, and how close it sits to a rounding boundary.
fn darken_reference(
    values: &[u16],
    width: u32,
    height: u32,
    settings: &DarkenSettings,
) -> (Vec<u16>, Rounding) {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let filtered = reference_darken(&widened, width, height, settings);
    let rounding = rounding_of(&filtered);
    let codes = reference_narrow(&filtered, settings.peak);

    (codes, rounding)
}

/// Checks every plane of `actual` is within a code of the reference.
///
/// Without thinning every value is a multiple of 1/8 and f32 holds it exactly, so no sample may sit a
/// code away. With thinning the merge divides by 255 or 1023 and the box by 9, so only the samples
/// at indices within `RISK` of a rounding boundary may, and every other sample is exact. At 8 bits
/// the closest is 0.0019 away and at 10 bits 0.0003, so 5 samples of the 10-bit luma plane are at
/// risk. Returns the count of samples a code away across the frame.
fn assert_matches_darken(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    settings: &DarkenSettings,
    label: &str,
) -> usize {
    let element = element_of(format.sample);
    let mut total_off_by_one = 0usize;

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let (expected, rounding) = darken_reference(&values[plane], width, height, settings);
        let result = agreement(&found, &expected);
        let moved = changed(&found, &values[plane]);
        let allowed: &[usize] = if settings.t == 0.0 { &[] } else { &rounding.at_risk };
        let stray: Vec<usize> = (0..found.len())
            .filter(|index| found[*index] != expected[*index] && !allowed.contains(index))
            .collect();
        total_off_by_one += result.off_by_one;

        println!(
            "{label} {channel}: largest {}, {} of {} off, {} allowed, {moved} moved, nearest margin {}",
            result.largest,
            result.off_by_one,
            found.len(),
            allowed.len(),
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

#[test]
fn darken_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0.0f32, 64.0] {
            let format = yuv420(sample);
            let values = lined_scene(&format);
            let planes = encoded(&format, &values);
            let settings = darken_settings_for(sample, thinning);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_once(DARKEN, format, format, args, &planes);

            let label = format!("{sample} thinning {thinning}");
            let off_by_one = assert_matches_darken(&format, &values, &filtered, &settings, &label);
            println!("{label}: {off_by_one} samples a code away");
        }
    }
}

/// A 150 line in 200 becomes 135, and with `protection = 50` 148. A 600 line in 800 becomes 538.5,
/// a tie that rounds up to 539.
#[test]
fn darken_deepens_a_faded_line_on_the_device() {
    let cases = [
        (SampleFormat::U8, 200u16, 150u16, json!({}), 135u16),
        (ten_bit(), 800, 600, json!({}), 539),
        (SampleFormat::U8, 200, 150, json!({ "protection": 50 }), 148),
    ];

    for (sample, background, line, args, expected) in cases {
        let format = yuv420(sample);
        let planes = faded_line_planes(&format, background, line);

        let filtered = run_once(DARKEN, format, format, args.clone(), &planes);

        let luma = luma_codes(&filtered, sample);
        let column = (WIDTH / 2) as usize;
        for (index, value) in luma.iter().enumerate() {
            let want = if index % WIDTH as usize == column {
                expected
            } else {
                background
            };
            assert_eq!(*value, want, "{sample} {args} at {index}");
        }
    }
}

/// Lines 1 code beyond `thr` below `cap` darken and lines on it or nearer do not.
#[test]
fn darken_switches_branches_at_the_threshold_on_the_device() {
    let cases = [
        (
            SampleFormat::U8,
            200u16,
            [(186u16, 184u16), (187, 187), (188, 188)],
        ),
        (ten_bit(), 800, [(747, 741), (748, 748), (749, 749)]),
    ];

    for (sample, background, lines) in cases {
        for (line, expected) in lines {
            let format = yuv420(sample);
            let planes = faded_line_planes(&format, background, line);

            let filtered = run_once(DARKEN, format, format, json!({}), &planes);

            let luma = luma_codes(&filtered, sample);
            let column = (WIDTH / 2) as usize;
            let found = luma[column];
            assert_eq!(found, expected, "{sample} line {line}");
        }
    }
}

#[test]
fn a_flat_darken_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        for thinning in [0, 64] {
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
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_once(DARKEN, format, format, args, &planes);

            assert_eq!(filtered, planes, "{sample} thinning {thinning}");
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
            let (mask, mask_chroma) = uniform_mask(&format, 0);
            let args = json!({ "thinning": thinning, "planes": ["y", "u", "v"] });

            let filtered = run_masked(DARKEN, format, args, &planes, mask, mask_chroma);

            assert_eq!(filtered, planes, "{sample} thinning {thinning}");
        }
    }
}

#[test]
fn darken_leaves_unselected_planes_as_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(DARKEN, format, json!({}));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn a_strength_of_zero_hands_back_the_plane() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = lined_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(DARKEN, format, format, json!({ "strength": 0 }), &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}
