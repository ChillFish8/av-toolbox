use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::testing::{reference_narrow, reference_vinverse};
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{agreement, changed, element_of, encoded, peak_of, ten_bit, yuv420};
use crate::support::{from_bytes, run_masked, run_once, shared_planes, split_mask, to_bytes, uniform_mask};

const STAGE: &str = "lineart/vinverse";

const CONTRA_STR: f32 = 2.7;

const SCL: f32 = 0.25;

/// The most a sample may move in `amnt_is_scaled_with_the_bit_depth`, on the 8-bit scale.
const AMNT: f32 = 20.0;

/// The samples of one plane that sit a code from the reference. None is safe, as the closest
/// reference value sits 0.00625 codes from a rounding boundary at 10 bits (0.0406 at 8 bits), far
/// beyond the 2.4e-5 of f32 error.
const OFF_BY_ONE_SAMPLES: usize = 0;

/// A scene with a comb on every row, a ramp down the plane, detail across it and some noise, shifted
/// per plane and scaled to the sample's depth.
fn combed_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let peak = peak_of(format.sample);

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
                    let comb = (row % 2) * 40;
                    let ramp = row * 2;
                    let detail = (column * 5) % 24;
                    let noise = (index * 37 + plane as u32 * 11) % 9;
                    let value = (30 + comb + ramp + detail + noise) * (peak + 1) / 256;

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// The reference's answer for one plane, rounded to codes.
fn reference_codes(values: &[u16], width: u32, height: u32, peak: u32, amnt: Option<f32>) -> Vec<u16> {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();
    let filtered = reference_vinverse(&widened, width, height, CONTRA_STR, SCL, amnt);

    reference_narrow(&filtered, peak as f32)
}

/// Checks every plane of `actual` is within a code of the reference, with `OFF_BY_ONE_SAMPLES`
/// samples a code away.
fn assert_matches_reference(
    format: &FrameFormat,
    values: &[Vec<u16>],
    actual: &[Vec<u8>],
    amnt: Option<f32>,
    label: &str,
) {
    let element = element_of(format.sample);
    let peak = peak_of(format.sample);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&actual[plane], element);
        let expected = reference_codes(&values[plane], width, height, peak, amnt);
        let result = agreement(&found, &expected);

        println!(
            "{label} {channel}: largest {}, {} of {} a code away",
            result.largest,
            result.off_by_one,
            found.len(),
        );
        assert!(
            result.largest <= 1,
            "{label} {channel} differs by {}",
            result.largest
        );
        assert_eq!(
            result.off_by_one, OFF_BY_ONE_SAMPLES,
            "{label} {channel} has samples a code away"
        );
    }
}

#[test]
fn vinverse_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(STAGE, format, format, json!({}), &planes);

        let label = format!("{sample}");
        assert_matches_reference(&format, &values, &filtered, None, &label);
    }
}

#[test]
fn amnt_is_scaled_with_the_bit_depth() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let scale = (peak_of(sample) + 1) as f32 / 256.0;
        let limit = AMNT * scale;
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);

        let filtered = run_once(STAGE, format, format, json!({ "amnt": AMNT }), &planes);

        let label = format!("{sample} amnt");
        assert_matches_reference(&format, &values, &filtered, Some(limit), &label);

        let mut clamped = 0usize;
        let mut unclamped = 0usize;
        for (channel, source) in format.channels().iter().zip(&values) {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let widened: Vec<f32> = source.iter().map(|value| f32::from(*value)).collect();
            let free = reference_vinverse(&widened, width, height, CONTRA_STR, SCL, None);

            for (result, value) in free.iter().zip(&widened) {
                if (result - value).abs() > limit {
                    clamped += 1;
                } else {
                    unclamped += 1;
                }
            }
        }

        println!("{label}: {clamped} clamped, {unclamped} unclamped");
        assert!(clamped > 0, "{label} never reaches its limit");
        assert!(unclamped > 0, "{label} is limited everywhere");
    }
}

#[test]
fn vinverse_removes_a_comb_on_the_device() {
    let format = yuv420(SampleFormat::U8);
    let planes: Vec<Vec<u8>> = format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| if (index / width) % 2 == 0 { 100 } else { 140 })
                .collect()
        })
        .collect();

    let filtered = run_once(STAGE, format, format, json!({}), &planes);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let samples = &filtered[plane];
        let mut largest = 0u8;

        for row in 1..height - 2 {
            let this = samples[(row * width) as usize];
            let next = samples[((row + 1) * width) as usize];
            largest = largest.max(this.abs_diff(next));
        }

        println!("{channel}: largest row to row step {largest}");
        assert!(largest < 2, "{channel} still combs by {largest}");
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
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

        let filtered = run_once(STAGE, format, format, json!({}), &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 0);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn a_full_mask_hands_back_the_filtered_frame() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 255);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        let label = format!("{sample} full mask");
        assert_matches_reference(&format, &values, &filtered, None, &label);
    }
}

#[test]
fn a_mask_merges_chroma_through_its_chroma_plane() {
    let format = yuv420(SampleFormat::U8);
    let values = combed_scene(&format);
    let planes = encoded(&format, &values);
    let (mask, mask_chroma) = split_mask(&format);

    let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = from_bytes(&filtered[plane], Element::U8);
        let source = &values[plane];
        let mut left_changed = 0usize;

        for row in 0..height {
            for column in 0..width {
                let index = (row * width + column) as usize;
                if column < width / 2 {
                    left_changed += usize::from(found[index] != source[index]);
                } else {
                    assert_eq!(found[index], source[index], "{channel} at {column},{row}");
                }
            }
        }

        let left_samples = (width / 2 * height) as usize;
        println!("{channel}: {left_changed} of {left_samples} left samples changed");
        assert!(left_changed > 0, "{channel} left half is untouched");

        let total_changed = changed(&found, source);
        assert_eq!(
            total_changed, left_changed,
            "{channel} changed samples on the right"
        );
    }
}

#[test]
fn unselected_planes_share_the_input_buffers() {
    let format = yuv420(SampleFormat::U8);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}
