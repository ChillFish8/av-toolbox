use av_toolbox_cubek::testing::reference_vinverse;
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    assert_f32_matches_sixteen_bit,
    assert_matches,
    assert_unchanged,
    changed,
    flat_values,
    moved,
    peak_of,
    run_codes,
    run_masked_codes,
    sixteen_bit,
    yuv420,
};
use crate::support::{encoded, shared_planes, split_mask, ten_bit, uniform_mask, widened};

const STAGE: &str = "lineart/vinverse";

const CONTRA_STR: f32 = 2.7;

const SCL: f32 = 0.25;

/// The most a sample may move in `amnt_is_scaled_with_the_bit_depth`, on the 8-bit scale.
const AMNT: f32 = 20.0;

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

/// Checks every plane of `found` matches the reference.
fn assert_matches_reference(
    format: &FrameFormat,
    values: &[Vec<u16>],
    found: &[Vec<f32>],
    amnt: Option<f32>,
    label: &str,
) {
    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let codes = widened(&values[plane]);
        let expected = reference_vinverse(&codes, width, height, CONTRA_STR, SCL, amnt);
        let plane_label = format!("{label} {channel}");

        assert_matches(&found[plane], &expected, &plane_label);
    }
}

#[test]
fn vinverse_matches_the_reference() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);

        let found = run_codes(STAGE, format, json!({}), &planes);

        let label = format!("{sample}");
        assert_matches_reference(&format, &values, &found, None, &label);
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

        let found = run_codes(STAGE, format, json!({ "amnt": AMNT }), &planes);

        let label = format!("{sample} amnt");
        assert_matches_reference(&format, &values, &found, Some(limit), &label);

        let mut clamped = 0usize;
        let mut unclamped = 0usize;
        for (channel, source) in format.channels().iter().zip(&values) {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let codes = widened(source);
            let free = reference_vinverse(&codes, width, height, CONTRA_STR, SCL, None);

            for (result, value) in free.iter().zip(&codes) {
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

    let found = run_codes(STAGE, format, json!({}), &planes);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let samples = &found[plane];
        let mut largest = 0.0f32;

        for row in 1..height - 2 {
            let this = samples[(row * width) as usize];
            let next = samples[((row + 1) * width) as usize];
            largest = largest.max((this - next).abs());
        }

        println!("{channel}: largest row to row step {largest}");
        assert!(largest < 2.0, "{channel} still combs by {largest}");
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
        let values = flat_values(&format, flat);
        let planes = encoded(&format, &values);

        let found = run_codes(STAGE, format, json!({}), &planes);

        let label = format!("{sample}");
        assert_unchanged(&found, &values, &label);
    }
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let mask = uniform_mask(&format, 0);

        let found = run_masked_codes(STAGE, format, json!({}), &planes, mask);

        let label = format!("{sample}");
        assert_unchanged(&found, &values, &label);
    }
}

#[test]
fn a_full_mask_hands_back_the_filtered_frame() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(sample);
        let values = combed_scene(&format);
        let planes = encoded(&format, &values);
        let mask = uniform_mask(&format, 255);

        let found = run_masked_codes(STAGE, format, json!({}), &planes, mask);

        let label = format!("{sample} full mask");
        assert_matches_reference(&format, &values, &found, None, &label);
    }
}

#[test]
fn a_mask_merges_chroma_through_its_chroma_plane() {
    let format = yuv420(SampleFormat::U8);
    let values = combed_scene(&format);
    let planes = encoded(&format, &values);
    let mask = split_mask(&format);

    let filtered = run_masked_codes(STAGE, format, json!({}), &planes, mask);

    for (plane, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let found = &filtered[plane];
        let source = &values[plane];
        let mut left_changed = 0usize;

        for row in 0..height {
            for column in 0..width {
                let index = (row * width + column) as usize;
                if column < width / 2 {
                    left_changed += usize::from(moved(found[index], source[index]));
                } else {
                    let kept = !moved(found[index], source[index]);
                    assert!(kept, "{channel} at {column},{row}");
                }
            }
        }

        let left_samples = (width / 2 * height) as usize;
        println!("{channel}: {left_changed} of {left_samples} left samples changed");
        assert!(left_changed > 0, "{channel} left half is untouched");

        let total_changed = changed(found, source);
        assert_eq!(
            total_changed, left_changed,
            "{channel} changed samples on the right"
        );
    }
}

#[test]
fn unselected_planes_share_the_input_buffers() {
    let format = yuv420(SampleFormat::F32);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn vinverse_f32_matches_sixteen_bit() {
    let format = yuv420(sixteen_bit());
    let values = combed_scene(&format);

    assert_f32_matches_sixteen_bit(STAGE, json!({}), &values, 1.0, 0.0);
}
