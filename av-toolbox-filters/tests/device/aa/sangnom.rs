use av_toolbox_cubek::aa::{sangnom_aaf, sangnom_max};
use av_toolbox_cubek::testing::{
    assert_same_plane,
    reference_narrow,
    reference_sangnom_antialias,
    transposed_plane,
};
use av_toolbox_gpu::{FrameFormat, SampleFormat};
use serde_json::json;

use super::{bits_of, decoded, grey, largest_vertical_step, one_sample_diagonal, yuv420};
use crate::support::{encoded, peak_of, run_masked, run_once, shared_planes, ten_bit, uniform_mask};

const STAGE: &str = "aa/sangnom";

const DEFAULT_AA: u32 = 48;

/// The strongest `aa`, which lets the most edges be followed.
const HARSH_AA: u32 = 128;

const WIDTH: u32 = 24;

const HEIGHT: u32 = 16;

/// The directions `aa/sangnom` runs along, as its arguments name them.
const DIRECTIONS: [&str; 3] = ["vertical", "horizontal", "both"];

/// A shallow line two samples thick over a textured background, drawn without anti-aliasing and
/// scaled to `sample`'s depth.
fn aliased_diagonal(width: u32, height: u32, sample: SampleFormat) -> Vec<u16> {
    let scale = (peak_of(sample) + 1) / 256;

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let line_start = row * 3 / 2;
            let on_line = column >= line_start && column < line_start + 2;
            let texture = (column * 7 + row * 3) % 11;
            let value = if on_line { 220 } else { 40 + texture };

            (value * scale) as u16
        })
        .collect()
}

/// A peak field with a mid-grey dot wherever `column + 2 * row` is a multiple of 5.
///
/// Beside a dot, a kept line reads peak, peak, dot, and the SangNom predictor gives
/// `(9 * peak - dot) >> 3`, above the peak.
fn dotted_peak(width: u32, height: u32, sample: SampleFormat) -> Vec<u16> {
    let peak = peak_of(sample);
    let dot = peak.div_ceil(2);

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let lattice = column + 2 * row;
            let on_dot = lattice.is_multiple_of(5);
            let value = if on_dot { dot } else { peak };

            value as u16
        })
        .collect()
}

fn vertical_reference(values: &[f32], width: u32, height: u32, sample: SampleFormat, aa: u32) -> Vec<f32> {
    let bits = bits_of(sample);
    let aaf = sangnom_aaf(aa, bits);
    let max = sangnom_max(bits);

    reference_sangnom_antialias(values, width, height, aaf, max)
}

fn horizontal_reference(values: &[f32], width: u32, height: u32, sample: SampleFormat, aa: u32) -> Vec<f32> {
    let swapped = transposed_plane(values, width, height);
    let filtered = vertical_reference(&swapped, height, width, sample, aa);

    transposed_plane(&filtered, height, width)
}

/// The reference's answer along `direction`, before narrowing.
fn reference_values(
    values: &[u16],
    width: u32,
    height: u32,
    sample: SampleFormat,
    aa: u32,
    direction: &str,
) -> Vec<f32> {
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();

    match direction {
        "vertical" => vertical_reference(&widened, width, height, sample, aa),
        "horizontal" => horizontal_reference(&widened, width, height, sample, aa),
        _ => {
            let vertical = vertical_reference(&widened, width, height, sample, aa);

            horizontal_reference(&vertical, width, height, sample, aa)
        },
    }
}

/// Runs a `Grey` clip of `values` along `direction` and checks it against the reference exactly.
fn assert_grey_matches(values: &[u16], sample: SampleFormat, aa: u32, direction: &str) {
    let format = grey(WIDTH, HEIGHT, sample);
    let plane_values = [values.to_vec()];
    let planes = encoded(&format, &plane_values);
    let args = json!({ "aa": aa, "direction": direction });

    let filtered = run_once(STAGE, format, format, args, &planes);

    let found = decoded(&format, &filtered);
    let reference = reference_values(values, WIDTH, HEIGHT, sample, aa, direction);
    let peak = peak_of(sample) as f32;
    let expected = reference_narrow(&reference, peak);
    let changed = expected
        .iter()
        .zip(values)
        .filter(|(result, value)| result != value)
        .count();

    let label = format!("{sample} {direction}");
    assert!(changed > 0, "{label} leaves the clip unchanged");
    assert_same_plane(&found[0], &expected, WIDTH, &label);
}

#[test]
fn vertical_matches_the_reference() {
    let values = aliased_diagonal(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, DEFAULT_AA, "vertical");
}

#[test]
fn horizontal_matches_the_transposed_reference() {
    let values = aliased_diagonal(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, DEFAULT_AA, "horizontal");
}

#[test]
fn both_runs_vertical_then_horizontal() {
    let values = aliased_diagonal(WIDTH, HEIGHT, SampleFormat::U8);

    assert_grey_matches(&values, SampleFormat::U8, DEFAULT_AA, "both");
}

#[test]
fn ten_bit_matches_and_clamps_to_the_peak() {
    let sample = ten_bit();
    let peak = peak_of(sample) as f32;
    let values = dotted_peak(WIDTH, HEIGHT, sample);
    let widened: Vec<f32> = values.iter().map(|value| f32::from(*value)).collect();

    let vertical = vertical_reference(&widened, WIDTH, HEIGHT, sample, HARSH_AA);
    let vertical_above_peak = vertical.iter().filter(|value| **value > peak).count();
    assert!(
        vertical_above_peak > 0,
        "the vertical intermediate never passes the peak"
    );

    for direction in DIRECTIONS {
        let reference = reference_values(&values, WIDTH, HEIGHT, sample, HARSH_AA, direction);
        let above_peak = reference.iter().filter(|value| **value > peak).count();
        assert!(above_peak > 0, "{direction} never passes the peak");

        assert_grey_matches(&values, sample, HARSH_AA, direction);
    }
}

#[test]
fn the_diagonal_matches_at_ten_bits() {
    let sample = ten_bit();
    let values = aliased_diagonal(WIDTH, HEIGHT, sample);

    for direction in DIRECTIONS {
        assert_grey_matches(&values, sample, DEFAULT_AA, direction);
    }
}

#[test]
fn only_the_selected_planes_change() {
    let format = yuv420(WIDTH, HEIGHT, SampleFormat::U8);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}

/// A `yuv420` clip whose every plane holds an aliased diagonal.
fn diagonal_planes(format: &FrameFormat) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            aliased_diagonal(width, height, format.sample)
        })
        .collect()
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let values = diagonal_planes(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 0);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn a_full_mask_hands_back_the_filtered_plane() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let values = diagonal_planes(&format);
        let planes = encoded(&format, &values);
        let (mask, mask_chroma) = uniform_mask(&format, 255);

        let filtered = run_masked(STAGE, format, json!({}), &planes, mask, mask_chroma);

        let found = decoded(&format, &filtered);
        let reference = reference_values(&values[0], WIDTH, HEIGHT, sample, DEFAULT_AA, "both");
        let peak = peak_of(sample) as f32;
        let expected = reference_narrow(&reference, peak);
        let label = format!("{sample} full mask");
        assert_same_plane(&found[0], &expected, WIDTH, &label);
        assert_eq!(found[1..], values[1..], "{label} changed the chroma");
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
        let values: Vec<Vec<u16>> = format
            .channels()
            .iter()
            .map(|channel| {
                let samples = format.channel_samples(*channel).expect("a listed channel");
                vec![flat; samples]
            })
            .collect();
        let planes = encoded(&format, &values);
        let args = json!({ "planes": ["y", "u", "v"] });

        let filtered = run_once(STAGE, format, format, args, &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn every_plane_of_a_subsampled_clip_matches_at_its_own_size() {
    let sample = SampleFormat::U8;
    let format = yuv420(WIDTH, HEIGHT, sample);
    let values = diagonal_planes(&format);
    let planes = encoded(&format, &values);
    let args = json!({ "planes": ["y", "u", "v"], "direction": "both" });

    let filtered = run_once(STAGE, format, format, args, &planes);

    let found = decoded(&format, &filtered);
    let peak = peak_of(sample) as f32;
    for (index, channel) in format.channels().iter().enumerate() {
        let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
        let reference = reference_values(&values[index], width, height, sample, DEFAULT_AA, "both");
        let expected = reference_narrow(&reference, peak);

        let label = format!("{channel:?} at {width}x{height}");
        assert_ne!(expected, values[index], "{label} leaves the plane unchanged");
        assert_same_plane(&found[index], &expected, width, &label);
    }
}

#[test]
fn a_one_sample_diagonal_comes_back_with_smaller_steps() {
    let width = 32;
    let height = 24;
    let format = grey(width, height, SampleFormat::U8);
    let values = one_sample_diagonal(width, height);
    let plane_values = [values.clone()];
    let planes = encoded(&format, &plane_values);
    let before = largest_vertical_step(&values, width);

    for direction in ["vertical", "both"] {
        let args = json!({ "direction": direction });

        let filtered = run_once(STAGE, format, format, args, &planes);

        let found = decoded(&format, &filtered);
        let after = largest_vertical_step(&found[0], width);
        assert!(
            after < before,
            "{direction} leaves a step of {after}, the input's was {before}"
        );
    }
}
