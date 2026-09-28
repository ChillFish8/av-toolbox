use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::testing::reference_edges;
use av_toolbox_gpu::{FrameFormat, StreamFormat};
use serde_json::{Value, json};

use crate::reference::{normalise_u8, square};
use crate::support::{
    SelectFrames,
    assert_masks_close,
    backend,
    built_mask,
    grey,
    mask_format,
    ten_bit_grey,
    wide_bytes,
    yuv420,
};

fn run(identifier: &str, args: Value, format: FrameFormat, frames: Vec<Vec<Vec<u8>>>) -> Vec<Vec<u8>> {
    let backend = backend();
    let output = mask_format(format.width, format.height);
    let stage = format!("edges/{identifier}");
    let mask = built_mask(&backend, &stage, args, &[StreamFormat::Frames(format)], output);

    backend.build(SelectFrames { mask, format, frames })
}

fn operator_for(identifier: &str) -> Operator {
    match identifier {
        "kirsch" => Operator::Kirsch,
        "prewitt" => Operator::Prewitt,
        "sobel" => Operator::Sobel,
        _ => unreachable!("test identifiers are fixed"),
    }
}

/// Mirrors `EdgeArgs::limits`, rounding each threshold to the nearest code value at `peak`.
fn limits(multi: f32, lthr: f32, hthr: f32, peak: f32) -> EdgeLimits {
    let low = (lthr * peak).round() / peak;
    let high = (hthr * peak).round() / peak;

    EdgeLimits {
        multi,
        low,
        high,
        peak: 1.0,
    }
}

/// Mirrors `store_mask`, writing `round(clamp(v, 0, 1) * 255)` as u8.
fn store(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(0.0, 1.0);

            (clamped * 255.0).round() as u8
        })
        .collect()
}

/// A row-repeated pattern that only changes value at each segment boundary, so every row is
/// identical, the vertical gradient is always zero, and the horizontal gradient at a boundary is
/// exactly that segment's jump.
fn horizontal_steps<T: Copy>(width: u32, height: u32, segments: &[(u32, T)]) -> Vec<T> {
    let mut row = Vec::with_capacity(width as usize);
    for (length, value) in segments {
        for _ in 0..*length {
            row.push(*value);
        }
    }
    assert_eq!(row.len(), width as usize, "segments must cover the width exactly");

    let mut output = Vec::with_capacity(row.len() * height as usize);
    for _ in 0..height {
        output.extend_from_slice(&row);
    }

    output
}

#[test]
fn each_operator_matches_the_reference_on_a_square() {
    let width = 24;
    let height = 24;
    let luma = square(width, height);
    let normalised = normalise_u8(&luma);
    let edge_limits = limits(1.0, 0.0, 1.0, 1.0);

    for identifier in ["kirsch", "prewitt", "sobel"] {
        let operator = operator_for(identifier);
        let expected_values = reference_edges(&normalised, width, height, operator, edge_limits);
        let expected = store(&expected_values);

        let format = grey(width, height);
        let frames = vec![vec![luma.clone()]];
        let masks = run(identifier, json!({}), format, frames);

        assert_masks_close(&masks[0], &expected, 1, 0.0);
        assert!(masks[0].contains(&0), "{identifier} found no background");
        assert!(
            masks[0].iter().any(|sample| *sample > 128),
            "{identifier} found no strong edge"
        );
    }
}

/// A fixture built from two boundaries: a jump of 6 code values, whose Sobel magnitude of
/// `24 / 255` sits well clear of `lthr`, and a jump of 56, whose magnitude of `224 / 255` sits
/// well clear of `hthr` and of 1. Both boundaries mark 2 of the 20 columns, so each covers 10% of
/// the plane over its 10 rows.
fn threshold_fixture(width: u32, height: u32) -> Vec<u8> {
    horizontal_steps(width, height, &[(5, 40), (9, 46), (6, 102)])
}

#[test]
fn the_thresholds_match_the_reference() {
    let width = 20;
    let height = 10;
    let low = 0.2;
    let high = 0.75;
    let luma = threshold_fixture(width, height);
    let normalised = normalise_u8(&luma);

    let raw_limits = limits(1.0, 0.0, 1.0, 1.0);
    let raw_values = reference_edges(&normalised, width, height, Operator::Sobel, raw_limits);

    // Column 4 sits on the small boundary, column 13 on the large one.
    let low_tier_magnitude = raw_values[4];
    let high_tier_magnitude = raw_values[13];
    assert!(
        low_tier_magnitude > 0.0 && low_tier_magnitude < low - 0.05,
        "the low-tier magnitude {low_tier_magnitude} is not clear of lthr {low}"
    );
    assert!(
        high_tier_magnitude > high + 0.05 && high_tier_magnitude < 1.0 - 0.05,
        "the high-tier magnitude {high_tier_magnitude} is not clear of hthr {high} and 1"
    );

    let total = raw_values.len() as f32;
    let low_share = raw_values
        .iter()
        .filter(|value| **value > 0.0 && **value < low)
        .count() as f32
        / total;
    let high_share = raw_values
        .iter()
        .filter(|value| **value > high && **value < 1.0)
        .count() as f32
        / total;
    assert!(low_share >= 0.05, "too few samples below lthr: {low_share}");
    assert!(high_share >= 0.05, "too few samples above hthr: {high_share}");

    let thresholded_limits = limits(1.0, low, high, 255.0);
    let thresholded_values = reference_edges(&normalised, width, height, Operator::Sobel, thresholded_limits);
    let thresholded = store(&thresholded_values);
    let unthresholded = store(&raw_values);
    assert_ne!(
        thresholded, unthresholded,
        "the thresholds had no effect on the reference"
    );

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run("sobel", json!({ "lthr": low, "hthr": high }), format, frames);

    assert_masks_close(&masks[0], &thresholded, 1, 0.0);
}

#[test]
fn a_large_multi_fills_rather_than_wraps() {
    let width = 24;
    let height = 24;
    let luma = square(width, height);

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run("kirsch", json!({ "multi": 50.0 }), format, frames);

    let edge_samples: Vec<u8> = masks[0].iter().copied().filter(|sample| *sample != 0).collect();
    assert!(!edge_samples.is_empty(), "no edge was found");
    assert!(
        edge_samples.iter().all(|sample| *sample == 255),
        "an edge sample did not fill to 255"
    );
}

/// `lthr = 0.1745` rounds to `179 / 1023` at the real 10-bit peak, and to `44 / 255` if the stage
/// mistakenly rounded against 255 instead. `multi` lifts a single 1-code jump's Prewitt magnitude
/// of `3 / 1023` to land strictly between those two roundings, clear of each by over `0.0008`
/// (checked below), so the two roundings give different masks at the jump.
const TEN_BIT_LTHR: f32 = 0.1745;
const TEN_BIT_MULTI: f32 = 59.25294;

#[test]
fn ten_bit_input_scales_its_thresholds() {
    let width = 10;
    let height = 4;
    let luma: Vec<u16> = horizontal_steps(width, height, &[(5u32, 500u16), (5, 501)]);
    let normalised: Vec<f32> = luma.iter().map(|sample| f32::from(*sample) / 1023.0).collect();

    let low_at_1023 = (TEN_BIT_LTHR * 1023.0).round() / 1023.0;
    let low_at_255 = (TEN_BIT_LTHR * 255.0).round() / 255.0;

    let raw_limits = limits(1.0, 0.0, 1.0, 1.0);
    let raw_values = reference_edges(&normalised, width, height, Operator::Prewitt, raw_limits);
    let jump_magnitude = raw_values[4];
    let scaled_magnitude = jump_magnitude * TEN_BIT_MULTI;

    assert!(
        (jump_magnitude - 3.0 / 1023.0).abs() < 1.0e-6,
        "the jump's raw magnitude was {jump_magnitude}, expected 3 / 1023"
    );

    let margin_to_255 = scaled_magnitude - low_at_255;
    let margin_to_1023 = low_at_1023 - scaled_magnitude;
    assert!(
        margin_to_255 > 0.0008,
        "too little margin to the 255 rounding: {margin_to_255}"
    );
    assert!(
        margin_to_1023 > 0.0008,
        "too little margin to the 1023 rounding: {margin_to_1023}"
    );

    let correct_limits = limits(TEN_BIT_MULTI, TEN_BIT_LTHR, 1.0, 1023.0);
    let wrong_limits = limits(TEN_BIT_MULTI, TEN_BIT_LTHR, 1.0, 255.0);
    let correct_values = reference_edges(&normalised, width, height, Operator::Prewitt, correct_limits);
    let wrong_values = reference_edges(&normalised, width, height, Operator::Prewitt, wrong_limits);
    let correct = store(&correct_values);
    let wrong = store(&wrong_values);
    assert_ne!(
        correct, wrong,
        "rounding at 1023 gave the same mask as rounding at 255"
    );

    let format = ten_bit_grey(width, height);
    let plane = wide_bytes(&luma);
    let frames = vec![vec![plane]];
    let args = json!({ "lthr": TEN_BIT_LTHR, "multi": TEN_BIT_MULTI });
    let masks = run("prewitt", args, format, frames);

    assert_masks_close(&masks[0], &correct, 1, 0.0);
}

#[test]
fn yuv_input_masks_the_luma_only() {
    let width = 24;
    let height = 24;
    let luma = square(width, height);
    let chroma_samples = (width / 2 * height / 2) as usize;
    let chroma_u: Vec<u8> = (0..chroma_samples).map(|index| (index * 13) as u8).collect();
    let chroma_v: Vec<u8> = (0..chroma_samples).map(|index| (index * 29) as u8).collect();

    let grey_format = grey(width, height);
    let grey_frames = vec![vec![luma.clone()]];
    let grey_masks = run("sobel", json!({}), grey_format, grey_frames);

    let yuv_format = yuv420(width, height);
    let yuv_frames = vec![vec![luma, chroma_u, chroma_v]];
    let yuv_masks = run("sobel", json!({}), yuv_format, yuv_frames);

    assert_eq!(yuv_masks[0].len(), (width * height) as usize);
    assert_eq!(yuv_masks[0], grey_masks[0]);
}

#[test]
fn a_flat_frame_has_no_edges() {
    let width = 16;
    let height = 16;
    let luma = vec![128u8; (width * height) as usize];

    for identifier in ["kirsch", "prewitt", "sobel"] {
        let format = grey(width, height);
        let frames = vec![vec![luma.clone()]];
        let masks = run(identifier, json!({}), format, frames);

        assert!(
            masks[0].iter().all(|sample| *sample == 0),
            "{identifier} found an edge on a flat frame"
        );
    }
}
