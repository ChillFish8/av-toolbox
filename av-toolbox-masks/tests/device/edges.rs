use av_toolbox_cubek::edges::{EdgeLimits, Operator};
use av_toolbox_cubek::testing::reference_edges;
use av_toolbox_gpu::{BuildContext, BuildTask, FrameFormat, StreamFormat};
use av_toolbox_masks::testing::store_mask;
use cubecl::prelude::Runtime;
use serde_json::{Value, json};

use crate::reference::square;
use crate::support::{
    SelectFrames,
    assert_masks_close,
    backend,
    built_mask,
    empty,
    grey,
    mask_format,
    plane,
    read_u8,
    ten_bit_grey,
    upload_f32,
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
    let low = (lthr * peak).round();
    let high = (hthr * peak).round();

    EdgeLimits {
        multi,
        low,
        high,
        peak,
    }
}

/// Mirrors `store_mask`, writing `round(clamp(v, 0, peak) * 255 / peak)` as u8.
fn store(values: &[f32], peak: f32) -> Vec<u8> {
    values
        .iter()
        .map(|value| {
            let clamped = value.clamp(0.0, peak);

            (clamped * 255.0 / peak).round() as u8
        })
        .collect()
}

/// `samples` as f32 code values, unscaled.
fn code_values(samples: &[u8]) -> Vec<f32> {
    samples.iter().map(|sample| f32::from(*sample)).collect()
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
    let peak = 255.0;
    let luma = square(width, height);
    let code_values = code_values(&luma);
    let edge_limits = limits(1.0, 0.0, 1.0, peak);

    for identifier in ["kirsch", "prewitt", "sobel"] {
        let operator = operator_for(identifier);
        let expected_values = reference_edges(&code_values, width, height, operator, edge_limits);
        let expected = store(&expected_values, peak);

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

/// A fixture built from three boundaries, in 8-bit code values: a jump of 6 (Sobel magnitude 24),
/// well below `lthr`; a jump of 20 (magnitude 80), clearly between `lthr` and `hthr`; and a jump
/// of 56 (magnitude 224), well above `hthr` and clear of `peak`. Each boundary marks 2 of the 20
/// columns, so each covers 10% of the plane over its 10 rows.
fn threshold_fixture(width: u32, height: u32) -> Vec<u8> {
    horizontal_steps(width, height, &[(5, 40), (5, 46), (5, 66), (5, 122)])
}

#[test]
fn the_thresholds_match_the_reference() {
    let width = 20;
    let height = 10;
    let peak = 255.0;
    let low = 0.2;
    let high = 0.75;
    let luma = threshold_fixture(width, height);
    let code_values = code_values(&luma);

    let raw_limits = limits(1.0, 0.0, 1.0, peak);
    let raw_values = reference_edges(&code_values, width, height, Operator::Sobel, raw_limits);

    let thresholded_limits = limits(1.0, low, high, peak);
    let low_bound = thresholded_limits.low;
    let high_bound = thresholded_limits.high;
    assert_eq!(low_bound, 51.0, "lthr did not round to the expected bound");
    assert_eq!(high_bound, 191.0, "hthr did not round to the expected bound");

    // Column 4 sits on the small boundary, column 9 on the middle one, column 14 on the large one.
    let low_tier_magnitude = raw_values[4];
    let middle_tier_magnitude = raw_values[9];
    let high_tier_magnitude = raw_values[14];
    let margin = 15.0;
    assert!(
        low_tier_magnitude > 0.0 && low_tier_magnitude < low_bound - margin,
        "the low-tier magnitude {low_tier_magnitude} is not clear of lthr {low_bound}"
    );
    assert!(
        middle_tier_magnitude > low_bound + margin && middle_tier_magnitude < high_bound - margin,
        "the middle-tier magnitude {middle_tier_magnitude} is not clear of lthr {low_bound} and hthr {high_bound}"
    );
    assert!(
        high_tier_magnitude > high_bound + margin && high_tier_magnitude < peak - margin,
        "the high-tier magnitude {high_tier_magnitude} is not clear of hthr {high_bound} and peak {peak}"
    );

    let total = raw_values.len() as f32;
    let low_share = raw_values
        .iter()
        .filter(|value| **value > 0.0 && **value < low_bound)
        .count() as f32
        / total;
    let middle_share = raw_values
        .iter()
        .filter(|value| **value >= low_bound && **value <= high_bound)
        .count() as f32
        / total;
    let high_share = raw_values
        .iter()
        .filter(|value| **value > high_bound && **value < peak)
        .count() as f32
        / total;
    assert!(low_share >= 0.05, "too few samples below lthr: {low_share}");
    assert!(
        middle_share >= 0.05,
        "too few samples between lthr and hthr: {middle_share}"
    );
    assert!(high_share >= 0.05, "too few samples above hthr: {high_share}");

    let thresholded_values =
        reference_edges(&code_values, width, height, Operator::Sobel, thresholded_limits);
    let thresholded = store(&thresholded_values, peak);
    let unthresholded = store(&raw_values, peak);
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

/// `lthr = 0.1745` rounds to `179` at the real 10-bit peak, and to `44` if the stage mistakenly
/// rounded against 255 instead. `multi` lifts a single 1-code jump's Prewitt magnitude of `3` to
/// `177.76`, which sits clearly below the 1023 rounding and clearly above the 255 rounding, so the
/// two roundings give different masks at the jump.
const TEN_BIT_LTHR: f32 = 0.1745;
const TEN_BIT_MULTI: f32 = 59.25294;

#[test]
fn ten_bit_input_scales_its_thresholds() {
    let width = 10;
    let height = 4;
    let peak = 1023.0;
    let luma: Vec<u16> = horizontal_steps(width, height, &[(5u32, 500u16), (5, 501)]);
    let code_values: Vec<f32> = luma.iter().map(|sample| f32::from(*sample)).collect();

    let low_at_1023 = (TEN_BIT_LTHR * 1023.0).round();
    let low_at_255 = (TEN_BIT_LTHR * 255.0).round();

    let raw_limits = limits(1.0, 0.0, 1.0, peak);
    let raw_values = reference_edges(&code_values, width, height, Operator::Prewitt, raw_limits);
    let jump_magnitude = raw_values[4];
    let scaled_magnitude = jump_magnitude * TEN_BIT_MULTI;

    assert_eq!(
        jump_magnitude, 3.0,
        "the jump's raw magnitude was {jump_magnitude}, expected 3"
    );

    let margin_to_255 = scaled_magnitude - low_at_255;
    let margin_to_1023 = low_at_1023 - scaled_magnitude;
    assert!(
        margin_to_255 > 1.0,
        "too little margin to the 255 rounding: {margin_to_255}"
    );
    assert!(
        margin_to_1023 > 1.0,
        "too little margin to the 1023 rounding: {margin_to_1023}"
    );

    let correct_limits = limits(TEN_BIT_MULTI, TEN_BIT_LTHR, 1.0, 1023.0);
    let wrong_limits = limits(TEN_BIT_MULTI, TEN_BIT_LTHR, 1.0, 255.0);
    let correct_values = reference_edges(&code_values, width, height, Operator::Prewitt, correct_limits);
    let wrong_values = reference_edges(&code_values, width, height, Operator::Prewitt, wrong_limits);
    let correct = store(&correct_values, 1023.0);
    let wrong = store(&wrong_values, 255.0);
    assert_ne!(
        correct, wrong,
        "rounding at 1023 gave the same mask as rounding at 255"
    );

    let format = ten_bit_grey(width, height);
    let wide_plane = wide_bytes(&luma);
    let frames = vec![vec![wide_plane]];
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

/// Neighbours of one target pixel arranged so its Kirsch magnitude is exactly `15 * (bright - dim)`
/// in 8-bit code values: `top_left`, `top` and `top_right` at `bright`, every other pixel at `dim`.
/// Kirsch's integer inputs keep this exact in f32, so it lands precisely on a rounded threshold.
fn kirsch_tie_fixture(width: u32, bright: u8, dim: u8) -> Vec<u8> {
    let width = width as usize;
    let mut luma = vec![dim; width * 5];
    luma[width + 1] = bright;
    luma[width + 2] = bright;
    luma[width + 3] = bright;

    luma
}

const KIRSCH_TIE_MAGNITUDE: f32 = 90.0;
const KIRSCH_TIE_INDEX: usize = 12;

#[test]
fn a_kirsch_tie_at_lthr_is_kept() {
    let width = 5;
    let height = 5;
    let luma = kirsch_tie_fixture(width, 156, 150);
    let lthr = KIRSCH_TIE_MAGNITUDE / 255.0;

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run("kirsch", json!({ "lthr": lthr }), format, frames);

    assert_eq!(
        masks[0][KIRSCH_TIE_INDEX], 90,
        "a magnitude tied to lthr was not kept, got {}",
        masks[0][KIRSCH_TIE_INDEX]
    );
}

#[test]
fn a_kirsch_tie_at_hthr_is_kept() {
    let width = 5;
    let height = 5;
    let luma = kirsch_tie_fixture(width, 134, 128);
    let hthr = KIRSCH_TIE_MAGNITUDE / 255.0;

    let format = grey(width, height);
    let frames = vec![vec![luma]];
    let masks = run("kirsch", json!({ "hthr": hthr }), format, frames);

    assert_eq!(
        masks[0][KIRSCH_TIE_INDEX], 90,
        "a magnitude tied to hthr was not kept, got {}",
        masks[0][KIRSCH_TIE_INDEX]
    );
}

/// Runs `store_mask` on one sample, directly on the device.
struct StoreMaskClamp {
    value: f32,
    peak: f32,
}

impl BuildTask for StoreMaskClamp {
    type Output = Vec<u8>;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let context = build.frame_context();
        let input_handle = upload_f32(client, &[self.value]);
        let output_handle = empty(client, 1);
        let input_plane = plane(&input_handle, 1, 1);
        let output_plane = plane(&output_handle, 1, 1);

        store_mask(client, &context, input_plane, output_plane, self.peak);

        read_u8(client, &output_handle)
    }
}

#[test]
fn store_mask_clamps_above_peak_to_255() {
    let peak = 40.0;
    let task = StoreMaskClamp {
        value: peak * 3.0,
        peak,
    };
    let backend = backend();
    let output = backend.build(task);

    assert_eq!(output, vec![255]);
}

#[test]
fn store_mask_clamps_below_zero_to_0() {
    let peak = 40.0;
    let task = StoreMaskClamp { value: -peak, peak };
    let backend = backend();
    let output = backend.build(task);

    assert_eq!(output, vec![0]);
}
