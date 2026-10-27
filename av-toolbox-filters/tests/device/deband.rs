use av_toolbox_cubek::deband::{DebandParams, salt};
use av_toolbox_cubek::depth::{Element, Source};
use av_toolbox_cubek::testing::{reference_deband, reference_deband_codes};
use av_toolbox_gpu::{ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};
use serde_json::json;

use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    encoded,
    f32_parity,
    float_planes,
    frame,
    from_bytes,
    luma_is_shared,
    passes_through,
    planes_as_codes,
    run_into_codes,
    run_once,
    run_once_at,
    widened,
};

fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

fn as_f32(format: FrameFormat) -> FrameFormat {
    FrameFormat {
        sample: SampleFormat::F32,
        ..format
    }
}

/// Every plane of `format`, a diagonal ramp wrapped at `peak`, as u16 values.
fn ramps(format: &FrameFormat, peak: u32) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let column = index % width;
                    let row = index / width;
                    (((column * 3 + row * 5) / 8) % (peak + 1)) as u16
                })
                .collect()
        })
        .collect()
}

/// Checks at least 99.9% of `actual` is within 1 code of `expected`.
fn assert_mostly_within_one(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}");

    let close = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| (*got - *want).abs() <= 1.0)
        .count();
    let total = expected.len();
    let needed = (total as f64 * 0.999).ceil() as usize;

    assert!(close >= needed, "{label}: {close} of {total} samples within 1");
}

const THRESHOLDS: [f32; 3] = [6.0, 4.0, 4.0];
const GRAINS: [f32; 3] = [4.0, 2.0, 2.0];

fn reference_args() -> serde_json::Value {
    json!({
        "iterations": 2,
        "threshold": THRESHOLDS,
        "grain": GRAINS,
        "seed": 11,
    })
}

/// What `deband/placebo` debands one plane of a limited range clip with, under [reference_args].
fn reference_params(plane: usize, peak: u32) -> DebandParams {
    let neutrals = [
        16.0 * (peak + 1) as f32 / 256.0,
        (peak + 1) as f32 / 2.0,
        (peak + 1) as f32 / 2.0,
    ];

    DebandParams {
        element: Source::U16,
        iterations: 2,
        radius: 16.0,
        threshold: THRESHOLDS[plane],
        grain: GRAINS[plane],
        neutral: neutrals[plane],
        peak: peak as f32,
    }
}

/// Runs `deband/placebo` on a ramp clip at `index` and checks every plane against the unrounded
/// reference.
fn matches_reference(input: FrameFormat, bits: u32, index: u64) {
    let peak = (1u32 << bits) - 1;
    let values = ramps(&input, peak);
    let planes = encoded(&input, &values);
    let output = as_f32(input);

    let debanded = run_once_at("deband/placebo", input, output, reference_args(), &planes, index);
    let codes = planes_as_codes(&output, &debanded, bits);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let params = reference_params(plane, peak);
        let plane_salt = salt(index, 11, plane as u32);
        let expected = reference_deband_codes(&values[plane], width, height, &params, plane_salt);
        let label = channel.to_string();

        assert_mostly_within_one(&codes[plane], &expected, &label);
    }
}

#[test]
fn eight_and_ten_bit_clips_match_the_reference() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };

    let eight_bit_input = yuv420(64, 32, SampleFormat::U8);
    matches_reference(eight_bit_input, 8, 0);

    let ten_bit_input = yuv420(64, 32, ten_bit);
    matches_reference(ten_bit_input, 10, 7);
}

#[test]
fn an_odd_sized_clip_matches_the_reference() {
    let input = yuv420(33, 17, SampleFormat::U8);
    matches_reference(input, 8, 3);
}

#[test]
fn eight_bit_output_after_a_depth_change_matches_the_rounded_reference() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let values = ramps(&input, 255);
    let planes = encoded(&input, &values);

    let output = as_f32(input);

    let debanded = run_once_at("deband/placebo", input, output, reference_args(), &planes, 0);
    let lowered_args = json!({ "depth": 8, "dither": "none" });
    let lowered = run_once("resample/depth", output, input, lowered_args, &debanded);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let params = reference_params(plane, 255);
        let plane_salt = salt(0, 11, plane as u32);
        let reference = reference_deband(&values[plane], width, height, &params, plane_salt);
        let expected = widened(&reference);
        let found_codes = from_bytes(&lowered[plane], Element::U8);
        let found = widened(&found_codes);
        let label = channel.to_string();

        assert_mostly_within_one(&found, &expected, &label);
    }
}

/// Every plane of `format`, a limited range 16-bit ramp.
fn sixteen_bit_ramps(format: &FrameFormat) -> Vec<Vec<u16>> {
    let narrow = ramps(format, 255);

    narrow
        .iter()
        .map(|plane| plane.iter().map(|value| 4096 + value * 211).collect())
        .collect()
}

#[test]
fn deband_f32_matches_sixteen_bit() {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = yuv420(64, 32, sixteen_bit);
    let values = sixteen_bit_ramps(&input);

    let parity = f32_parity("deband/placebo", input, reference_args(), &values);

    println!("deband twin: {parity:?}");
    parity.assert_within("deband/placebo", 0.01, 0.0);
}

#[test]
fn the_grain_survives_without_rounding() {
    let input = FrameFormat {
        layout: FrameLayout::Grey,
        ..yuv420(64, 32, SampleFormat::U8)
    };
    let samples = 64 * 32;
    let values = vec![vec![128u16; samples]];
    let planes = encoded(&input, &values);

    let codes = run_into_codes("deband/placebo", input, json!({ "grain": 20.0 }), &planes, 8);

    let fractional = codes[0]
        .iter()
        .filter(|code| (*code - code.round()).abs() > 0.01)
        .count();
    assert!(
        fractional > samples / 2,
        "{fractional} of {samples} codes off whole values"
    );
}

#[test]
fn the_frame_index_moves_the_grain() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let output = as_f32(input);
    let planes: Vec<Vec<u8>> = input
        .channels()
        .iter()
        .map(|channel| {
            let samples = input.channel_samples(*channel).expect("a listed channel");
            vec![128u8; samples]
        })
        .collect();
    let args = json!({ "grain": 20.0 });

    let first_args = args.clone();
    let first = run_once_at("deband/placebo", input, output, first_args, &planes, 4);

    let again_args = args.clone();
    let again = run_once_at("deband/placebo", input, output, again_args, &planes, 4);

    let next = run_once_at("deband/placebo", input, output, args, &planes, 5);

    assert_eq!(first, again);
    assert_ne!(first[0], next[0]);
}

#[test]
fn a_skipped_plane_is_shared() {
    let input = yuv420(64, 32, SampleFormat::F32);
    let args = json!({ "threshold": [0.0, 3.0, 3.0], "grain": [0.0, 4.0, 4.0] });

    assert!(luma_is_shared("deband/placebo", input, input, args));
}

#[test]
fn a_skipped_integer_plane_is_converted() {
    let input = yuv420(64, 32, SampleFormat::U8);
    let values = ramps(&input, 255);
    let planes = encoded(&input, &values);
    let args = json!({ "threshold": [0.0, 3.0, 3.0], "grain": [0.0, 4.0, 4.0] });

    let codes = run_into_codes("deband/placebo", input, args, &planes, 8);

    let luma = widened(&values[0]);
    assert_codes_within(&codes[0], &luma, ROUND_TRIP, "Y");
}

#[test]
fn a_frame_with_every_plane_skipped_passes_through_in_its_sample() {
    for sample in [SampleFormat::U8, SampleFormat::F32] {
        let input = yuv420(64, 32, sample);
        let args = json!({ "threshold": 0.0, "grain": 0.0 });
        let values = ramps(&input, 255);
        let planes = if sample == SampleFormat::F32 {
            float_planes(&input, &values, 8)
        } else {
            encoded(&input, &values)
        };

        let passed = passes_through("deband/placebo", input, input, args.clone());
        let found = run_once_at("deband/placebo", input, input, args, &planes, 0);

        assert!(passed, "{sample}");
        assert_eq!(found, planes, "{sample}");
    }
}
