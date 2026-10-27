mod darken;
mod thin;
mod toon;
mod vinverse;

use av_toolbox_gpu::{Channel, ChromaLocation, Depth, FrameFormat, FrameLayout, SampleFormat, Subsampling};

use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    f32_parity,
    frame,
    run_into_codes,
    run_masked_into_codes,
    widened,
};

pub(super) const WIDTH: u32 = 64;

pub(super) const HEIGHT: u32 = 48;

pub(super) fn yuv420(sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(WIDTH, HEIGHT, layout, sample, ChromaLocation::Left)
}

/// Runs `stage` on one `format` frame into F32, and reads every plane back as codes of the input's depth.
pub(super) fn run_codes(
    stage: &str,
    format: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
) -> Vec<Vec<f32>> {
    let bits = bits_of(format.sample);

    run_into_codes(stage, format, args, planes, bits)
}

/// Runs `stage` on one `format` frame with `mask` into F32, and reads every plane back as codes of the
/// input's depth.
pub(super) fn run_masked_codes(
    stage: &str,
    format: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    mask: (Vec<u8>, Vec<u8>),
) -> Vec<Vec<f32>> {
    let bits = bits_of(format.sample);

    run_masked_into_codes(stage, format, args, planes, mask, bits)
}

pub(super) fn sixteen_bit() -> SampleFormat {
    SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    }
}

pub(super) fn bits_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 8,
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        SampleFormat::F32 => 16,
    }
}

pub(super) fn peak_of(sample: SampleFormat) -> u32 {
    (1 << bits_of(sample)) - 1
}

/// Whether the code `found` is more than a round trip through F32 away from `code`.
pub(super) fn moved(found: f32, code: u16) -> bool {
    (found - f32::from(code)).abs() > ROUND_TRIP
}

/// The samples of `found` that moved away from `input`.
pub(super) fn changed(found: &[f32], input: &[u16]) -> usize {
    found
        .iter()
        .zip(input)
        .filter(|(sample, code)| moved(**sample, **code))
        .count()
}

/// Checks `found` matches the reference's `expected` codes, and prints the largest difference.
pub(super) fn assert_matches(found: &[f32], expected: &[f32], label: &str) {
    let largest = found
        .iter()
        .zip(expected)
        .map(|(sample, wanted)| (sample - wanted).abs())
        .fold(0.0, f32::max);
    println!("{label}: largest difference {largest}");

    assert_codes_within(found, expected, ROUND_TRIP, label);
}

/// Checks every plane of `found` still holds the codes of `values`.
pub(super) fn assert_unchanged(found: &[Vec<f32>], values: &[Vec<u16>], label: &str) {
    for (plane, (samples, codes)) in found.iter().zip(values).enumerate() {
        let expected = widened(codes);
        let plane_label = format!("{label} plane {plane}");
        assert_codes_within(samples, &expected, ROUND_TRIP, &plane_label);
    }
}

/// Checks `stage`'s F32 and 16-bit runs on `values` differ by at most `largest` codes, with no more than a
/// `beyond_one` share of samples more than one code apart.
pub(super) fn assert_f32_matches_sixteen_bit(
    stage: &str,
    args: serde_json::Value,
    values: &[Vec<u16>],
    largest: f32,
    beyond_one: f32,
) {
    let input = yuv420(sixteen_bit());

    let parity = f32_parity(stage, input, args, values);
    println!("{stage}: {parity:?}");

    assert!(parity.largest <= largest, "{stage}: {parity:?}");
    assert!(parity.beyond_one <= beyond_one, "{stage}: {parity:?}");
}

/// Every plane of `format` held at `flat`.
pub(super) fn flat_values(format: &FrameFormat, flat: u16) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            vec![flat; samples]
        })
        .collect()
}

/// A scene of three bands of rows, scaled to the sample's depth.
///
/// - The top band is a flat 200 with 1-wide lines. Three of them sit 1 code below, on and 1 code
///   above `cap - thr`, so their closing straddles `x + thr`. The others are deeper.
/// - The middle band is hashed noise, which puts lines of every depth in every direction.
/// - The bottom band is a flat 230, above the cap, with dark 1-wide and 2-wide lines.
pub(super) fn lined_scene(format: &FrameFormat) -> Vec<Vec<u16>> {
    let sample = format.sample;
    let peak = peak_of(sample);
    let scale = (peak + 1) / 256;
    let cap = 191 * scale;
    let thr = 4 * scale;
    let depths = [
        cap - thr - 1,
        cap - thr,
        cap - thr + 1,
        150 * scale,
        100 * scale,
        20 * scale,
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
                        0 if column % 4 == 2 => depths[(column / 4) as usize % depths.len()],
                        0 => 200 * scale,
                        1 => {
                            let hash = (index + plane as u32 * 7).wrapping_mul(2_654_435_761) >> 20;
                            (60 + hash % 140) * scale
                        },
                        _ if column % 8 == 3 => 150 * scale,
                        _ if column % 8 == 6 || column % 8 == 7 => 100 * scale,
                        _ => 230 * scale,
                    };

                    value.min(peak) as u16
                })
                .collect()
        })
        .collect()
}

/// A `yuv420` frame whose planes are `background`, with a 1-wide vertical line of `line` down the
/// middle column of the luma plane.
pub(super) fn faded_line_values(format: &FrameFormat, background: u16, line: u16) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            (0..width * height)
                .map(|index| {
                    let on_line = *channel == Channel::Y && index % width == width / 2;
                    if on_line { line } else { background }
                })
                .collect()
        })
        .collect()
}
