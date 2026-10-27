use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::values::FloatCodes;
use av_toolbox_gpu::{Channel, ChromaLocation, ColourRange, Depth, FrameFormat, FrameLayout, SampleFormat};
use serde_json::json;

use super::{element_of, encoded, frame, from_bytes, run_masked_into, run_once, widened};

/// F32 samples read from a plane's bytes.
pub(crate) fn float_samples(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();

    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

/// A plane's bytes holding `samples` as F32.
pub(crate) fn float_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

/// `bits` deep code values in `range` as full range float samples.
pub(crate) fn as_float(codes: &[u16], bits: u32, range: SampleRange, chroma: bool) -> Vec<f32> {
    let mapping = FloatCodes::new(bits, range, chroma);

    codes
        .iter()
        .map(|code| (f32::from(*code) - mapping.offset) / mapping.scale)
        .collect()
}

/// Full range float samples as unrounded `bits` deep code values in `range`.
pub(crate) fn as_codes(samples: &[f32], bits: u32, range: SampleRange, chroma: bool) -> Vec<f32> {
    let mapping = FloatCodes::new(bits, range, chroma);

    samples
        .iter()
        .map(|sample| sample * mapping.scale + mapping.offset)
        .collect()
}

/// Checks every code of `found` is within `tolerance` of `expected`, naming `label` in a failure.
pub(crate) fn assert_codes_within(found: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(found.len(), expected.len(), "{label}");

    for (index, (got, want)) in found.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() <= tolerance,
            "{label}: sample {index} was {got}, expected {want} within {tolerance}"
        );
    }
}

/// The depth of the codes `sample` maps onto. F32 maps onto 16-bit codes.
pub(crate) fn code_bits(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 8,
        SampleFormat::U16 { depth } => u32::from(depth.bits()),
        SampleFormat::F32 => 16,
    }
}

/// The integer range `format` returns to. Anything not labelled full range is limited.
pub(crate) fn range_of(format: &FrameFormat) -> SampleRange {
    match format.colour.range {
        ColourRange::Full => SampleRange::Full,
        _ => SampleRange::Limited,
    }
}

fn is_chroma(channel: Channel) -> bool {
    matches!(channel, Channel::U | Channel::V)
}

/// Every plane of `values`, `bits` deep codes of `format`, as F32 samples.
pub(crate) fn float_planes(format: &FrameFormat, values: &[Vec<u16>], bits: u32) -> Vec<Vec<u8>> {
    let range = range_of(format);

    format
        .channels()
        .iter()
        .zip(values)
        .map(|(channel, plane)| {
            let chroma = is_chroma(*channel);
            let samples = as_float(plane, bits, range, chroma);
            float_bytes(&samples)
        })
        .collect()
}

/// Runs `stage` on one frame into F32, and reads every output plane back as `bits` deep codes.
pub(crate) fn run_into_codes(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    bits: u32,
) -> Vec<Vec<f32>> {
    let output = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };

    let found = run_once(stage, input, output, args, planes);

    planes_as_codes(&output, &found, bits)
}

/// Runs `stage` on one frame into the input's own sample, and reads every output plane back as codes.
fn run_kept_into_codes(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
) -> Vec<Vec<f32>> {
    let found = run_once(stage, input, input, args, planes);

    planes_as_codes(&input, &found, 16)
}

/// Runs `stage` on `planes` with a mask of the full and chroma bytes in `mask` into F32, and reads
/// every output plane back as `bits` deep codes.
pub(crate) fn run_masked_into_codes(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    mask: (Vec<u8>, Vec<u8>),
    bits: u32,
) -> Vec<Vec<f32>> {
    let output = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };

    let found = run_masked_into(stage, input, output, args, planes, mask);

    planes_as_codes(&output, &found, bits)
}

/// The planes of a `format` frame, held in `planes`, as `bits` deep codes.
///
/// Integer samples are their own codes. F32 samples map onto codes in the range `format` is labelled with.
pub(crate) fn planes_as_codes(format: &FrameFormat, planes: &[Vec<u8>], bits: u32) -> Vec<Vec<f32>> {
    let range = range_of(format);
    let element = element_of(format.sample);

    format
        .channels()
        .iter()
        .zip(planes)
        .map(|(channel, bytes)| {
            if format.sample != SampleFormat::F32 {
                let codes = from_bytes(bytes, element);
                return widened(&codes);
            }

            let chroma = is_chroma(*channel);
            let samples = float_samples(bytes);
            as_codes(&samples, bits, range, chroma)
        })
        .collect()
}

/// How far a stage's F32 run strays from its 16-bit run, read as 16-bit codes.
#[derive(Debug)]
pub(crate) struct Parity {
    /// The largest difference over every plane.
    pub(crate) largest: f32,
    /// The share of samples differing by more than one code.
    pub(crate) beyond_one: f32,
}

impl Parity {
    /// Checks the largest difference is at most `largest` codes and no more than a `beyond_one`
    /// share of samples are more than one code apart, naming `stage` in a failure.
    pub(crate) fn assert_within(&self, stage: &str, largest: f32, beyond_one: f32) {
        assert!(self.largest <= largest, "{stage}: {self:?}");
        assert!(self.beyond_one <= beyond_one, "{stage}: {self:?}");
    }

    /// How far `float` strays from `sixteen_bit`, both planes of codes.
    pub(crate) fn between(sixteen_bit: &[Vec<f32>], float: &[Vec<f32>]) -> Self {
        let differences: Vec<f32> = sixteen_bit
            .iter()
            .flatten()
            .zip(float.iter().flatten())
            .map(|(integer_code, float_code)| (integer_code - float_code).abs())
            .collect();
        let largest = differences.iter().copied().fold(0.0, f32::max);
        let beyond = differences.iter().filter(|difference| **difference > 1.0).count();
        let beyond_one = beyond as f32 / differences.len() as f32;

        Self { largest, beyond_one }
    }
}

/// Runs `stage` on `values` as 16-bit codes of `input` and as the same codes in F32, and compares
/// the two outputs.
pub(crate) fn f32_parity(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    values: &[Vec<u16>],
) -> Parity {
    let float_input = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };
    let integer_planes = encoded(&input, values);
    let float_input_planes = float_planes(&float_input, values, 16);

    let sixteen_bit = run_into_codes(stage, input, args.clone(), &integer_planes, 16);
    let float = run_into_codes(stage, float_input, args, &float_input_planes, 16);

    Parity::between(&sixteen_bit, &float)
}

/// The same as [f32_parity], for a stage that keeps 16-bit samples as 16-bit.
pub(crate) fn kept_sample_parity(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    values: &[Vec<u16>],
) -> Parity {
    let float_input = FrameFormat {
        sample: SampleFormat::F32,
        ..input
    };
    let integer_planes = encoded(&input, values);
    let float_input_planes = float_planes(&float_input, values, 16);

    let sixteen_bit = run_kept_into_codes(stage, input, args.clone(), &integer_planes);
    let float = run_into_codes(stage, float_input, args, &float_input_planes, 16);

    Parity::between(&sixteen_bit, &float)
}

/// How far a code may move on its way into F32 and back.
pub(crate) const ROUND_TRIP: f32 = 0.02;

/// Runs `stage` on `values`, `bits` deep codes of `input`, into F32. Checks each selected plane is
/// within one code of `reference`, and the rest are the input.
pub(crate) fn assert_planes_match(
    stage: &str,
    input: FrameFormat,
    args: serde_json::Value,
    bits: u32,
    selected: &[bool],
    values: &[Vec<u16>],
    reference: impl Fn(&[f32], u32, u32) -> Vec<f32>,
) {
    let planes = encoded(&input, values);

    let found = run_into_codes(stage, input, args, &planes, bits);

    for (plane, channel) in input.channels().iter().enumerate() {
        let (width, height) = input.channel_dimensions(*channel).expect("carried");
        let codes = widened(&values[plane]);
        let label = channel.to_string();

        if !selected[plane] {
            let untouched = format!("{label} should be untouched");
            assert_codes_within(&found[plane], &codes, ROUND_TRIP, &untouched);
            continue;
        }

        let expected = reference(&codes, width, height);
        assert_codes_within(&found[plane], &expected, 1.0, &label);
    }
}

/// Runs `stage` on a flat 16-bit frame at its peak and checks it stays there.
pub(crate) fn assert_a_flat_frame_is_unchanged(stage: &str) {
    let sixteen_bit = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };
    let input = frame(64, 48, FrameLayout::Grey, sixteen_bit, ChromaLocation::Left);
    let samples = input.channel_samples(Channel::Y).expect("a listed channel");
    let values = vec![vec![65535u16; samples]];
    let planes = encoded(&input, &values);
    let peak = widened(&values[0]);

    let found = run_into_codes(stage, input, json!({}), &planes, 16);

    assert_codes_within(&found[0], &peak, ROUND_TRIP, stage);
}
