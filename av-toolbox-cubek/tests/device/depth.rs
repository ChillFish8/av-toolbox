use av_toolbox_cubek::depth::{
    Conversion,
    DepthPlanes,
    Destination,
    Shift,
    Source,
    Thresholds,
    convert,
    convert_to_float_sample,
    plane_shift,
    upload_pattern,
};
use av_toolbox_cubek::matrix::SampleRange;
use av_toolbox_cubek::testing::{reference_depth, reference_float_depth};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_u8, upload_u8};

struct Depth {
    /// The input plane's bytes, little endian for U16.
    input: Vec<u8>,
    width: u32,
    height: u32,
    conversion: Conversion,
    thresholds: Thresholds,
    shift: Shift,
}

impl BuildTask for Depth {
    type Output = Vec<u8>;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let output_sample_bytes = match self.conversion.output {
            Destination::U8 => 1,
            Destination::U16 => 2,
            Destination::F32 => 4,
        };
        let samples = (self.width * self.height) as usize;
        let output_bytes = samples * output_sample_bytes;

        let input = upload_u8(client, &self.input);
        let output = empty(client, output_bytes);
        let pattern = upload_pattern(client, &self.thresholds);
        let input_plane = plane(&input, self.width, self.height);
        let output_plane = plane(&output, self.width, self.height);
        let planes = DepthPlanes {
            input: input_plane,
            output: output_plane,
        };
        convert(client, &sizing, planes, &self.conversion, &pattern, self.shift);

        read_u8(client, &output)
    }
}

fn unwiden(bytes: &[u8]) -> Vec<u16> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

/// A plane `width` samples across and `height` down, filling every value of `bits`.
fn gradient(width: u32, height: u32, bits: u32) -> Vec<u16> {
    let modulus = 1u32 << bits;

    (0..width * height)
        .map(|index| ((index * 7) % modulus) as u16)
        .collect()
}

/// The mean of one `size` by `size` block, starting at `start_x` and `start_y`.
fn block_mean(values: &[u16], width: u32, start_x: u32, start_y: u32, size: u32) -> f64 {
    let mut total = 0u64;

    for row in start_y..start_y + size {
        for column in start_x..start_x + size {
            total += u64::from(values[(row * width + column) as usize]);
        }
    }

    total as f64 / f64::from(size * size)
}

fn float_bytes(samples: &[f32]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

fn convert_on_device(
    input: &[u16],
    width: u32,
    height: u32,
    conversion: &Conversion,
    thresholds: &Thresholds,
    shift: Shift,
) -> Vec<u16> {
    let floats: Vec<f32> = input.iter().map(|value| f32::from(*value)).collect();

    convert_float_on_device(&floats, width, height, conversion, thresholds, shift)
}

/// Converts `input` on the device, stored as the conversion's input samples.
fn convert_float_on_device(
    input: &[f32],
    width: u32,
    height: u32,
    conversion: &Conversion,
    thresholds: &Thresholds,
    shift: Shift,
) -> Vec<u16> {
    let bytes = match conversion.input {
        Source::U8 => input.iter().map(|value| *value as u8).collect(),
        Source::U16 => input
            .iter()
            .flat_map(|value| (*value as u16).to_le_bytes())
            .collect(),
        Source::F32 => float_bytes(input),
    };

    let output = backend().build(Depth {
        input: bytes,
        width,
        height,
        conversion: *conversion,
        thresholds: thresholds.clone(),
        shift,
    });

    match conversion.output {
        Destination::U8 => output.iter().map(|value| u16::from(*value)).collect(),
        Destination::U16 => unwiden(&output),
        Destination::F32 => panic!("float output is read with read_floats"),
    }
}

/// Every rounding mode, raising and lowering, against the CPU reference.
#[test]
fn every_mode_matches_the_reference() {
    let modes = [
        Thresholds::rounding(),
        Thresholds::ordered(),
        Thresholds::blue_noise(),
    ];
    let changes = [
        (8, 10, SampleRange::Limited, false),
        (10, 8, SampleRange::Limited, false),
        (8, 16, SampleRange::Full, false),
        (16, 8, SampleRange::Full, false),
        (12, 10, SampleRange::Limited, false),
        (8, 10, SampleRange::Full, true),
        (10, 8, SampleRange::Full, true),
    ];

    for thresholds in &modes {
        for (input_bits, output_bits, range, chroma) in changes {
            let conversion = Conversion::new(input_bits, output_bits, range, chroma);
            let input = gradient(96, 80, input_bits);
            let shift = plane_shift(1);

            let expected = reference_depth(&input, 96, &conversion, thresholds, shift);
            let converted = convert_on_device(&input, 96, 80, &conversion, thresholds, shift);

            assert_eq!(
                converted, expected,
                "{input_bits} to {output_bits}, {range:?}, chroma {chroma}"
            );
        }
    }
}

/// Dithering keeps the average of a flat area where the extra bits put it.
#[test]
fn blue_noise_keeps_a_flat_areas_mean() {
    let conversion = Conversion::new(10, 8, SampleRange::Limited, false);
    let thresholds = Thresholds::blue_noise();
    let shift = plane_shift(0);

    for value in [401u16, 402, 403] {
        let input = vec![value; 128 * 128];
        let converted = convert_on_device(&input, 128, 128, &conversion, &thresholds, shift);
        let true_mean = f64::from(value) / 4.0;

        for block_y in 0..2 {
            for block_x in 0..2 {
                let mean = block_mean(&converted, 128, block_x * 64, block_y * 64, 64);
                assert!(
                    (mean - true_mean).abs() < 0.5,
                    "{value}: {mean} against {true_mean}"
                );
            }
        }
    }
}

/// A float plane running a little past both ends of `low..=low + 1`, with a NaN every 97 samples.
fn float_gradient(width: u32, height: u32, low: f32) -> Vec<f32> {
    (0..width * height)
        .map(|index| {
            if index % 97 == 0 {
                return f32::NAN;
            }

            let step = (index * 7) % 300;
            step as f32 / 256.0 - 0.1 + low
        })
        .collect()
}

/// Asserts every sample is within one of the reference.
///
/// The device may fuse the multiply and add of a float conversion into one rounding, which moves a
/// sample sitting right on a rounding step by one.
fn assert_within_one(converted: &[u16], expected: &[u16], label: &str) {
    assert_eq!(converted.len(), expected.len(), "{label}");

    for (index, (found, wanted)) in converted.iter().zip(expected).enumerate() {
        assert!(
            found.abs_diff(*wanted) <= 1,
            "{label}: sample {index} was {found}, expected {wanted}"
        );
    }
}

/// Every rounding mode, on float luma and chroma, against the CPU reference.
#[test]
fn every_float_conversion_matches_the_reference() {
    let modes = [
        Thresholds::rounding(),
        Thresholds::ordered(),
        Thresholds::blue_noise(),
    ];
    let changes = [(8, false), (8, true), (10, false), (16, true)];

    for thresholds in &modes {
        for (output_bits, chroma) in changes {
            let conversion = Conversion::from_float(output_bits, SampleRange::Full, chroma);
            let low = if chroma { -0.5 } else { 0.0 };
            let input = float_gradient(96, 80, low);
            let shift = plane_shift(1);

            let expected = reference_float_depth(&input, 96, &conversion, thresholds, shift);
            let converted = convert_float_on_device(&input, 96, 80, &conversion, thresholds, shift);

            let label = format!("{output_bits} bits, chroma {chroma}");
            assert_within_one(&converted, &expected, &label);
        }
    }
}

/// Converts a U8 plane to F32 on the device.
#[test]
fn u8_planes_convert_to_floats_like_the_reference() {
    let conversion = Conversion::to_float(8, SampleRange::Limited, false);
    let input: Vec<u8> = (0..=255).collect();
    let thresholds = Thresholds::rounding();

    let output = backend().build(Depth {
        input: input.clone(),
        width: 256,
        height: 1,
        conversion,
        thresholds,
        shift: plane_shift(0),
    });

    let (chunks, _) = output.as_chunks::<4>();
    let converted: Vec<f32> = chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
    for (code, found) in input.iter().zip(&converted) {
        let expected = convert_to_float_sample(f32::from(*code), &conversion);
        assert!(
            (found - expected).abs() <= 1e-6,
            "code {code} was {found}, expected {expected}"
        );
    }
}

/// Converts ten-bit chroma to F32 on the device.
#[test]
fn ten_bit_chroma_converts_to_centred_floats_like_the_reference() {
    let ranges = [SampleRange::Limited, SampleRange::Full];

    for range in ranges {
        let conversion = Conversion::to_float(10, range, true);
        let input: Vec<u16> = (0..1024).collect();
        let bytes = input.iter().flat_map(|value| value.to_le_bytes()).collect();
        let thresholds = Thresholds::rounding();

        let output = backend().build(Depth {
            input: bytes,
            width: 1024,
            height: 1,
            conversion,
            thresholds,
            shift: plane_shift(1),
        });

        let (chunks, _) = output.as_chunks::<4>();
        let converted: Vec<f32> = chunks.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();
        for (code, found) in input.iter().zip(&converted) {
            let expected = convert_to_float_sample(f32::from(*code), &conversion);
            assert!(
                (found - expected).abs() <= 1e-6,
                "{range:?} code {code} was {found}, expected {expected}"
            );
        }

        assert_eq!(converted[512], 0.0, "{range:?}");
    }
}
