use av_toolbox_cubek::depth::{
    Conversion,
    DepthPlanes,
    Element,
    Shift,
    Thresholds,
    convert,
    plane_shift,
    upload_pattern,
};
use av_toolbox_cubek::testing::reference_depth;
use av_toolbox_gpu::{BuildContext, BuildTask};
use cubecl::prelude::Runtime;

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

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let frame_context = build.frame_context();
        let sizing = Sizing(frame_context);
        let output_sample_bytes = match self.conversion.output {
            Element::U8 => 1,
            Element::U16 => 2,
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

fn widen(samples: &[u16]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
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

fn convert_on_device(
    input: &[u16],
    width: u32,
    height: u32,
    conversion: &Conversion,
    thresholds: &Thresholds,
    shift: Shift,
) -> Vec<u16> {
    let bytes = match conversion.input {
        Element::U8 => input.iter().map(|value| *value as u8).collect(),
        Element::U16 => widen(input),
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
        Element::U8 => output.iter().map(|value| u16::from(*value)).collect(),
        Element::U16 => unwiden(&output),
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
        (8, 10, false),
        (10, 8, false),
        (8, 16, true),
        (16, 8, true),
        (12, 10, false),
    ];

    for thresholds in &modes {
        for (input_bits, output_bits, full_range) in changes {
            let conversion = Conversion::new(input_bits, output_bits, full_range);
            let input = gradient(96, 80, input_bits);
            let shift = plane_shift(1);

            let expected = reference_depth(&input, 96, &conversion, thresholds, shift);
            let converted = convert_on_device(&input, 96, 80, &conversion, thresholds, shift);

            assert_eq!(
                converted, expected,
                "{input_bits} to {output_bits}, full range {full_range}"
            );
        }
    }
}

/// Dithering keeps the average of a flat area where the extra bits put it.
#[test]
fn blue_noise_keeps_a_flat_areas_mean() {
    let conversion = Conversion::new(10, 8, false);
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
