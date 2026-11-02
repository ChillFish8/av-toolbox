use av_toolbox_cubek::depth::{Conversion, convert_to_float_sample};
use av_toolbox_cubek::matrix::{
    Coefficients,
    Direction,
    MatrixConversion,
    MatrixPlanes,
    MatrixSamples,
    SampleRange,
    convert,
    upload_affine,
};
use av_toolbox_cubek::testing::{
    MATRIX_FIXTURES,
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    decode_samples,
    encode_samples,
    reference_matrix,
};
use av_toolbox_gpu::{BuildContext, BuildTask};

use crate::support::{Sizing, backend, empty, plane, read_u8, upload_u8};

/// Converts three planes, given and returned as little-endian sample bytes, on the device.
struct Convert {
    conversion: MatrixConversion,
    planes: [Vec<u8>; 3],
    width: u32,
    height: u32,
}

impl BuildTask for Convert {
    type Output = [Vec<u8>; 3];

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let affine = self.conversion.affine();
        let values = upload_affine(client, &affine);
        let plane_samples = (self.width * self.height) as usize;
        let plane_bytes = plane_samples * sample_bytes(self.conversion.output_samples);

        let inputs = self.planes.each_ref().map(|bytes| upload_u8(client, bytes));
        let outputs = [(); 3].map(|_| empty(client, plane_bytes));
        let input_planes = inputs
            .each_ref()
            .map(|handle| plane(handle, self.width, self.height));
        let output_planes = outputs
            .each_ref()
            .map(|handle| plane(handle, self.width, self.height));
        let planes = MatrixPlanes {
            inputs: input_planes,
            outputs: output_planes,
        };

        convert(client, &sizing, planes, &self.conversion, &values);

        outputs.each_ref().map(|handle| read_u8(client, handle))
    }
}

fn converted(conversion: MatrixConversion, planes: [Vec<u8>; 3], width: u32, height: u32) -> Vec<f64> {
    let task = Convert {
        conversion,
        planes,
        width,
        height,
    };
    let written = backend().build(task);
    let bytes = written.concat();

    decode_samples(&bytes, conversion.output_samples)
}

fn sample_bytes(samples: MatrixSamples) -> usize {
    match samples {
        MatrixSamples::U8 => 1,
        MatrixSamples::U16 { .. } => 2,
        MatrixSamples::F32 => 4,
    }
}

fn split_bytes(bytes: &[u8]) -> [Vec<u8>; 3] {
    let plane = bytes.len() / 3;

    [0, 1, 2].map(|index| bytes[index * plane..(index + 1) * plane].to_vec())
}

fn tolerance(samples: MatrixSamples) -> f64 {
    match samples {
        MatrixSamples::F32 => 1e-5,
        _ => 1.0,
    }
}

fn assert_within(found: &[f64], expected: &[f64], bound: f64, label: &str) {
    for (index, (value, wanted)) in found.iter().zip(expected).enumerate() {
        let difference = (value - wanted).abs();

        assert!(
            difference <= bound,
            "{label}: sample {index} is {value}, expected {wanted}"
        );
    }
}

#[test]
fn the_kernel_matches_every_vapoursynth_fixture() {
    for fixture in MATRIX_FIXTURES {
        let conversion = fixture.conversion();
        let planes = split_bytes(fixture.input);
        let expected = decode_samples(fixture.output, conversion.output_samples);

        let found = converted(conversion, planes, MATRIX_WIDTH, MATRIX_HEIGHT);

        let bound = tolerance(conversion.output_samples);
        assert_within(&found, &expected, bound, fixture.name);
    }
}

#[test]
fn the_kernel_clamps_out_of_gamut_samples_like_the_reference() {
    let conversion = MatrixConversion {
        coefficients: Coefficients::Bt709,
        direction: Direction::ToRgb,
        input_samples: MatrixSamples::U16 { bits: 10 },
        output_samples: MatrixSamples::U16 { bits: 10 },
        input_range: SampleRange::Limited,
        output_range: SampleRange::Full,
    };
    let luma = [940.0, 64.0, 0.0, 1023.0];
    let blue = [64.0, 960.0, 0.0, 1023.0];
    let red = [960.0, 64.0, 1023.0, 0.0];
    let samples = conversion.input_samples;
    let planes = [&luma[..], &blue[..], &red[..]].map(|values| encode_samples(values, samples));

    let found = converted(conversion, planes, 4, 1);

    let expected = reference_matrix(&conversion, [&luma, &blue, &red]).concat();
    assert_within(&found, &expected, 1.0, "out of gamut");
    assert!(found.iter().all(|value| (0.0..=1023.0).contains(value)));
}

#[test]
fn a_float_round_trip_returns_the_input() {
    let luma: Vec<f64> = (0..64).map(|index| f64::from(index) / 63.0).collect();
    let blue: Vec<f64> = (0..64).map(|index| f64::from(index % 9) / 8.0 - 0.5).collect();
    let red: Vec<f64> = (0..64).map(|index| f64::from(index % 7) / 6.0 - 0.5).collect();
    let samples = MatrixSamples::F32;
    let forward = MatrixConversion {
        coefficients: Coefficients::Bt2020Ncl,
        direction: Direction::ToRgb,
        input_samples: samples,
        output_samples: samples,
        input_range: SampleRange::Full,
        output_range: SampleRange::Full,
    };
    let backward = MatrixConversion {
        direction: Direction::ToYuv,
        ..forward
    };
    let planes = [&luma, &blue, &red].map(|values| encode_samples(values, samples));

    let rgb = converted(forward, planes, 8, 8);
    let rgb_bytes = encode_samples(&rgb, samples);
    let rgb_planes = split_bytes(&rgb_bytes);
    let back = converted(backward, rgb_planes, 8, 8);

    let expected = [luma, blue, red].concat();
    assert_within(&back, &expected, 1e-5, "round trip");
}

/// Eight-bit limited YUV planes as full range floats, the way `resample/depth` converts them.
fn as_float(planes: &[Vec<f64>; 3]) -> [Vec<f64>; 3] {
    let chroma = [false, true, true];

    [0, 1, 2].map(|channel| {
        let conversion = Conversion::to_float(8, SampleRange::Limited, chroma[channel]);

        planes[channel]
            .iter()
            .map(|code| convert_to_float_sample(*code as f32, &conversion))
            .map(f64::from)
            .collect()
    })
}

#[test]
fn eight_bit_yuv_converts_straight_to_float_rgb() {
    let luma: Vec<f64> = (0..64).map(|index| f64::from(16 + index * 219 / 63)).collect();
    let blue: Vec<f64> = (0..64).map(|index| f64::from(16 + (index * 37) % 225)).collect();
    let red: Vec<f64> = (0..64).map(|index| f64::from(16 + (index * 53) % 225)).collect();
    let codes = [luma, blue, red];
    let direct = MatrixConversion {
        coefficients: Coefficients::Bt709,
        direction: Direction::ToRgb,
        input_samples: MatrixSamples::U8,
        output_samples: MatrixSamples::F32,
        input_range: SampleRange::Limited,
        output_range: SampleRange::Full,
    };
    let in_float = MatrixConversion {
        input_samples: MatrixSamples::F32,
        input_range: SampleRange::Full,
        ..direct
    };
    let code_planes = codes
        .each_ref()
        .map(|values| encode_samples(values, MatrixSamples::U8));
    let floats = as_float(&codes);
    let float_planes = floats
        .each_ref()
        .map(|values| encode_samples(values, MatrixSamples::F32));

    let found = converted(direct, code_planes, 8, 8);
    let expected = converted(in_float, float_planes, 8, 8);

    assert_within(&found, &expected, 1e-5, "eight-bit to float");
}
