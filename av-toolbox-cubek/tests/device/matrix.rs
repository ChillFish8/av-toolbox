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
use cubecl::prelude::Runtime;

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

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let sizing = Sizing(build.frame_context());
        let affine = self.conversion.affine();
        let values = upload_affine(client, &affine);
        let plane_bytes = self.planes[0].len();

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

        convert(client, &sizing, planes, self.conversion.samples, &values);

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

    decode_samples(&bytes, conversion.samples)
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
        let expected = decode_samples(fixture.output, conversion.samples);

        let found = converted(conversion, planes, MATRIX_WIDTH, MATRIX_HEIGHT);

        let bound = tolerance(conversion.samples);
        assert_within(&found, &expected, bound, fixture.name);
    }
}

#[test]
fn the_kernel_clamps_out_of_gamut_samples_like_the_reference() {
    let conversion = MatrixConversion {
        coefficients: Coefficients::Bt709,
        direction: Direction::ToRgb,
        samples: MatrixSamples::U16 { bits: 10 },
        input_range: SampleRange::Limited,
        output_range: SampleRange::Full,
    };
    let luma = [940.0, 64.0, 0.0, 1023.0];
    let blue = [64.0, 960.0, 0.0, 1023.0];
    let red = [960.0, 64.0, 1023.0, 0.0];
    let samples = conversion.samples;
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
        samples,
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
