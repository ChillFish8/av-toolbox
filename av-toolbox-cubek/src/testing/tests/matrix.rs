use crate::matrix::{Coefficients, Direction, MatrixConversion, MatrixSamples, SampleRange};
use crate::testing::{MATRIX_FIXTURES, decode_samples, reference_matrix};

fn to_rgb(samples: MatrixSamples, range: SampleRange) -> MatrixConversion {
    MatrixConversion {
        coefficients: Coefficients::Bt709,
        direction: Direction::ToRgb,
        input_samples: samples,
        output_samples: samples,
        input_range: range,
        output_range: SampleRange::Full,
    }
}

fn split(values: &[f64]) -> [&[f64]; 3] {
    let plane = values.len() / 3;

    [&values[..plane], &values[plane..2 * plane], &values[2 * plane..]]
}

#[test]
fn the_reference_matches_every_vapoursynth_fixture() {
    for fixture in MATRIX_FIXTURES {
        let conversion = fixture.conversion();
        let input = decode_samples(fixture.input, conversion.input_samples);
        let expected = decode_samples(fixture.output, conversion.output_samples);
        let planes = split(&input);

        let found = reference_matrix(&conversion, planes);

        let tolerance = match conversion.output_samples {
            MatrixSamples::F32 => 1e-5,
            _ => 1.0,
        };
        let found_values: Vec<f64> = found.concat();
        for (index, (value, wanted)) in found_values.iter().zip(&expected).enumerate() {
            let difference = (value - wanted).abs();

            assert!(
                difference <= tolerance,
                "{}: sample {index} is {value}, VapourSynth wrote {wanted}",
                fixture.name
            );
        }
    }
}

#[test]
fn out_of_gamut_integer_samples_clamp_instead_of_wrapping() {
    let conversion = to_rgb(MatrixSamples::U8, SampleRange::Limited);
    let luma = [235.0, 235.0, 16.0];
    let blue = [16.0, 240.0, 16.0];
    let red = [240.0, 16.0, 128.0];

    let [red_out, _, blue_out] = reference_matrix(&conversion, [&luma, &blue, &red]);

    assert_eq!(red_out, vec![255.0, 54.0, 0.0]);
    assert_eq!(blue_out, vec![18.0, 255.0, 0.0]);
}

#[test]
fn eight_bit_limited_black_and_white_become_float_black_and_white() {
    let conversion = MatrixConversion {
        output_samples: MatrixSamples::F32,
        ..to_rgb(MatrixSamples::U8, SampleRange::Limited)
    };
    let luma = [16.0, 235.0];
    let chroma = [128.0, 128.0];

    let planes = reference_matrix(&conversion, [&luma, &chroma, &chroma]);

    for plane in planes {
        assert_eq!(plane, vec![0.0, 1.0]);
    }
}

#[test]
fn limited_black_and_white_become_full_range_black_and_white() {
    let conversion = to_rgb(MatrixSamples::U8, SampleRange::Limited);
    let luma = [16.0, 235.0];
    let chroma = [128.0, 128.0];

    let planes = reference_matrix(&conversion, [&luma, &chroma, &chroma]);

    for plane in planes {
        assert_eq!(plane, vec![0.0, 255.0]);
    }
}
