use crate::matrix::{Coefficients, Direction, MatrixConversion, MatrixSamples, SampleRange, scaling};

pub const MATRIX_WIDTH: u32 = 16;

pub const MATRIX_HEIGHT: u32 = 8;

/// One conversion written by VapourSynth's resize.
///
/// `metadata` holds `<direction> <matrix> <range> <bits>`, bits 32 meaning float. `input` and `output` hold three
/// planes each, in channel order.
pub struct MatrixFixture {
    pub name: &'static str,
    pub metadata: &'static str,
    pub input: &'static [u8],
    pub output: &'static [u8],
}

macro_rules! fixture {
    ($name:literal) => {
        MatrixFixture {
            name: $name,
            metadata: include_str!(concat!("fixtures/matrix/", $name, ".txt")),
            input: include_bytes!(concat!("fixtures/matrix/", $name, ".in")),
            output: include_bytes!(concat!("fixtures/matrix/", $name, ".out")),
        }
    };
}

/// Every matrix fixture.
pub const MATRIX_FIXTURES: &[MatrixFixture] = &[
    fixture!("to_rgb_bt709_limited_8"),
    fixture!("to_rgb_bt709_full_8"),
    fixture!("to_rgb_bt709_limited_16"),
    fixture!("to_rgb_bt709_full_16"),
    fixture!("to_rgb_bt709_full_32"),
    fixture!("to_rgb_bt601_limited_8"),
    fixture!("to_rgb_bt601_full_8"),
    fixture!("to_rgb_bt601_limited_16"),
    fixture!("to_rgb_bt601_full_16"),
    fixture!("to_rgb_bt601_full_32"),
    fixture!("to_rgb_bt2020_ncl_limited_8"),
    fixture!("to_rgb_bt2020_ncl_full_8"),
    fixture!("to_rgb_bt2020_ncl_limited_16"),
    fixture!("to_rgb_bt2020_ncl_full_16"),
    fixture!("to_rgb_bt2020_ncl_full_32"),
    fixture!("to_yuv_bt709_limited_8"),
    fixture!("to_yuv_bt709_full_8"),
    fixture!("to_yuv_bt709_limited_16"),
    fixture!("to_yuv_bt709_full_16"),
    fixture!("to_yuv_bt709_full_32"),
    fixture!("to_yuv_bt601_limited_8"),
    fixture!("to_yuv_bt601_full_8"),
    fixture!("to_yuv_bt601_limited_16"),
    fixture!("to_yuv_bt601_full_16"),
    fixture!("to_yuv_bt601_full_32"),
    fixture!("to_yuv_bt2020_ncl_limited_8"),
    fixture!("to_yuv_bt2020_ncl_full_8"),
    fixture!("to_yuv_bt2020_ncl_limited_16"),
    fixture!("to_yuv_bt2020_ncl_full_16"),
    fixture!("to_yuv_bt2020_ncl_full_32"),
];

impl MatrixFixture {
    /// The conversion the fixture was written with.
    pub fn conversion(&self) -> MatrixConversion {
        let words: Vec<&str> = self.metadata.split_whitespace().collect();
        let [direction_word, matrix_word, range_word, bits_word] = words[..] else {
            panic!("{}: the metadata is not four words", self.name);
        };

        let direction = match direction_word {
            "to_rgb" => Direction::ToRgb,
            "to_yuv" => Direction::ToYuv,
            other => panic!("{}: unexpected direction {other}", self.name),
        };
        let coefficients = match matrix_word {
            "bt709" => Coefficients::Bt709,
            "bt601" => Coefficients::Bt601,
            "bt2020_ncl" => Coefficients::Bt2020Ncl,
            other => panic!("{}: unexpected matrix {other}", self.name),
        };
        let range = match range_word {
            "limited" => SampleRange::Limited,
            "full" => SampleRange::Full,
            other => panic!("{}: unexpected range {other}", self.name),
        };
        let samples = match bits_word {
            "8" => MatrixSamples::U8,
            "16" => MatrixSamples::U16 { bits: 16 },
            "32" => MatrixSamples::F32,
            other => panic!("{}: unexpected bit depth {other}", self.name),
        };
        let (input_range, output_range) = match direction {
            Direction::ToRgb => (range, SampleRange::Full),
            Direction::ToYuv => (SampleRange::Full, range),
        };

        MatrixConversion {
            coefficients,
            direction,
            input_samples: samples,
            output_samples: samples,
            input_range,
            output_range,
        }
    }
}

/// Reads little-endian samples of `samples`' type as f64.
pub fn decode_samples(bytes: &[u8], samples: MatrixSamples) -> Vec<f64> {
    match samples {
        MatrixSamples::U8 => bytes.iter().map(|byte| f64::from(*byte)).collect(),
        MatrixSamples::U16 { .. } => {
            let (pairs, _) = bytes.as_chunks::<2>();

            pairs
                .iter()
                .map(|pair| f64::from(u16::from_le_bytes(*pair)))
                .collect()
        },
        MatrixSamples::F32 => {
            let (quads, _) = bytes.as_chunks::<4>();

            quads
                .iter()
                .map(|quad| f64::from(f32::from_le_bytes(*quad)))
                .collect()
        },
    }
}

/// Writes `values` as little-endian samples of `samples`' type. Integer values must already be whole.
pub fn encode_samples(values: &[f64], samples: MatrixSamples) -> Vec<u8> {
    match samples {
        MatrixSamples::U8 => values.iter().map(|value| *value as u8).collect(),
        MatrixSamples::U16 { .. } => values
            .iter()
            .flat_map(|value| (*value as u16).to_le_bytes())
            .collect(),
        MatrixSamples::F32 => values
            .iter()
            .flat_map(|value| (*value as f32).to_le_bytes())
            .collect(),
    }
}

/// Converts on the CPU in f64, normalising each sample, applying the matrix, then scaling back, step by step.
///
/// Integer outputs are rounded half up and clamped to their depth. Float outputs are left as they come.
pub fn reference_matrix(conversion: &MatrixConversion, planes: [&[f64]; 3]) -> [Vec<f64>; 3] {
    let input_chroma = conversion.input_chroma();
    let output_chroma = conversion.output_chroma();
    let (red_weight, blue_weight) = conversion.coefficients.weights();
    let green_weight = 1.0 - red_weight - blue_weight;
    let largest = match conversion.output_samples {
        MatrixSamples::U8 => Some(255.0),
        MatrixSamples::U16 { bits } => Some(f64::from((1u32 << bits) - 1)),
        MatrixSamples::F32 => None,
    };

    let mut outputs = [Vec::new(), Vec::new(), Vec::new()];
    let sample_count = planes[0].len();
    let pixels = (0..sample_count).map(|index| planes.map(|plane| plane[index]));
    for pixel in pixels {
        let mut normalised = [0.0; 3];
        for channel in 0..3 {
            let (offset, scale) = scaling(
                conversion.input_samples,
                conversion.input_range,
                input_chroma[channel],
            );
            normalised[channel] = (pixel[channel] - offset) / scale;
        }

        let converted = match conversion.direction {
            Direction::ToRgb => {
                let [luma, blue_difference, red_difference] = normalised;
                let red = luma + 2.0 * (1.0 - red_weight) * red_difference;
                let blue = luma + 2.0 * (1.0 - blue_weight) * blue_difference;
                let green = (luma - red_weight * red - blue_weight * blue) / green_weight;

                [red, green, blue]
            },
            Direction::ToYuv => {
                let [red, green, blue] = normalised;
                let luma = red_weight * red + green_weight * green + blue_weight * blue;
                let blue_difference = (blue - luma) / (2.0 * (1.0 - blue_weight));
                let red_difference = (red - luma) / (2.0 * (1.0 - red_weight));

                [luma, blue_difference, red_difference]
            },
        };

        for channel in 0..3 {
            let (offset, scale) = scaling(
                conversion.output_samples,
                conversion.output_range,
                output_chroma[channel],
            );
            let code = converted[channel] * scale + offset;
            let finished = match largest {
                Some(peak) => (code + 0.5).floor().clamp(0.0, peak),
                None => code,
            };

            outputs[channel].push(finished);
        }
    }

    outputs
}
