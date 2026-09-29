use half::f16;

use crate::artcnn::{Layer, Weights, layer_shapes};

/// onnxruntime's output for R8F64, with its weights rounded to f16.
///
/// It was made from [golden_input](crate::testing::golden_input).
pub const GOLDEN_R8F64: &[u8] = include_bytes!("fixtures/golden_r8f64.f32");

const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;

/// The 16x16 input the golden fixture was made from, `((7x + 13y) mod 16) / 15`.
pub fn golden_input() -> Vec<f32> {
    let mut input = Vec::new();

    for row in 0..16u32 {
        for column in 0..16u32 {
            let pattern = (column * 7 + row * 13) % 16;
            input.push(pattern as f32 / 15.0);
        }
    }

    input
}

/// The SiLU activation, `x / (1 + e^-x)`, in f64.
pub fn reference_silu(value: f64) -> f64 {
    value / (1.0 + (-value).exp())
}

/// Mirrors a 3x3 convolution with zero padding on the CPU in f64, NHWC in and out.
pub fn reference_conv3x3(input: &[f64], width: u32, height: u32, layer: &Layer) -> Vec<f64> {
    let in_channels = i64::from(layer.in_channels);
    let out_channels = i64::from(layer.out_channels);
    let columns = i64::from(width);
    let rows = i64::from(height);
    let mut output = Vec::with_capacity((rows * columns * out_channels) as usize);

    for row in 0..rows {
        for column in 0..columns {
            for out_channel in 0..out_channels {
                let mut sum = f64::from(layer.bias[out_channel as usize]);

                for tap_row in 0..3i64 {
                    for tap_column in 0..3i64 {
                        let source_row = row + tap_row - 1;
                        let source_column = column + tap_column - 1;
                        let inside = source_row >= 0
                            && source_column >= 0
                            && source_row < rows
                            && source_column < columns;

                        if !inside {
                            continue;
                        }

                        let source = (source_row * columns + source_column) * in_channels;
                        let tap = ((out_channel * 3 + tap_row) * 3 + tap_column) * in_channels;

                        for in_channel in 0..in_channels {
                            let weight = f64::from(layer.weights[(tap + in_channel) as usize]);
                            sum += weight * input[(source + in_channel) as usize];
                        }
                    }
                }

                output.push(sum);
            }
        }
    }

    output
}

/// Places channel `2i + j` of each sample at row `i` and column `j` of a 2x2 block, doubling the plane.
pub fn reference_depth_to_space(channels: &[f64], width: u32, height: u32) -> Vec<f64> {
    let doubled_width = (width * 2) as usize;
    let mut plane = vec![0.0; doubled_width * (height * 2) as usize];

    for row in 0..height as usize {
        for column in 0..width as usize {
            for channel in 0..4usize {
                let value = channels[(row * width as usize + column) * 4 + channel];
                let out_row = row * 2 + channel / 2;
                let out_column = column * 2 + channel % 2;
                plane[out_row * doubled_width + out_column] = value;
            }
        }
    }

    plane
}

/// Mirrors the ArtCNN network on the CPU in f64, taking `input` in `0..=1` and giving the doubled plane.
pub fn reference_artcnn(input: &[f32], width: u32, height: u32, weights: &Weights) -> Vec<f64> {
    let layers = &weights.layers;
    let widened: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let conv0 = reference_conv3x3(&widened, width, height, &layers[0]);
    let mut state = conv0.clone();
    let mut next = 1;

    for _ in 0..weights.blocks {
        let first = reference_conv3x3(&state, width, height, &layers[next]);
        let first_active: Vec<f64> = first.into_iter().map(reference_silu).collect();
        let second = reference_conv3x3(&first_active, width, height, &layers[next + 1]);
        let second_active: Vec<f64> = second.into_iter().map(reference_silu).collect();
        let third = reference_conv3x3(&second_active, width, height, &layers[next + 2]);

        state = third
            .iter()
            .zip(&state)
            .map(|(value, skip)| value + skip)
            .collect();
        next += 3;
    }

    let joined = reference_conv3x3(&state, width, height, &layers[next]);
    let global: Vec<f64> = joined
        .iter()
        .zip(&conv0)
        .map(|(value, skip)| value + skip)
        .collect();
    let channels = reference_conv3x3(&global, width, height, &layers[next + 1]);
    let plane = reference_depth_to_space(&channels, width, height);

    plane.into_iter().map(|value| value.clamp(0.0, 1.0)).collect()
}

/// Advances a 64-bit LCG and gives its top 53 bits as a value between 0 and 1.
fn next_unit(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);

    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Deterministic weights for a network `features` wide with `blocks` blocks.
///
/// They are scaled so activations stay near one.
pub fn synthetic_weights(features: u32, blocks: u32, seed: u64) -> Weights {
    let mut state = seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    let shapes = layer_shapes(features, blocks);
    let mut layers = Vec::with_capacity(shapes.len());

    for [out_channels, in_channels, _, _] in shapes {
        let fan_in = f64::from(in_channels * 9);
        let scale = (1.0 / fan_in).sqrt();
        let weight_count = (out_channels * in_channels * 9) as usize;
        let mut weights = Vec::with_capacity(weight_count);

        for _ in 0..weight_count {
            let unit = next_unit(&mut state);
            let value = (unit * 2.0 - 1.0) * scale;
            weights.push(f16::from_f64(value));
        }

        let mut bias = Vec::with_capacity(out_channels as usize);
        for _ in 0..out_channels {
            let unit = next_unit(&mut state);
            let value = (unit * 2.0 - 1.0) * 0.05;
            bias.push(f16::from_f64(value));
        }

        layers.push(Layer {
            out_channels,
            in_channels,
            weights,
            bias,
        });
    }

    Weights {
        features,
        blocks,
        layers,
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::*;
    use crate::artcnn::{Layer, Weights, layer_shapes};

    fn centre_only(out_channels: u32, in_channels: u32, centre: f32) -> Layer {
        let weight_count = (out_channels * in_channels * 9) as usize;
        let mut weights = vec![f16::ZERO; weight_count];

        for out_channel in 0..out_channels {
            for in_channel in 0..in_channels {
                let index = (((out_channel * 3 + 1) * 3 + 1) * in_channels + in_channel) as usize;
                weights[index] = f16::from_f32(centre);
            }
        }

        let bias = vec![f16::ZERO; out_channels as usize];
        Layer {
            out_channels,
            in_channels,
            weights,
            bias,
        }
    }

    #[test]
    fn a_one_sample_network_works_through_by_hand() {
        let mut layers: Vec<Layer> = layer_shapes(1, 1)
            .into_iter()
            .map(|[out_channels, in_channels, _, _]| centre_only(out_channels, in_channels, 1.0))
            .collect();
        let last = layers.last_mut().expect("a last layer");
        let centres = [0.25f32, 0.5, 0.75, 1.0];

        for (channel, centre) in centres.iter().enumerate() {
            let index = channel * 9 + 4;
            last.weights[index] = f16::from_f32(*centre);
        }

        let weights = Weights {
            features: 1,
            blocks: 1,
            layers,
        };

        let output = reference_artcnn(&[0.5], 1, 1, &weights);

        // Worked by hand with v = 0.5 and silu(x) = x / (1 + e^-x).
        // conv0 = 0.5, a = silu(0.5) = 0.311230, b = silu(a) = 0.179637,
        // block = b + 0.5 = 0.679637, global = block + conv0 = 1.179637,
        // channels = 0.25, 0.5, 0.75 and 1.0 times global, clamped to 1.
        let expected = [0.294909, 0.589819, 0.884728, 1.0];
        for (got, want) in output.iter().zip(expected) {
            assert!((got - want).abs() < 1e-5, "{output:?}");
        }
    }

    #[test]
    fn depth_to_space_places_channel_2i_plus_j_at_row_i_column_j() {
        let channels = [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];

        let plane = reference_depth_to_space(&channels, 2, 1);

        assert_eq!(plane, [0.0, 1.0, 10.0, 11.0, 2.0, 3.0, 12.0, 13.0]);
    }

    #[test]
    fn a_convolution_pads_with_zero() {
        let layer = Layer {
            out_channels: 1,
            in_channels: 1,
            weights: vec![f16::ONE; 9],
            bias: vec![f16::from_f32(0.5)],
        };
        let input = [1.0, 1.0, 1.0, 1.0];

        let output = reference_conv3x3(&input, 2, 2, &layer);

        // Every sample of a 2x2 plane of ones sees four ones inside the plane, plus the bias.
        assert_eq!(output, [4.5, 4.5, 4.5, 4.5]);
    }

    #[test]
    fn the_reference_matches_onnxruntime() {
        let bytes = av_toolbox_models::builtin("artcnn", "ArtCNN_R8F64.bin").expect("built in");
        let weights = Weights::parse(bytes, 64, 8).expect("valid");
        let input = golden_input();
        let (floats, _) = GOLDEN_R8F64.as_chunks::<4>();
        let golden: Vec<f32> = floats.iter().map(|chunk| f32::from_le_bytes(*chunk)).collect();

        let output = reference_artcnn(&input, 16, 16, &weights);

        let largest = output
            .iter()
            .zip(&golden)
            .map(|(got, want)| (got - f64::from(*want)).abs())
            .fold(0.0, f64::max);
        println!("largest difference from onnxruntime: {largest:e}");
        // Measured 6.03e-7, so the bound leaves over 16x headroom.
        assert!(largest <= 1e-5, "{largest:e}");
    }

    #[test]
    fn synthetic_weights_have_the_models_shapes_and_repeat() {
        let first = synthetic_weights(96, 1, 7);
        let again = synthetic_weights(96, 1, 7);
        let other = synthetic_weights(96, 1, 8);

        let shapes: Vec<[u32; 4]> = first
            .layers
            .iter()
            .map(|layer| [layer.out_channels, layer.in_channels, 3, 3])
            .collect();
        let expected = layer_shapes(96, 1);
        assert_eq!(shapes, expected);
        assert_eq!(first, again);
        assert_ne!(first, other);
    }

    #[test]
    fn the_golden_input_is_the_documented_pattern() {
        let input = golden_input();

        assert_eq!(input.len(), 256);
        assert_eq!(input[0], 0.0);
        assert_eq!(input[1], 7.0 / 15.0);
        assert_eq!(input[16], 13.0 / 15.0);
    }
}
