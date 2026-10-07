use half::f16;

use crate::cnn::Activation;
use crate::testing::reference_silu;

const LCG_MULTIPLIER: u64 = 6364136223846793005;
const LCG_INCREMENT: u64 = 1442695040888963407;

/// One `F -> F` 3x3 layer's weights, by output channel, row, column and input channel, and its bias.
///
/// Every value is exact in f16, so f16 and f32 maps convolve the same weights.
#[derive(Debug, Clone)]
pub struct SyntheticLayer {
    pub features: u32,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

/// Advances a 64-bit LCG and gives a value between -1 and 1, rounded to f16.
fn next_signed(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);

    let unit = (*state >> 11) as f64 / (1u64 << 53) as f64;
    let signed = unit * 2.0 - 1.0;

    f16::from_f64(signed).to_f32()
}

/// Deterministic weights for a layer `features` wide, scaled so its outputs stay near its inputs.
pub fn synthetic_layer(features: u32, with_bias: bool, seed: u64) -> SyntheticLayer {
    let mut state = seed.wrapping_mul(LCG_MULTIPLIER).wrapping_add(LCG_INCREMENT);
    let count = (features * 9 * features) as usize;
    let scale = (1.0 / f64::from(features * 9)).sqrt() as f32;
    let mut weights = Vec::with_capacity(count);

    for _ in 0..count {
        let value = next_signed(&mut state) * scale;
        let exact = f16::from_f32(value).to_f32();
        weights.push(exact);
    }

    let bias = with_bias.then(|| (0..features).map(|_| next_signed(&mut state) * 0.125).collect());

    SyntheticLayer {
        features,
        weights,
        bias,
    }
}

/// The ReLU activation, `max(x, 0)`, in f64.
pub fn reference_relu(value: f64) -> f64 {
    value.max(0.0)
}

/// Mirrors `layer` on the CPU in f64 over NHWC maps with zero padding: the convolution and bias, then
/// `activation`, then `residual` if given.
pub fn reference_layer(
    layer: &SyntheticLayer,
    input: &[f64],
    width: u32,
    height: u32,
    activation: Activation,
    residual: Option<&[f64]>,
) -> Vec<f64> {
    let features = layer.features as i64;
    let columns = i64::from(width);
    let rows = i64::from(height);
    let mut output = Vec::with_capacity((rows * columns * features) as usize);

    for row in 0..rows {
        for column in 0..columns {
            for out_channel in 0..features {
                let bias = layer
                    .bias
                    .as_ref()
                    .map_or(0.0, |values| values[out_channel as usize]);
                let mut sum = f64::from(bias);

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

                        let source = (source_row * columns + source_column) * features;
                        let tap = ((out_channel * 3 + tap_row) * 3 + tap_column) * features;

                        for in_channel in 0..features {
                            let weight = f64::from(layer.weights[(tap + in_channel) as usize]);
                            sum += weight * input[(source + in_channel) as usize];
                        }
                    }
                }

                output.push(sum);
            }
        }
    }

    let activated: Vec<f64> = match activation {
        Activation::None => output,
        Activation::Silu => output.into_iter().map(reference_silu).collect(),
        Activation::Relu => output.into_iter().map(reference_relu).collect(),
    };

    match residual {
        Some(added) => activated
            .iter()
            .zip(added)
            .map(|(value, skip)| value + skip)
            .collect(),
        None => activated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reference_relu_zeroes_negatives_and_keeps_the_rest() {
        assert_eq!(reference_relu(-2.5), 0.0);
        assert_eq!(reference_relu(0.0), 0.0);
        assert_eq!(reference_relu(1.25), 1.25);
    }

    #[test]
    fn the_reference_layer_without_a_bias_passes_the_centre_tap_through() {
        let features = 2;
        let mut weights = vec![0.0f32; (features * 9 * features) as usize];
        for channel in 0..features as usize {
            let centre = ((channel * 3 + 1) * 3 + 1) * features as usize + channel;
            weights[centre] = 1.0;
        }

        let layer = SyntheticLayer {
            features,
            weights,
            bias: None,
        };
        let input = [0.5, -0.75];

        let plain = reference_layer(&layer, &input, 1, 1, Activation::None, None);
        let rectified = reference_layer(&layer, &input, 1, 1, Activation::Relu, None);
        let added = reference_layer(&layer, &input, 1, 1, Activation::Relu, Some(&[1.0, 1.0]));

        assert_eq!(plain, vec![0.5, -0.75]);
        assert_eq!(rectified, vec![0.5, 0.0]);
        assert_eq!(added, vec![1.5, 1.0]);
    }

    #[test]
    fn synthetic_layers_hold_f16_exact_values_and_their_bias_only_when_asked() {
        let with_bias = synthetic_layer(16, true, 3);
        let without_bias = synthetic_layer(16, false, 3);

        let exact = with_bias
            .weights
            .iter()
            .all(|value| f16::from_f32(*value).to_f32() == *value);
        assert!(exact);
        assert_eq!(with_bias.weights.len(), 16 * 9 * 16);
        assert_eq!(with_bias.bias.as_ref().map(Vec::len), Some(16));
        assert_eq!(without_bias.bias, None);
    }
}
