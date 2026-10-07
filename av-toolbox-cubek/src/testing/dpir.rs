use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::artcnn::{LCG_INCREMENT, LCG_MULTIPLIER, conv3x3_values, next_unit};
use crate::dpir::{LayerKind, Variant, WeightLayer, Weights, layer_plan};

type Cache = Mutex<HashMap<(u32, u32), Weights>>;

/// A region of a padded frame, `(x, y, width, height)`, and the padding before the plane's first column and
/// row, `(left, top)`.
type ReferencePlacement = (u32, u32, u32, u32, u32, u32);

/// The scale of the strength channel, the 8-bit code range.
const STRENGTH_SCALE: f64 = 255.0;

/// The largest magnitude of a 3x3 body weight.
const BODY_SCALE: f64 = 0.01;

/// Deterministic weights for the `variant` network.
///
/// They keep activations bounded and the output mostly between 0 and 1, so the clamp does not hide a
/// difference from a reference. The weights of each seed are made once and kept.
pub fn synthetic_weights(variant: Variant, seed: u32) -> Weights {
    static CACHE: OnceLock<Cache> = OnceLock::new();

    let cache = CACHE.get_or_init(Default::default);
    let key = (variant.image_channels(), seed);
    let mut cached = cache.lock().expect("the weights cache is not poisoned");
    let weights = cached.entry(key).or_insert_with(|| build_weights(variant, seed));

    weights.clone()
}

fn build_weights(variant: Variant, seed: u32) -> Weights {
    let plan = layer_plan(variant);
    let last = plan.len() - 1;
    let mut state = u64::from(seed)
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add(LCG_INCREMENT);
    let mut layers = Vec::with_capacity(plan.len());

    for (index, (kind, in_channels, out_channels)) in plan.into_iter().enumerate() {
        let is_end = index == 0 || index == last;
        let (low, high) = value_range(kind, is_end, in_channels);
        let count = kind.value_count(in_channels, out_channels);
        let values = (0..count)
            .map(|_| (low + next_unit(&mut state) * (high - low)) as f32)
            .collect();

        layers.push(WeightLayer {
            kind,
            in_channels,
            out_channels,
            values,
        });
    }

    Weights::new(variant, layers).expect("the synthetic layers follow the plan")
}

/// The range a layer's values are drawn from.
fn value_range(kind: LayerKind, is_end: bool, in_channels: u32) -> (f64, f64) {
    match kind {
        LayerKind::Conv if is_end => (0.0, 2.0 / f64::from(9 * in_channels)),
        LayerKind::Conv => (-BODY_SCALE, BODY_SCALE),
        LayerKind::Down => symmetric_range(4 * in_channels),
        LayerKind::Up => symmetric_range(in_channels),
    }
}

fn symmetric_range(rows: u32) -> (f64, f64) {
    let bound = 1.0 / f64::from(rows).sqrt();

    (-bound, bound)
}

/// Mirrors [mirror](crate::mirror) on the CPU.
pub fn reference_mirror(index: i64, length: i64) -> i64 {
    let period = 2 * length;
    let unfolded = if index < 0 { -index - 1 } else { index };
    let folded = unfolded % period;

    if folded >= length {
        period - 1 - folded
    } else {
        folded
    }
}

/// Mirrors DRUNet's head on the CPU in f64, giving NHWC maps of the placement's region.
///
/// The input is each of `planes` divided by `peak`, then a constant channel of `strength / 255`. The region's
/// samples map onto the planes past the placement's padding, mirroring with the edge repeated, and samples
/// outside the region are zero in every channel.
pub fn reference_head(
    planes: &[Vec<f64>],
    width: u32,
    height: u32,
    peak: f64,
    strength: f64,
    layer: &WeightLayer,
    placement: ReferencePlacement,
) -> Vec<f64> {
    let (region_x, region_y, region_width, region_height, left, top) = placement;
    let strength_value = strength / STRENGTH_SCALE;
    let mut input = Vec::new();

    for row in 0..region_height {
        let padded_row = i64::from(region_y + row) - i64::from(top);
        let plane_row = reference_mirror(padded_row, i64::from(height));

        for column in 0..region_width {
            let padded_column = i64::from(region_x + column) - i64::from(left);
            let plane_column = reference_mirror(padded_column, i64::from(width));
            let index = (plane_row * i64::from(width) + plane_column) as usize;

            for plane in planes {
                input.push(plane[index] / peak);
            }

            input.push(strength_value);
        }
    }

    reference_conv3x3(&input, region_width, region_height, layer)
}

/// Mirrors a bias-free 3x3 convolution with zero padding on the CPU in f64, NHWC in and out.
pub fn reference_conv3x3(input: &[f64], width: u32, height: u32, layer: &WeightLayer) -> Vec<f64> {
    let weight = |index: usize| f64::from(layer.values[index]);

    conv3x3_values(
        input,
        width,
        height,
        layer.in_channels,
        layer.out_channels,
        weight,
        None,
    )
}

/// Mirrors DRUNet's 2x2 stride 2 convolution on the CPU in f64, NHWC in and out.
pub fn reference_down(input: &[f64], width: u32, height: u32, layer: &WeightLayer) -> Vec<f64> {
    let in_channels = layer.in_channels as usize;
    let out_channels = layer.out_channels as usize;
    let columns = width as usize;
    let half_columns = columns / 2;
    let half_rows = height as usize / 2;
    let weights = onnx_down_weights(layer);
    let mut output = Vec::with_capacity(half_columns * half_rows * out_channels);

    for row in 0..half_rows {
        for column in 0..half_columns {
            for out_channel in 0..out_channels {
                let mut sum = 0.0;

                for in_channel in 0..in_channels {
                    for tap_row in 0..2 {
                        for tap_column in 0..2 {
                            let source_row = row * 2 + tap_row;
                            let source_column = column * 2 + tap_column;
                            let source = (source_row * columns + source_column) * in_channels + in_channel;
                            let tap =
                                ((out_channel * in_channels + in_channel) * 2 + tap_row) * 2 + tap_column;
                            sum += weights[tap] * input[source];
                        }
                    }
                }

                output.push(sum);
            }
        }
    }

    output
}

/// A down layer's matrix as ONNX holds the convolution's weight, by output channel, input channel, row and
/// column.
///
/// Row `(row * 2 + column) * in + input channel` of the `[4 * in, out]` matrix holds that tap.
fn onnx_down_weights(layer: &WeightLayer) -> Vec<f64> {
    let in_channels = layer.in_channels as usize;
    let out_channels = layer.out_channels as usize;
    let mut weights = Vec::with_capacity(out_channels * in_channels * 4);

    for out_channel in 0..out_channels {
        for in_channel in 0..in_channels {
            for tap in 0..4 {
                let matrix_row = tap * in_channels + in_channel;
                let value = layer.values[matrix_row * out_channels + out_channel];
                weights.push(f64::from(value));
            }
        }
    }

    weights
}

/// Mirrors DRUNet's 2x2 stride 2 transposed convolution on the CPU in f64, NHWC in and out.
pub fn reference_up(input: &[f64], width: u32, height: u32, layer: &WeightLayer) -> Vec<f64> {
    let in_channels = layer.in_channels as usize;
    let out_channels = layer.out_channels as usize;
    let columns = width as usize;
    let rows = height as usize;
    let doubled_columns = columns * 2;
    let weights = onnx_up_weights(layer);
    let mut output = vec![0.0; doubled_columns * rows * 2 * out_channels];

    for row in 0..rows {
        for column in 0..columns {
            for in_channel in 0..in_channels {
                let value = input[(row * columns + column) * in_channels + in_channel];

                for tap_row in 0..2 {
                    for tap_column in 0..2 {
                        let out_row = row * 2 + tap_row;
                        let out_column = column * 2 + tap_column;
                        let target = (out_row * doubled_columns + out_column) * out_channels;

                        for out_channel in 0..out_channels {
                            let tap =
                                ((in_channel * out_channels + out_channel) * 2 + tap_row) * 2 + tap_column;
                            output[target + out_channel] += value * weights[tap];
                        }
                    }
                }
            }
        }
    }

    output
}

/// An up layer's matrix as ONNX holds the transposed convolution's weight, by input channel, output channel,
/// row and column.
///
/// Column `(row * 2 + column) * out + output channel` of the `[in, 4 * out]` matrix holds that tap.
fn onnx_up_weights(layer: &WeightLayer) -> Vec<f64> {
    let in_channels = layer.in_channels as usize;
    let out_channels = layer.out_channels as usize;
    let matrix_columns = 4 * out_channels;
    let mut weights = Vec::with_capacity(in_channels * matrix_columns);

    for in_channel in 0..in_channels {
        for out_channel in 0..out_channels {
            for tap in 0..4 {
                let matrix_column = tap * out_channels + out_channel;
                let value = layer.values[in_channel * matrix_columns + matrix_column];
                weights.push(f64::from(value));
            }
        }
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_mirror_folds_with_the_edge_repeated() {
        assert_eq!(reference_mirror(-1, 5), 0);
        assert_eq!(reference_mirror(-3, 5), 2);
        assert_eq!(reference_mirror(5, 5), 4);
        assert_eq!(reference_mirror(-4, 3), 2);
        assert_eq!(reference_mirror(7, 1), 0);
    }
}
