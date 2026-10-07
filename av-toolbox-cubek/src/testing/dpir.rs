use std::collections::HashMap;
use std::slice::Iter;
use std::sync::{Mutex, OnceLock};

use super::artcnn::{LCG_INCREMENT, LCG_MULTIPLIER, conv3x3_values, next_unit};
use super::bwdif::decode_values;
use crate::dpir::{LayerKind, Variant, WeightLayer, Weights, layer_plan};

type Cache = Mutex<HashMap<(u32, u32), Weights>>;

/// A region of a padded frame, `(x, y, width, height)`, and the padding before the plane's first column and
/// row, `(left, top)`.
type ReferencePlacement = (u32, u32, u32, u32, u32, u32);

/// The scale of the strength channel, the 8-bit code range.
const STRENGTH_SCALE: f64 = 255.0;

/// The largest magnitude of a 3x3 body weight.
const BODY_SCALE: f64 = 0.01;

/// The multiple of 8 the network pads each side to.
const ALIGNMENT: u32 = 8;

/// How many levels the network halves the frame through.
const HALVINGS: usize = 3;

/// How many ResBlocks run in a row at each level.
const BLOCKS: usize = 4;

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

/// Mirrors the whole DRUNet network on the CPU in f64 over `planes`, each `width` by `height`.
///
/// Each side is padded to a multiple of 8 by mirroring, with half the padding (rounded down) before the
/// plane. The output is cropped back to the planes, clamped to `0..=1` and scaled by `peak`, one plane a
/// channel.
pub fn reference_dpir(
    weights: &Weights,
    planes: &[Vec<f64>],
    width: u32,
    height: u32,
    peak: f64,
    strength: f64,
) -> Vec<Vec<f64>> {
    let (padded_width, left) = reference_padding(width);
    let (padded_height, top) = reference_padding(height);
    let placement = (0, 0, padded_width, padded_height, left, top);
    let mut layers = weights.layers().iter();
    let head_layer = next_layer(&mut layers);

    let mut state = reference_head(planes, width, height, peak, strength, head_layer, placement);
    let mut skips = Vec::with_capacity(HALVINGS + 1);
    let mut level_width = padded_width;
    let mut level_height = padded_height;

    for _ in 0..HALVINGS {
        skips.push(state.clone());

        let blocked = reference_blocks(&mut layers, &state, level_width, level_height);
        let down_layer = next_layer(&mut layers);
        state = reference_down(&blocked, level_width, level_height, down_layer);
        level_width /= 2;
        level_height /= 2;
    }

    skips.push(state.clone());
    state = reference_blocks(&mut layers, &state, level_width, level_height);

    for _ in 0..HALVINGS {
        let skip = skips.pop().expect("each level keeps its skip");
        add_values(&mut state, &skip);

        let up_layer = next_layer(&mut layers);
        let doubled = reference_up(&state, level_width, level_height, up_layer);
        level_width *= 2;
        level_height *= 2;
        state = reference_blocks(&mut layers, &doubled, level_width, level_height);
    }

    let first_skip = skips.pop().expect("the first level keeps its skip");
    add_values(&mut state, &first_skip);

    let tail_layer = next_layer(&mut layers);
    let output = reference_conv3x3(&state, padded_width, padded_height, tail_layer);
    let channels = tail_layer.out_channels as usize;
    let mut cropped = vec![Vec::with_capacity((width * height) as usize); channels];

    for row in 0..height {
        for column in 0..width {
            let padded_sample = (row + top) * padded_width + column + left;
            let start = padded_sample as usize * channels;

            for (channel, plane) in cropped.iter_mut().enumerate() {
                let value = output[start + channel];
                plane.push(value.clamp(0.0, 1.0) * peak);
            }
        }
    }

    cropped
}

/// The padded length of a side `length` long, and the padding before it.
fn reference_padding(length: u32) -> (u32, u32) {
    let padding = (ALIGNMENT - length % ALIGNMENT) % ALIGNMENT;

    (length + padding, padding / 2)
}

fn next_layer<'a>(layers: &mut Iter<'a, WeightLayer>) -> &'a WeightLayer {
    layers.next().expect("the weights hold every layer")
}

fn add_values(values: &mut [f64], addend: &[f64]) {
    for (value, added) in values.iter_mut().zip(addend) {
        *value += added;
    }
}

/// Runs a level's four ResBlocks from `input`, the next eight layers.
fn reference_blocks(layers: &mut Iter<'_, WeightLayer>, input: &[f64], width: u32, height: u32) -> Vec<f64> {
    let mut state = input.to_vec();

    for _ in 0..BLOCKS {
        let first_layer = next_layer(layers);
        let second_layer = next_layer(layers);
        let mut activated = reference_conv3x3(&state, width, height, first_layer);

        for value in &mut activated {
            *value = value.max(0.0);
        }

        let mut output = reference_conv3x3(&activated, width, height, second_layer);
        add_values(&mut output, &state);
        state = output;
    }

    state
}

/// One run of a DPIR model through ONNX Runtime. The input planes and the clamped output are raw little
/// endian f32 planes, one after another.
#[derive(Debug, Clone, Copy)]
pub struct GoldenRun {
    pub name: &'static str,
    pub variant: Variant,
    pub width: u32,
    pub height: u32,
    pub strength: f32,
    pub input: &'static [u8],
    pub expected: &'static [u8],
}

const GRAY_48X40: &[u8] = include_bytes!("fixtures/dpir/gray_48x40.input.f32");
const COLOUR_40X48: &[u8] = include_bytes!("fixtures/dpir/colour_40x48.input.f32");
const GRAY_50X38: &[u8] = include_bytes!("fixtures/dpir/gray_50x38.input.f32");

/// Every ONNX Runtime golden, at a peak of 1.0.
pub const GOLDEN_RUNS: [GoldenRun; 7] = [
    GoldenRun {
        name: "gray_48x40_s5",
        variant: Variant::Gray,
        width: 48,
        height: 40,
        strength: 5.0,
        input: GRAY_48X40,
        expected: include_bytes!("fixtures/dpir/gray_48x40_s5.f32"),
    },
    GoldenRun {
        name: "gray_48x40_s10",
        variant: Variant::Gray,
        width: 48,
        height: 40,
        strength: 10.0,
        input: GRAY_48X40,
        expected: include_bytes!("fixtures/dpir/gray_48x40_s10.f32"),
    },
    GoldenRun {
        name: "gray_48x40_s50",
        variant: Variant::Gray,
        width: 48,
        height: 40,
        strength: 50.0,
        input: GRAY_48X40,
        expected: include_bytes!("fixtures/dpir/gray_48x40_s50.f32"),
    },
    GoldenRun {
        name: "colour_40x48_s5",
        variant: Variant::Colour,
        width: 40,
        height: 48,
        strength: 5.0,
        input: COLOUR_40X48,
        expected: include_bytes!("fixtures/dpir/colour_40x48_s5.f32"),
    },
    GoldenRun {
        name: "colour_40x48_s10",
        variant: Variant::Colour,
        width: 40,
        height: 48,
        strength: 10.0,
        input: COLOUR_40X48,
        expected: include_bytes!("fixtures/dpir/colour_40x48_s10.f32"),
    },
    GoldenRun {
        name: "colour_40x48_s50",
        variant: Variant::Colour,
        width: 40,
        height: 48,
        strength: 50.0,
        input: COLOUR_40X48,
        expected: include_bytes!("fixtures/dpir/colour_40x48_s50.f32"),
    },
    GoldenRun {
        name: "gray_50x38_s10",
        variant: Variant::Gray,
        width: 50,
        height: 38,
        strength: 10.0,
        input: GRAY_50X38,
        expected: include_bytes!("fixtures/dpir/gray_50x38_s10.f32"),
    },
];

/// Splits raw little endian f32 `bytes` into `count` planes of `width` by `height`.
///
/// Panics if `bytes` does not hold exactly that many values.
pub fn planes_of(bytes: &[u8], width: u32, height: u32, count: usize) -> Vec<Vec<f32>> {
    let samples = width as usize * height as usize;
    let plane_bytes = samples * size_of::<f32>();
    assert_eq!(
        bytes.len(),
        count * plane_bytes,
        "the bytes do not hold {count} planes of {width}x{height}"
    );

    bytes.chunks_exact(plane_bytes).map(decode_values).collect()
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

    #[test]
    fn every_golden_holds_its_planes() {
        for run in &GOLDEN_RUNS {
            let channels = run.variant.image_channels() as usize;
            let inputs = planes_of(run.input, run.width, run.height, channels);
            let expected = planes_of(run.expected, run.width, run.height, channels);

            assert_eq!(inputs.len(), channels, "{}", run.name);
            assert_eq!(expected.len(), channels, "{}", run.name);
        }
    }
}
