mod body;

use cubecl::backtrace::BackTrace;
use cubecl::server::ServerError;
use half::f16;

use super::tiling::haloed_scratch;
use crate::artcnn::{
    ArtcnnError,
    BodyPath,
    Layer,
    Model,
    Precision,
    Region,
    TILE_BUDGET,
    Weights,
    WeightsError,
    layer_shapes,
    tile_side,
    tiles,
};

fn builtin_r8f64() -> &'static [u8] {
    let bytes = av_toolbox_models::builtin("artcnn", "ArtCNN_R8F64.bin");
    bytes.expect("R8F64 is built in")
}

fn tiny() -> Weights {
    let shapes = layer_shapes(2, 1);
    let mut layers = Vec::new();

    for (index, [out_channels, in_channels, _, _]) in shapes.into_iter().enumerate() {
        let weight_count = (out_channels * in_channels * 9) as usize;
        let weights = (0..weight_count)
            .map(|value| f16::from_f32(value as f32 / 64.0))
            .collect();
        let bias = (0..out_channels)
            .map(|value| f16::from_f32(index as f32 - value as f32))
            .collect();

        layers.push(Layer {
            out_channels,
            in_channels,
            weights,
            bias,
        });
    }

    Weights {
        features: 2,
        blocks: 1,
        layers,
    }
}

#[test]
fn the_models_have_their_widths_depths_and_files() {
    assert_eq!((Model::R8F64.features(), Model::R8F64.blocks()), (64, 8));
    assert_eq!((Model::R16F96.features(), Model::R16F96.blocks()), (96, 16));
    assert_eq!(Model::R8F64.file_name(), "ArtCNN_R8F64.bin");
    assert_eq!(Model::R16F96.file_name(), "ArtCNN_R16F96.bin");
}

#[test]
fn the_layer_shapes_follow_the_width_and_block_count() {
    let shapes = layer_shapes(64, 8);

    assert_eq!(shapes.len(), 27);
    assert_eq!(shapes[0], [64, 1, 3, 3]);
    assert!(shapes[1..26].iter().all(|shape| *shape == [64, 64, 3, 3]));
    assert_eq!(shapes[26], [4, 64, 3, 3]);
}

#[test]
fn the_builtin_r8f64_parses() {
    let bytes = builtin_r8f64();

    let weights = Weights::parse(bytes, 64, 8).expect("valid");

    let parameters: usize = weights
        .layers
        .iter()
        .map(|layer| layer.weights.len() + layer.bias.len())
        .sum();
    assert_eq!(weights.layers.len(), 27);
    assert_eq!(parameters, 926_148);
    assert_eq!(weights.halo(), 27);
}

#[test]
fn weights_round_trip_through_bytes() {
    let original = tiny();

    let bytes = original.to_bytes();
    let parsed = Weights::parse(&bytes, 2, 1).expect("valid");

    assert_eq!(parsed, original);
}

#[test]
fn the_first_weight_is_read_in_ohwi_order() {
    let bytes = builtin_r8f64();
    let weights = Weights::parse(bytes, 64, 8).expect("valid");
    let header = 8 + 4 + 16;
    let first = f16::from_le_bytes([bytes[header], bytes[header + 1]]);
    let second = f16::from_le_bytes([bytes[header + 2], bytes[header + 3]]);

    assert_eq!(weights.layers[0].weights[0], first);
    assert_eq!(weights.layers[0].weights[1], second);
}

#[test]
fn a_file_with_the_wrong_magic_is_refused() {
    let mut bytes = tiny().to_bytes();
    bytes[0] = b'X';

    let error = Weights::parse(&bytes, 2, 1).expect_err("bad magic");

    assert_eq!(error, WeightsError::BadMagic);
}

#[test]
fn a_truncated_file_is_refused() {
    let bytes = tiny().to_bytes();
    let cut = &bytes[..bytes.len() - 1];

    let error = Weights::parse(cut, 2, 1).expect_err("truncated");

    assert!(matches!(error, WeightsError::Truncated { .. }), "{error}");
}

#[test]
fn a_file_for_another_depth_is_refused() {
    let bytes = tiny().to_bytes();

    let error = Weights::parse(&bytes, 2, 2).expect_err("wrong depth");

    assert_eq!(
        error,
        WeightsError::LayerCount {
            found: 6,
            expected: 9
        }
    );
}

#[test]
fn a_file_for_another_width_is_refused() {
    let bytes = tiny().to_bytes();

    let error = Weights::parse(&bytes, 3, 1).expect_err("wrong width");

    assert_eq!(
        error,
        WeightsError::LayerShape {
            index: 0,
            found: [2, 1, 3, 3],
            expected: [3, 1, 3, 3]
        }
    );
}

#[test]
fn the_r8f64_file_is_refused_as_r16f96() {
    let bytes = builtin_r8f64();

    let error = Weights::parse(bytes, 96, 16).expect_err("wrong model");

    assert_eq!(
        error,
        WeightsError::LayerCount {
            found: 27,
            expected: 51
        }
    );
}

#[test]
fn trailing_bytes_are_refused() {
    let mut bytes = tiny().to_bytes();
    bytes.push(0);

    let error = Weights::parse(&bytes, 2, 1).expect_err("trailing");

    assert_eq!(error, WeightsError::TrailingBytes { count: 1 });
}

#[test]
fn every_refusal_reads_as_a_sentence() {
    let errors = [
        WeightsError::BadMagic,
        WeightsError::Truncated {
            needed: 10,
            available: 4,
        },
        WeightsError::LayerCount {
            found: 6,
            expected: 9,
        },
        WeightsError::LayerShape {
            index: 0,
            found: [2, 1, 3, 3],
            expected: [3, 1, 3, 3],
        },
        WeightsError::TrailingBytes { count: 1 },
        WeightsError::LayerLength {
            index: 2,
            found: [35, 2],
            expected: [36, 2],
        },
    ];

    for error in errors {
        let message = error.to_string();

        assert!(!message.is_empty() && !message.contains('\u{2014}'), "{message}");
    }
}

#[test]
fn weights_of_the_model_pass_validation() {
    let weights = tiny();

    let checked = weights.validate();

    assert_eq!(checked, Ok(()));
}

#[test]
fn weights_missing_a_layer_are_refused() {
    let mut weights = tiny();
    weights.layers.pop();

    let error = weights.validate().expect_err("missing layer");

    assert_eq!(
        error,
        WeightsError::LayerCount {
            found: 5,
            expected: 6
        }
    );
}

#[test]
fn a_layer_of_another_shape_is_refused() {
    let mut weights = tiny();
    weights.layers[1].in_channels = 3;

    let error = weights.validate().expect_err("wrong shape");

    assert_eq!(
        error,
        WeightsError::LayerShape {
            index: 1,
            found: [2, 3, 3, 3],
            expected: [2, 2, 3, 3]
        }
    );
}

#[test]
fn a_layer_missing_a_weight_is_refused() {
    let mut weights = tiny();
    weights.layers[2].weights.pop();

    let error = weights.validate().expect_err("short weights");

    // Layer 2 maps 2 channels to 2 over 9 taps, so it needs 2 * 2 * 9 = 36 weights and 2 bias values.
    assert_eq!(
        error,
        WeightsError::LayerLength {
            index: 2,
            found: [35, 2],
            expected: [36, 2]
        }
    );
}

#[test]
fn a_layer_missing_a_bias_value_is_refused() {
    let mut weights = tiny();
    weights.layers[5].bias.pop();

    let error = weights.validate().expect_err("short bias");

    // The last layer maps 2 channels to 4 over 9 taps, so it needs 4 * 2 * 9 = 72 weights and 4 bias values.
    assert_eq!(
        error,
        WeightsError::LayerLength {
            index: 5,
            found: [72, 3],
            expected: [72, 4]
        }
    );
}

#[test]
fn a_layer_whose_weight_count_overflows_u32_is_refused_not_wrapped() {
    let features = 21_846;
    let shapes = layer_shapes(features, 1);
    let mut layers = Vec::new();

    for [out_channels, in_channels, _, _] in shapes {
        layers.push(Layer {
            out_channels,
            in_channels,
            weights: Vec::new(),
            bias: Vec::new(),
        });
    }

    layers[0].weights = vec![f16::ZERO; features as usize * 9];
    layers[0].bias = vec![f16::ZERO; features as usize];

    // 21,846 * 21,846 * 9 = 4,295,229,444 weights, which wraps in u32 to 4,295,229,444 - 2^32 = 262,148.
    layers[1].weights = vec![f16::ZERO; 262_148];
    layers[1].bias = vec![f16::ZERO; features as usize];

    let weights = Weights {
        features,
        blocks: 1,
        layers,
    };

    let error = weights.validate().expect_err("wrapped weight count");

    assert_eq!(
        error,
        WeightsError::LayerLength {
            index: 1,
            found: [262_148, 21_846],
            expected: [4_295_229_444, 21_846]
        }
    );
}

#[test]
fn tiles_cover_the_plane_once_with_clipped_halos() {
    let width = 23;
    let height = 17;

    let list = tiles(width, height, 8, 3);

    let mut covered = vec![0u32; (width * height) as usize];
    for (region, core) in &list {
        assert!(region.x <= core.x && region.y <= core.y);
        assert!(region.x + region.width >= core.x + core.width);
        assert!(region.x + region.width <= width && region.y + region.height <= height);

        let left_halo = core.x - region.x;
        let expected_left = core.x.min(3);
        assert_eq!(left_halo, expected_left);

        for row in core.y..core.y + core.height {
            for column in core.x..core.x + core.width {
                covered[(row * width + column) as usize] += 1;
            }
        }
    }

    assert!(covered.iter().all(|count| *count == 1));
    assert_eq!(list.len(), 9);
}

#[test]
fn a_plane_smaller_than_a_tile_is_one_tile_with_no_halo() {
    let list = tiles(5, 4, 64, 27);

    let whole = Region {
        x: 0,
        y: 0,
        width: 5,
        height: 4,
    };
    assert_eq!(list, [(whole, whole)]);
}

#[test]
fn the_tile_side_fits_the_budget() {
    // R8F64 at f16 on the convolution path holds 4 maps of 64 f16 values per sample, 512 bytes.
    // sqrt(2^30 / 512) = 1448 samples a side haloed, so the centre is 1448 - 2 * 27 = 1394.
    let convolution = tile_side(
        64,
        27,
        Precision::F16,
        BodyPath::Convolution,
        TILE_BUDGET,
        usize::MAX,
    );

    // At f32 on the im2col path, a sample holds 4 * 64 * 4 + 9 * 64 * 4 = 3328 bytes.
    // sqrt(2^30 / 3328) = 568, so the centre is 568 - 54 = 514.
    let columns = tile_side(64, 27, Precision::F32, BodyPath::Im2col, TILE_BUDGET, usize::MAX);

    assert_eq!(convolution, 1394);
    assert_eq!(columns, 514);
}

#[test]
fn a_tiny_budget_still_gives_a_tile() {
    let side = tile_side(96, 51, Precision::F32, BodyPath::Im2col, 1024, usize::MAX);

    assert_eq!(side, 1);
}

#[test]
fn a_small_allocation_limit_shrinks_the_tile() {
    // At f32 on the im2col path, the columns are the largest buffer, 9 * 64 * 4 = 2304 bytes a sample.
    // 2^28 / 2304 = 116508 samples, and sqrt(116508) = 341 a side haloed, so the centre is 341 - 54 = 287.
    let side = tile_side(64, 27, Precision::F32, BodyPath::Im2col, TILE_BUDGET, 1 << 28);

    assert_eq!(side, 287);
}

#[test]
fn a_large_allocation_limit_leaves_the_tile_to_the_budget() {
    // 2^30 / 2304 = 466033 samples, and sqrt(466033) = 682 a side haloed, more than the budget's 568.
    let side = tile_side(64, 27, Precision::F32, BodyPath::Im2col, TILE_BUDGET, 1 << 30);

    assert_eq!(side, 514);
}

#[test]
fn the_allocation_limit_bounds_one_map_on_the_convolution_path() {
    // At f16 on the convolution path, the largest buffer is one map, 64 * 2 = 128 bytes a sample.
    // 2^20 / 128 = 8192 samples, and sqrt(8192) = 90 a side haloed, so the centre is 90 - 54 = 36.
    let side = tile_side(
        64,
        27,
        Precision::F16,
        BodyPath::Convolution,
        TILE_BUDGET,
        1 << 20,
    );

    assert_eq!(side, 36);
}

#[test]
fn a_frame_smaller_than_a_tile_needs_only_its_own_scratch() {
    let sizes = haloed_scratch(64, 27, Precision::F32, BodyPath::Im2col, 514, 100, 40);

    // A 100x40 frame of 64 f32 maps is 100 * 40 * 64 * 4 = 1,024,000 bytes, and the columns nine times that.
    assert_eq!(sizes.map_bytes, 1_024_000);
    assert_eq!(sizes.column_bytes, 9_216_000);
}

#[test]
fn a_frame_larger_than_a_tile_needs_the_haloed_tile() {
    let sizes = haloed_scratch(64, 27, Precision::F16, BodyPath::Convolution, 10, 200, 150);

    // A tile of 10 with a halo of 27 is 64 a side, so 64 * 64 * 64 * 2 = 524,288 bytes a map.
    assert_eq!(sizes.map_bytes, 524_288);
    assert_eq!(sizes.column_bytes, 0);
}

#[test]
fn a_frame_clips_only_the_side_it_is_short_on() {
    let sizes = haloed_scratch(8, 3, Precision::F32, BodyPath::Convolution, 10, 12, 100);

    // The haloed tile is 16 a side, clipped to the 12 wide frame, so 12 * 16 * 8 * 4 = 6,144 bytes.
    assert_eq!(sizes.map_bytes, 6_144);
}

#[test]
#[should_panic(expected = "a tile must be at least one sample on a side")]
fn tiles_refuse_a_tile_of_zero() {
    tiles(10, 10, 0, 3);
}

#[test]
fn a_device_failure_becomes_a_device_error() {
    let failure = ServerError::Generic {
        reason: "lost".to_string(),
        backtrace: BackTrace::capture(),
    };

    let error = ArtcnnError::from(failure);

    assert!(matches!(error, ArtcnnError::Device(_)), "{error}");
}
