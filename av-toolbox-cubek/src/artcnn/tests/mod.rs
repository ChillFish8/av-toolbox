use half::f16;

use crate::artcnn::{
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
    ];

    for error in errors {
        let message = error.to_string();

        assert!(!message.is_empty() && !message.contains('\u{2014}'), "{message}");
    }
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
    let convolution = tile_side(64, 27, Precision::F16, BodyPath::Convolution, TILE_BUDGET);

    // At f32 on the im2col path, a sample holds 4 * 64 * 4 + 9 * 64 * 4 = 3328 bytes.
    // sqrt(2^30 / 3328) = 568, so the centre is 568 - 54 = 514.
    let columns = tile_side(64, 27, Precision::F32, BodyPath::Im2col, TILE_BUDGET);

    assert_eq!(convolution, 1394);
    assert_eq!(columns, 514);
}

#[test]
fn a_tiny_budget_still_gives_a_tile() {
    let side = tile_side(96, 51, Precision::F32, BodyPath::Im2col, 1024);

    assert_eq!(side, 1);
}
