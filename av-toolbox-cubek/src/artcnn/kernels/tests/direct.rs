use crate::artcnn::{Activation, DirectVariant, WeightStaging};

#[test]
fn the_default_variant_splits_both_model_widths() {
    let variant = DirectVariant::default();

    for features in [64, 96] {
        let shape = variant.shape(features, Activation::Silu, false);

        assert!(shape.is_ok(), "{features}: {shape:?}");
    }
}

#[test]
fn a_block_width_off_the_tile_is_refused() {
    let variant = DirectVariant {
        columns: 24,
        ..DirectVariant::default()
    };

    let shape = variant.shape(64, Activation::Silu, false);

    assert_eq!(
        shape,
        Err("24 columns is not a positive multiple of 16".to_string())
    );
}

#[test]
fn channel_groups_that_do_not_split_the_features_into_tiles_are_refused() {
    for (features, groups) in [(64, 3), (96, 4), (64, 0)] {
        let variant = DirectVariant {
            channel_groups: groups,
            planes_n: 1,
            ..DirectVariant::default()
        };

        let shape = variant.shape(features, Activation::None, true);

        let expected = format!("{groups} groups do not split {features} features into 16-channel tiles");
        assert_eq!(shape, Err(expected));
    }
}

#[test]
fn planes_that_do_not_split_the_block_are_refused() {
    let pixel_split = DirectVariant {
        planes_m: 3,
        ..DirectVariant::default()
    };
    let channel_split = DirectVariant {
        planes_n: 4,
        ..DirectVariant::default()
    };

    let pixel_shape = pixel_split.shape(64, Activation::Silu, false);
    let channel_shape = channel_split.shape(96, Activation::Silu, false);

    let pixel_error = "3 planes do not split the block's 4 pixel tiles".to_string();
    let channel_error = "4 planes do not split the block's 6 channel tiles".to_string();
    assert_eq!(pixel_shape, Err(pixel_error));
    assert_eq!(channel_shape, Err(channel_error));
}

#[test]
fn each_staged_weight_buffer_takes_one_tap_of_shared_memory() {
    let staging = [
        WeightStaging::Global,
        WeightStaging::Single,
        WeightStaging::Double,
    ];
    // 64 output channels of 64 input channels and 8 halves of padding, in f16.
    let tap_bytes = 64 * 72 * 2;

    let bytes: Vec<usize> = staging
        .into_iter()
        .map(|weights| {
            let variant = DirectVariant {
                weights,
                ..DirectVariant::default()
            };
            let shape = variant.shape(64, Activation::Silu, false);

            shape.expect("the variant is valid").shared_bytes()
        })
        .collect();

    // The global variant still declares one 16-byte load, since shared memory cannot be empty.
    assert_eq!(bytes[1] - bytes[0], tap_bytes - 16);
    assert_eq!(bytes[2] - bytes[1], tap_bytes);
}

#[test]
fn the_default_block_with_its_border_fits_beside_the_scratch_tiles() {
    let variant = DirectVariant::default();

    let shape = variant.shape(96, Activation::Silu, false);
    let bytes = shape.expect("the default is valid").shared_bytes();

    // A 6x18 bordered block of 104 halves, one 16-byte load, and two 16x16 f32 scratch tiles.
    assert_eq!(bytes, 6 * 18 * 104 * 2 + 16 + 2 * 256 * 4);
}

#[test]
fn a_block_of_no_rows_is_refused() {
    let variant = DirectVariant {
        rows: 0,
        ..DirectVariant::default()
    };

    let shape = variant.shape(64, Activation::Silu, false);

    assert_eq!(shape, Err("a block of 0 rows is empty".to_string()));
}
