use super::{R9700, TestDevice};
use crate::cnn::kernels::direct::check_variant;
use crate::cnn::{Activation, DirectVariant, WeightStaging, direct_variants};

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

#[test]
fn direct_variants_drop_a_variant_beyond_the_devices_shared_memory() {
    let properties = R9700.properties();
    let staged = DirectVariant {
        rows: 8,
        planes_m: 4,
        weights: WeightStaging::Single,
        ..DirectVariant::default()
    };
    let shape = staged.shape(96, Activation::Silu, false);
    let staged_bytes = shape.expect("the variant is valid").shared_bytes();

    let at_64 = direct_variants(&properties, 64);
    let at_96 = direct_variants(&properties, 96);

    // A 10x18 block of 104 halves, a staged tap of 96 channels and eight scratch tiles.
    assert_eq!(staged_bytes, 10 * 18 * 104 * 2 + 96 * 104 * 2 + 8 * 256 * 4);
    assert!(staged_bytes > R9700.shared_bytes);
    assert!(at_64.contains(&staged), "{at_64:?}");
    assert!(!at_96.contains(&staged), "{at_96:?}");
}

#[test]
fn every_offered_variant_fits_a_small_shared_memory() {
    let device = TestDevice {
        shared_bytes: 32 * 1024,
        ..R9700
    };
    let properties = device.properties();

    let offered = direct_variants(&properties, 96);

    let default_variant = DirectVariant::default();
    assert!(offered.contains(&default_variant), "{offered:?}");
    for variant in &offered {
        let shape = variant.shape(96, Activation::Silu, false);
        let bytes = shape.expect("an offered variant is valid").shared_bytes();

        assert!(bytes <= device.shared_bytes, "{variant:?} takes {bytes} bytes");
    }
}

#[test]
fn direct_variants_offer_only_the_devices_widest_subgroup() {
    let wave32 = TestDevice {
        plane_size_min: 32,
        plane_size_max: 32,
        ..R9700
    };
    let wave32_properties = wave32.properties();
    let wave64_properties = R9700.properties();

    let at_32 = direct_variants(&wave32_properties, 64);
    let at_64 = direct_variants(&wave64_properties, 64);

    let wide_default = DirectVariant::default();
    let narrow_default = DirectVariant {
        plane_width: 32,
        ..DirectVariant::default()
    };
    assert!(at_32.contains(&narrow_default), "{at_32:?}");
    assert!(!at_32.contains(&wide_default), "{at_32:?}");
    assert!(at_64.contains(&wide_default), "{at_64:?}");
    assert!(!at_64.contains(&narrow_default), "{at_64:?}");
    assert!(at_32.iter().all(|variant| variant.plane_width == 32));
    assert!(at_64.iter().all(|variant| variant.plane_width == 64));
}

#[test]
fn a_plane_narrower_than_the_widest_subgroup_is_refused() {
    let properties = R9700.properties();
    let narrow = DirectVariant {
        plane_width: 32,
        ..DirectVariant::default()
    };

    let refused = check_variant(&properties, &narrow, 64);

    let expected = "a plane of 32 units does not hold whole subgroups of up to 64".to_string();
    assert_eq!(refused, Err(expected));
}

#[test]
fn a_plane_wider_than_the_widest_subgroup_still_runs() {
    let wave32 = TestDevice {
        plane_size_min: 32,
        plane_size_max: 32,
        ..R9700
    };
    let properties = wave32.properties();
    let wide = DirectVariant::default();

    let checked = check_variant(&properties, &wide, 64);

    assert_eq!(checked, Ok(()));
}

#[test]
fn direct_variants_drop_a_variant_with_more_units_than_the_device_holds() {
    let device = TestDevice { units: 128, ..R9700 };
    let properties = device.properties();

    let offered = direct_variants(&properties, 64);

    assert!(!offered.is_empty());
    assert!(
        offered.iter().all(|variant| variant.units() <= 128),
        "{offered:?}"
    );
    assert!(offered.iter().all(|variant| variant.planes() <= 2), "{offered:?}");
}

#[test]
fn a_device_without_the_cooperative_matrix_is_offered_no_variant() {
    let device = TestDevice {
        offers_matrix: false,
        ..R9700
    };
    let properties = device.properties();

    let offered = direct_variants(&properties, 64);

    assert!(offered.is_empty(), "{offered:?}");
}

#[test]
fn a_shape_without_a_bias_declares_the_same_shared_memory() {
    let variant = DirectVariant::default();

    let with_bias = variant
        .shape(64, Activation::Relu, false)
        .expect("the default splits 64");
    let without_bias = with_bias.without_bias();

    assert_ne!(with_bias, without_bias);
    assert_eq!(with_bias.shared_bytes(), without_bias.shared_bytes());
}

#[test]
fn the_r9700_offers_the_default_variant_at_128_features_and_one_row_blocks_at_512() {
    let properties = R9700.properties();
    let default_variant = DirectVariant::default();
    let one_row = DirectVariant {
        rows: 1,
        planes_n: 4,
        ..DirectVariant::default()
    };

    let at_128 = direct_variants(&properties, 128);
    let at_512 = direct_variants(&properties, 512);

    assert!(at_128.contains(&default_variant), "{at_128:?}");
    assert_eq!(at_512, vec![one_row]);
}
