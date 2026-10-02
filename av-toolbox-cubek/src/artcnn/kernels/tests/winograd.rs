use half::f16;

use super::R9700;
use crate::artcnn::kernels::tune::offered_candidates;
use crate::artcnn::kernels::winograd::{check_variant, transform_kernel, transform_weights};
use crate::artcnn::{
    Activation,
    Candidate,
    Layer,
    Model,
    WinogradVariant,
    winograd_allowed,
    winograd_variants,
};

/// A 3x3 kernel whose transform is exact in f16.
const KERNEL: [[f64; 3]; 3] = [[0.25, -1.5, 2.0], [0.75, 3.0, -0.5], [-2.25, 1.25, 0.125]];

/// `G g Gᵀ` of [KERNEL] with Lavin and Gray's `G`, worked out in f64 with numpy.
const TRANSFORMED: [[f64; 4]; 4] = [
    [0.25, 0.375, 1.875, 2.0],
    [-0.625, 0.78125, -0.59375, 0.8125],
    [-1.375, -0.84375, 0.78125, 1.3125],
    [-2.25, -0.4375, -1.6875, 0.125],
];

fn variant_of(tile_rows: u32, tile_columns: u32, planes_n: u32) -> WinogradVariant {
    WinogradVariant {
        tile_rows,
        tile_columns,
        planes_m: 1,
        planes_n,
        plane_width: 64,
    }
}

#[test]
fn the_kernel_transform_is_g_g_g_transposed_before_rounding() {
    let transformed = transform_kernel(&KERNEL);

    assert_eq!(transformed, TRANSFORMED);
}

#[test]
fn the_transformed_weights_lie_by_position_output_channel_and_input_channel() {
    let (out_channels, in_channels) = (2, 3);
    let mut weights = vec![f16::ZERO; out_channels * 9 * in_channels];

    for (tap_row, kernel_row) in KERNEL.iter().enumerate() {
        for (tap_column, weight) in kernel_row.iter().enumerate() {
            let tap = (3 + tap_row) * 3 + tap_column;
            weights[tap * in_channels + 2] = f16::from_f64(*weight);
        }
    }

    let layer = Layer {
        out_channels: out_channels as u32,
        in_channels: in_channels as u32,
        weights,
        bias: vec![f16::ZERO; out_channels],
    };

    let transformed = transform_weights(&layer);

    assert_eq!(transformed.len(), 16 * out_channels * in_channels);
    for (index, value) in transformed.iter().enumerate() {
        let position = index / (out_channels * in_channels);
        let out_channel = index / in_channels % out_channels;
        let in_channel = index % in_channels;
        let expected = match (out_channel, in_channel) {
            (1, 2) => TRANSFORMED[position / 4][position % 4],
            _ => 0.0,
        };

        assert_eq!(
            value.to_f64(),
            expected,
            "position {position}, {out_channel}, {in_channel}"
        );
    }
}

#[test]
fn the_transformed_weights_round_to_f16_once() {
    let tenth = f16::from_f64(0.1);
    let layer = Layer {
        out_channels: 1,
        in_channels: 1,
        weights: vec![tenth; 9],
        bias: vec![f16::ZERO],
    };

    let transformed = transform_weights(&layer);

    // The middle positions sum all nine taps at a quarter each, which f16 cannot hold exactly.
    let exact = 2.25 * tenth.to_f64();
    assert_eq!(transformed[5], f16::from_f64(exact));
    assert_ne!(f16::from_f64(exact).to_f64(), exact);
}

#[test]
fn winograd_is_allowed_for_r8f64_and_r16f96() {
    for model in [Model::R8F64, Model::R16F96] {
        assert!(winograd_allowed(model), "{model:?}");
    }
}

fn holds_winograd(candidates: &[Candidate]) -> bool {
    candidates
        .iter()
        .any(|candidate| matches!(candidate, Candidate::Winograd(_)))
}

#[test]
fn a_network_of_no_allow_listed_model_is_offered_no_winograd_variant() {
    let properties = R9700.properties();

    for features in [64, 96] {
        let candidates = offered_candidates(&properties, features, None);

        assert!(!holds_winograd(&candidates), "{candidates:?}");
    }
}

#[test]
fn allow_listed_models_are_offered_winograd_before_the_fused_cubek_convolution() {
    let properties = R9700.properties();

    for model in [Model::R8F64, Model::R16F96] {
        let candidates = offered_candidates(&properties, model.features(), Some(model));

        assert!(holds_winograd(&candidates), "{model:?}: {candidates:?}");
        assert_eq!(candidates.last(), Some(&Candidate::FusedCubek));
    }
}

#[test]
fn winograd_variants_offer_only_layouts_that_split_the_width() {
    let properties = R9700.properties();

    let at_64 = winograd_variants(&properties, 64);
    let at_96 = winograd_variants(&properties, 96);

    let four_planes = variant_of(1, 16, 4);
    let six_planes = variant_of(1, 16, 6);
    assert!(at_64.contains(&four_planes), "{at_64:?}");
    assert!(!at_64.contains(&six_planes), "{at_64:?}");
    assert!(at_96.contains(&six_planes), "{at_96:?}");
    assert!(!at_96.contains(&four_planes), "{at_96:?}");
}

#[test]
fn a_winograd_variant_beyond_the_devices_shared_memory_is_refused() {
    let properties = R9700.properties();
    let large_block = variant_of(8, 32, 6);

    let shape = large_block.shape(96, Activation::Silu, false);
    let bytes = shape.expect("the variant is valid").shared_bytes();
    let refused = check_variant(&properties, &large_block, 96);

    // 256 tiles of 104 halves, and six planes' scratch tiles for 16 groups of tiles each.
    assert_eq!(bytes, 256 * 104 * 2 + 6 * 16 * 256 * 4);
    assert_eq!(
        refused,
        Err(format!(
            "{bytes} bytes of shared memory exceed the device's 65536"
        ))
    );
}

#[test]
fn a_layout_stages_one_position_in_little_shared_memory() {
    let variant = variant_of(1, 16, 4);

    let shape = variant.shape(64, Activation::Silu, false);
    let bytes = shape.expect("the variant is valid").shared_bytes();

    // One position of 16 tiles of 72 halves, and four planes' scratch tiles.
    assert_eq!(bytes, 16 * 72 * 2 + 4 * 256 * 4);
}

#[test]
fn winograd_shapes_that_do_not_split_their_work_are_refused() {
    let off_the_tile = variant_of(1, 8, 2);
    let uneven_planes = variant_of(1, 16, 3);

    let cases = [
        (
            off_the_tile,
            "a block of 1 by 8 tiles is not a positive multiple of 16 tiles",
        ),
        (uneven_planes, "3 planes do not split the 4 channel tiles"),
    ];

    for (variant, message) in cases {
        let shape = variant.shape(64, Activation::Silu, false);

        assert_eq!(shape.err().as_deref(), Some(message), "{variant:?}");
    }
}
