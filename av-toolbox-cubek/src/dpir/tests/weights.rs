use crate::dpir::{LayerKind, Variant, Weights, WeightsError, layer_plan};
use crate::testing::dpir::synthetic_weights;

const HEADER_BYTES: usize = 16;
const LAYER_HEADER_BYTES: usize = 12;

/// The byte offset where the layer at `index` starts.
fn layer_offset(weights: &Weights, index: usize) -> usize {
    let preceding = &weights.layers()[..index];
    let sizes = preceding
        .iter()
        .map(|layer| LAYER_HEADER_BYTES + layer.values.len() * 4);

    HEADER_BYTES + sizes.sum::<usize>()
}

fn write_word(bytes: &mut [u8], offset: usize, word: u32) {
    bytes[offset..offset + 4].copy_from_slice(&word.to_le_bytes());
}

fn gray_bytes() -> (Weights, Vec<u8>) {
    let weights = synthetic_weights(Variant::Gray, 1);
    let bytes = weights.to_bytes();

    (weights, bytes)
}

#[test]
fn the_plan_has_64_layers_with_the_reference_shapes() {
    for variant in [Variant::Gray, Variant::Colour] {
        let image_channels = variant.image_channels();
        let plan = layer_plan(variant);

        assert_eq!(plan.len(), 64);
        assert_eq!(plan[0], (LayerKind::Conv, image_channels + 1, 64));
        assert_eq!(plan[1], (LayerKind::Conv, 64, 64));
        assert_eq!(plan[8], (LayerKind::Conv, 64, 64));
        assert_eq!(plan[9], (LayerKind::Down, 64, 128));
        assert_eq!(plan[18], (LayerKind::Down, 128, 256));
        assert_eq!(plan[27], (LayerKind::Down, 256, 512));
        assert_eq!(plan[28], (LayerKind::Conv, 512, 512));
        assert_eq!(plan[35], (LayerKind::Conv, 512, 512));
        assert_eq!(plan[36], (LayerKind::Up, 512, 256));
        assert_eq!(plan[37], (LayerKind::Conv, 256, 256));
        assert_eq!(plan[45], (LayerKind::Up, 256, 128));
        assert_eq!(plan[54], (LayerKind::Up, 128, 64));
        assert_eq!(plan[55], (LayerKind::Conv, 64, 64));
        assert_eq!(plan[62], (LayerKind::Conv, 64, 64));
        assert_eq!(plan[63], (LayerKind::Conv, 64, image_channels));
    }
}

#[test]
fn weights_round_trip_through_bytes() {
    for variant in [Variant::Gray, Variant::Colour] {
        let weights = synthetic_weights(variant, 1);
        let bytes = weights.to_bytes();
        let parsed = Weights::parse(&bytes, variant).unwrap();

        assert_eq!(parsed, weights);
    }
}

#[test]
fn parse_refuses_the_other_variant() {
    let bytes = synthetic_weights(Variant::Colour, 1).to_bytes();
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    assert_eq!(
        error,
        WeightsError::Variant {
            found: Variant::Colour,
            expected: Variant::Gray,
        }
    );
}

#[test]
fn parse_refuses_a_truncated_file() {
    let (_, bytes) = gray_bytes();
    let error = Weights::parse(&bytes[..bytes.len() - 4], Variant::Gray).unwrap_err();

    assert_eq!(error, WeightsError::Truncated { index: 63 });
}

#[test]
fn parse_refuses_trailing_bytes() {
    let (_, mut bytes) = gray_bytes();
    bytes.extend_from_slice(&[0; 3]);
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    assert_eq!(error, WeightsError::Trailing { extra: 3 });
}

#[test]
fn parse_refuses_a_bad_magic() {
    let (_, mut bytes) = gray_bytes();
    bytes[0] = b'X';
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    assert_eq!(error, WeightsError::Magic);
}

#[test]
fn parse_refuses_an_unknown_kind() {
    let (weights, mut bytes) = gray_bytes();
    let offset = layer_offset(&weights, 5);
    write_word(&mut bytes, offset, 7);
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    assert_eq!(error, WeightsError::UnknownKind { index: 5, kind: 7 });
}

#[test]
fn parse_refuses_a_layer_of_the_wrong_shape() {
    let (weights, mut bytes) = gray_bytes();
    let offset = layer_offset(&weights, 9);
    write_word(&mut bytes, offset + 8, 64);
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    let expected = WeightsError::LayerShape {
        index: 9,
        found: (LayerKind::Down, 64, 64),
        expected: (LayerKind::Down, 64, 128),
    };
    assert_eq!(error, expected);
}

#[test]
fn parse_refuses_a_value_that_is_not_finite() {
    let (weights, mut bytes) = gray_bytes();
    let offset = layer_offset(&weights, 2) + LAYER_HEADER_BYTES;
    write_word(&mut bytes, offset, f32::NAN.to_bits());
    let error = Weights::parse(&bytes, Variant::Gray).unwrap_err();

    assert_eq!(error, WeightsError::NotFinite { index: 2 });
}
