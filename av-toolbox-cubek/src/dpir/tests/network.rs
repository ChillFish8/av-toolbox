use crate::cnn::Precision;
use crate::dpir::{column_bytes, level_bytes};

#[test]
fn level_bytes_halve_the_sides_and_double_the_features() {
    let f32_bytes = level_bytes(Precision::F32, 24, 16);
    let f16_bytes = level_bytes(Precision::F16, 24, 16);

    assert_eq!(f32_bytes, [98304, 49152, 24576, 12288]);
    assert_eq!(f16_bytes, [49152, 24576, 12288, 6144]);
}

#[test]
fn column_bytes_hold_nine_first_level_maps() {
    let first_level = level_bytes(Precision::F16, 24, 16)[0];

    assert_eq!(column_bytes(Precision::F16, 24, 16), 9 * first_level);
}
