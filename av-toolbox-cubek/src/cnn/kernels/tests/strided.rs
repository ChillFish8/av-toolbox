use super::{R9700, TestDevice};
use crate::cnn::kernels::strided::strided_shape;
use crate::cnn::{Stride, strided_variant};

#[test]
fn the_r9700_offers_every_dpir_down_and_up_layer() {
    let properties = R9700.properties();

    for channels in [64, 128, 256] {
        let down = strided_variant(&properties, Stride::Down, channels);
        assert!(down.is_some(), "down from {channels}");
    }

    for channels in [128, 256, 512] {
        let up = strided_variant(&properties, Stride::Up, channels);
        assert!(up.is_some(), "up from {channels}");
    }
}

#[test]
fn a_device_without_the_cooperative_matrix_is_offered_no_strided_layer() {
    let device = TestDevice {
        offers_matrix: false,
        ..R9700
    };
    let properties = device.properties();

    let down = strided_variant(&properties, Stride::Down, 64);

    assert!(down.is_none());
}

#[test]
fn a_strided_shape_past_the_devices_shared_memory_is_refused() {
    let device = TestDevice {
        shared_bytes: 16 * 1024,
        ..R9700
    };
    let properties = device.properties();

    let down = strided_variant(&properties, Stride::Down, 256);

    assert!(down.is_none());
}

#[test]
fn a_down_block_stages_twice_its_rows_and_columns() {
    let properties = R9700.properties();
    let variant = strided_variant(&properties, Stride::Down, 64).expect("offered");
    let shape = strided_shape(variant, Stride::Down, 64).expect("valid");

    let staged_samples = 2 * variant.pixel_tiles * 2 * 16;
    let staged_bytes = staged_samples as usize * (64 + 8) * 2;
    let scratch_bytes = (variant.planes_m * variant.planes_n) as usize * 256 * 4;

    assert_eq!(shape.shared_bytes(), staged_bytes + scratch_bytes);
}
