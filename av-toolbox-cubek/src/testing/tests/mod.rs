mod based;

use super::{reference_narrow, within_ulps};

#[test]
fn narrowing_rounds_half_up_and_clamps() {
    let values = [-3.0, 0.49, 0.5, 1.5, 254.5, 300.0];

    let narrowed = reference_narrow(&values, 255.0);

    assert_eq!(narrowed, vec![0, 0, 1, 2, 255, 255]);
}

#[test]
fn neighbouring_floats_are_one_ulp_apart() {
    let one = 1.0f32;
    let next = f32::from_bits(one.to_bits() + 1);
    let after_next = f32::from_bits(one.to_bits() + 2);
    let below_zero = f32::from_bits((-0.0f32).to_bits() + 1);

    assert!(within_ulps(next, one, 1));
    assert!(!within_ulps(after_next, one, 1));
    assert!(within_ulps(after_next, one, 2));
    assert!(within_ulps(below_zero, 0.0, 1));
    assert!(!within_ulps(-1.0, 1.0, 2));
}
