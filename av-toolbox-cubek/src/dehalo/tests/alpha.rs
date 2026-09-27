use crate::dehalo::alpha::{AlphaParams, alpha_sizes, halo_mask_sample, m4};
use crate::testing::reference_dehalo_alpha;

pub(super) fn defaults(bits: u32) -> AlphaParams {
    AlphaParams {
        rx: 2.0,
        ry: 2.0,
        darkstr: 1.0,
        brightstr: 1.0,
        lowsens: 50.0,
        highsens: 50.0,
        ss: 1.5,
        bits,
    }
}

#[test]
fn m4_rounds_to_a_multiple_of_four_but_never_below_sixteen() {
    assert_eq!(m4(10.0), Some(16));
    assert_eq!(m4(16.0), Some(16));
    assert_eq!(m4(33.0), Some(32));
    assert_eq!(m4(34.0), Some(36));
}

#[test]
fn m4_refuses_non_finite_or_overflowing_values() {
    assert_eq!(m4(f32::INFINITY), None);
    assert_eq!(m4(1e12), None);
}

#[test]
fn the_sizes_follow_the_radii_and_supersampling() {
    let params = defaults(8);
    let unsampled = AlphaParams { ss: 1.0, ..params };

    let sizes = alpha_sizes(64, 48, &params).expect("valid sizes");
    let plain = alpha_sizes(64, 48, &unsampled).expect("valid sizes");

    assert_eq!(sizes.small, (32, 24));
    assert_eq!(sizes.supersampled, Some((96, 72)));
    assert_eq!(plain.supersampled, None);
}

#[test]
fn a_huge_supersampling_factor_is_refused() {
    let params = AlphaParams {
        ss: 1000.0,
        ..defaults(8)
    };

    assert_eq!(alpha_sizes(64, 48, &params), None);
}

#[test]
fn thresholds_scale_with_the_depth() {
    let ten_bit = defaults(10);

    assert_eq!(ten_bit.peak(), 1023.0);
    assert_eq!(ten_bit.scale(50.0), 200.0);
}

#[test]
fn halo_mask_sample_matches_a_hand_worked_value() {
    // At 8 bits: full 255, low 50, offset 256, span 512, high 0.5, peak 255.
    //
    // contrast = (100 - 0) / 100.000001 ~= 1
    // sensitivity = 1 * 255 - 50 = 205
    // weight = (100 + 256) / 512 + 0.5 = 1.1953125
    // mask = clamp(205 * 1.1953125, 0, 255) = 245.0390625
    // result = 150 + (200 - 150) * 245.0390625 / 255 = 198.046875
    let params = defaults(8);

    let mask = halo_mask_sample(200.0, 150.0, 100.0, 0.0, &params);

    assert!((mask - 198.046_88).abs() < 1e-3, "got {mask}");
}

#[test]
fn a_flat_plane_comes_back_flat() {
    let flat = vec![120.0f32; 64 * 32];

    for ss in [1.0, 1.5] {
        let params = AlphaParams { ss, ..defaults(8) };
        let dehaloed = reference_dehalo_alpha(&flat, 64, 32, &params);

        assert!(
            dehaloed.iter().all(|value| (value - 120.0).abs() < 1e-3),
            "ss {ss}"
        );
    }
}

/// Rows of a step from 40 to 200 at column 32, with a 2-wide dark ring before it and a 2-wide
/// bright ring after.
pub(super) fn haloed_step(width: u32, height: u32) -> Vec<f32> {
    let row: Vec<f32> = (0..width)
        .map(|column| match column {
            0..30 => 40.0,
            30..32 => 10.0,
            32..34 => 240.0,
            _ => 200.0,
        })
        .collect();

    row.repeat(height as usize)
}

#[test]
fn a_haloed_step_loses_at_least_half_its_ringing() {
    let input = haloed_step(64, 16);
    let params = defaults(8);

    let dehaloed = reference_dehalo_alpha(&input, 64, 16, &params);

    let brightest = dehaloed.iter().copied().fold(f32::MIN, f32::max);
    let darkest = dehaloed.iter().copied().fold(f32::MAX, f32::min);
    let undershoot = 40.0 - darkest;
    let overshoot = brightest - 200.0;
    assert!(undershoot <= 15.0, "the dark ring only shrank to {darkest}");
    assert!(overshoot <= 20.0, "the bright ring only shrank to {brightest}");
}
