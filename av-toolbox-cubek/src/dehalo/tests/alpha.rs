use crate::dehalo::alpha::{AlphaParams, alpha_sizes, m4};
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
    assert_eq!(m4(10.0), 16);
    assert_eq!(m4(16.0), 16);
    assert_eq!(m4(33.0), 32);
    assert_eq!(m4(34.0), 36);
}

#[test]
fn the_sizes_follow_the_radii_and_supersampling() {
    let params = defaults(8);
    let unsampled = AlphaParams { ss: 1.0, ..params };

    let sizes = alpha_sizes(64, 48, &params);
    let plain = alpha_sizes(64, 48, &unsampled);

    assert_eq!(sizes.small, (32, 24));
    assert_eq!(sizes.supersampled, Some((96, 72)));
    assert_eq!(plain.supersampled, None);
}

#[test]
fn thresholds_scale_with_the_depth() {
    let ten_bit = defaults(10);

    assert_eq!(ten_bit.peak(), 1023.0);
    assert_eq!(ten_bit.scale(50.0), 200.0);
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

/// Rows of a step from 40 to 200 at column 32, with a dark ring before it and a bright ring after.
pub(super) fn haloed_step(width: u32, height: u32) -> Vec<f32> {
    let row: Vec<f32> = (0..width)
        .map(|column| match column {
            0..28 => 40.0,
            28..32 => 10.0,
            32..36 => 240.0,
            _ => 200.0,
        })
        .collect();

    row.repeat(height as usize)
}

#[test]
fn a_haloed_step_loses_some_of_its_ringing() {
    let input = haloed_step(64, 16);
    let params = defaults(8);

    let dehaloed = reference_dehalo_alpha(&input, 64, 16, &params);

    let brightest = dehaloed.iter().copied().fold(f32::MIN, f32::max);
    let darkest = dehaloed.iter().copied().fold(f32::MAX, f32::min);
    assert!(brightest < 240.0, "the bright ring stayed at {brightest}");
    assert!(darkest > 10.0, "the dark ring stayed at {darkest}");
}
