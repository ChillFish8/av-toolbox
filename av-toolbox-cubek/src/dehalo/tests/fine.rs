use super::alpha::{defaults, haloed_step};
use crate::dehalo::fine::{FineParams, contrast_sample, outside_sample};
use crate::testing::reference_fine_dehalo;

fn fine(bits: u32) -> FineParams {
    FineParams {
        alpha: defaults(bits),
        thmi: 80.0,
        thma: 128.0,
        thlimi: 50.0,
        thlima: 100.0,
        contra: 0.0,
        excl: true,
        edgeproc: 0.0,
    }
}

#[test]
fn the_radii_round_half_away_from_zero() {
    let mut params = fine(8);
    params.alpha.rx = 1.4;
    params.alpha.ry = 2.5;

    assert_eq!(params.radii(), (1, 3));
}

#[test]
fn contrast_sample_takes_the_zero_branch_on_opposite_signs() {
    // detail = (10 - 5) * 2.49 * 1 = 12.45, removed = 90 - 100 = -10.
    // The two disagree in sign, so nothing is restored.
    let restored = contrast_sample(100.0, 10.0, 5.0, 90.0, 1.0);

    assert!((restored - 100.0).abs() < 1e-3, "got {restored}");
}

#[test]
fn contrast_sample_takes_the_detail_branch_when_it_is_smaller() {
    // detail = (10 - 5) * 2.49 * 1 = 12.45, removed = 130 - 100 = 30.
    // Both agree in sign and detail is the smaller magnitude, so it is restored.
    let restored = contrast_sample(100.0, 10.0, 5.0, 130.0, 1.0);

    assert!((restored - 112.45).abs() < 1e-3, "got {restored}");
}

#[test]
fn contrast_sample_takes_the_removed_branch_when_it_is_smaller() {
    // detail = (10 - 5) * 2.49 * 1 = 12.45, removed = 105 - 100 = 5.
    // Both agree in sign and removed is the smaller magnitude, so it is restored.
    let restored = contrast_sample(100.0, 10.0, 5.0, 105.0, 1.0);

    assert!((restored - 105.0).abs() < 1e-3, "got {restored}");
}

#[test]
fn outside_sample_matches_hand_worked_values_across_excl_and_edgeproc() {
    // inside = strong = 20, band = clamp((100 - 20) * 2, 0, 255) = 160, no edgeproc widening.
    let excluded_off = outside_sample(100.0, 20.0, 50.0, false, 0.0, 255.0);
    // inside = max(20, 50) = 50, band = clamp((100 - 50) * 2, 0, 255) = 100.
    let excluded_on = outside_sample(100.0, 20.0, 50.0, true, 0.0, 255.0);
    // Same band as excluded_off, then widened by strong * 0.5 * 0.66 = 6.6.
    let widened_off = outside_sample(100.0, 20.0, 50.0, false, 0.5, 255.0);
    // Same band as excluded_on, then widened by strong * 0.5 * 0.66 = 6.6.
    let widened_on = outside_sample(100.0, 20.0, 50.0, true, 0.5, 255.0);

    assert!((excluded_off - 160.0).abs() < 1e-3, "got {excluded_off}");
    assert!((excluded_on - 100.0).abs() < 1e-3, "got {excluded_on}");
    assert!((widened_off - 166.6).abs() < 1e-3, "got {widened_off}");
    assert!((widened_on - 106.6).abs() < 1e-3, "got {widened_on}");
}

#[test]
fn a_flat_plane_comes_back_flat() {
    let flat = vec![90.0f32; 64 * 32];

    for contra in [0.0, 1.0] {
        let params = FineParams { contra, ..fine(8) };
        let dehaloed = reference_fine_dehalo(&flat, 64, 32, &params);

        assert!(
            dehaloed.iter().all(|value| (value - 90.0).abs() < 1e-3),
            "contra {contra}"
        );
    }
}

#[test]
fn a_haloed_step_stays_within_its_input_range_and_changes() {
    let input = haloed_step(64, 16);
    let params = fine(8);

    let dehaloed = reference_fine_dehalo(&input, 64, 16, &params);

    let brightest = dehaloed.iter().copied().fold(f32::MIN, f32::max);
    let darkest = dehaloed.iter().copied().fold(f32::MAX, f32::min);
    let changed = dehaloed
        .iter()
        .zip(&input)
        .any(|(after, before)| (after - before).abs() > 0.5);
    assert!(
        brightest <= 240.0 && darkest >= 10.0,
        "left the range {darkest}..{brightest}"
    );
    assert!(changed, "nothing near the edge changed");
}

#[test]
fn a_haloed_step_loses_at_least_a_tenth_of_its_overshoot() {
    let input = haloed_step(64, 16);
    let params = fine(8);

    let dehaloed = reference_fine_dehalo(&input, 64, 16, &params);

    let brightest = dehaloed.iter().copied().fold(f32::MIN, f32::max);
    let darkest = dehaloed.iter().copied().fold(f32::MAX, f32::min);
    // The input's ring sits at 10 and 240, 30 below and 40 above the step's 40 and 200, for a
    // total overshoot of 70.
    let total_overshoot = (40.0 - darkest) + (brightest - 200.0);
    assert!(
        darkest >= 10.0 && brightest <= 240.0,
        "left the input range {darkest}..{brightest}"
    );
    assert!(
        total_overshoot <= 63.0,
        "the overshoot only shrank to {total_overshoot}, from 70"
    );
}
