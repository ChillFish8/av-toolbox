use super::*;

#[test]
fn the_weights_sum_to_one_and_are_symmetric() {
    let weights = gaussian_weights(1.0, 4);

    let total: f32 = weights.iter().sum();
    assert!((total - 1.0).abs() < 1e-6);
    assert_eq!(weights.len(), 9);
    assert!((weights[0] - weights[8]).abs() < 1e-9);
}

#[test]
fn the_radius_reaches_four_deviations() {
    assert_eq!(radius_for(1.0), 4);
    assert_eq!(radius_for(1.6), 7);
    assert_eq!(radius_for(0.1), 1);
}

#[test]
fn the_widest_sigma_reaches_the_widest_radius() {
    assert_eq!(radius_for(MAX_SIGMA), MAX_RADIUS);
}

#[test]
fn a_small_radius_is_tiled() {
    let radius = radius_for(1.0);

    assert_eq!(radius, 4);
    assert_eq!(tier_for(1.0, radius), Tier::Tiled);
}

#[test]
fn the_adaptive_pre_smooth_is_tiled() {
    assert_eq!(tier_for(0.8, 1), Tier::Tiled);
}

#[test]
fn the_first_radius_past_the_tile_is_staged() {
    let radius = TILED_MAX_RADIUS + 1;

    assert_eq!(tier_for(4.25, radius), Tier::Staged);
}

#[test]
fn a_medium_radius_is_staged() {
    assert_eq!(tier_for(6.0, 24), Tier::Staged);
}

#[test]
fn a_truncated_wide_blur_is_staged() {
    assert_eq!(tier_for(64.0, 193), Tier::Staged);
}

#[test]
fn a_radius_past_the_staged_tier_is_plain() {
    let radius = STAGED_MAX_RADIUS + 1;

    assert_eq!(tier_for(64.0, radius), Tier::Plain);
}
