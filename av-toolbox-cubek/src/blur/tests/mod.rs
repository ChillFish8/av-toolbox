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
