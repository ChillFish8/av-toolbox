use crate::neighbourhood::{Edge, Spread};
use crate::testing::{reference_spread, reference_spread_with_edge};

#[test]
fn inflate_lifts_a_dip_to_its_neighbours_mean() {
    // 8 neighbours of 8 average to 8, which is above the centre's 0.
    let mut plane = vec![8.0; 9];
    plane[4] = 0.0;

    let spread = reference_spread(&plane, 3, 3, Spread::Inflate);

    assert_eq!(spread[4], 8.0);
}

#[test]
fn inflate_leaves_a_peak_alone() {
    // 8 neighbours of 0 average to 0, which is below the centre's 9.
    let mut plane = vec![0.0; 9];
    plane[4] = 9.0;

    let spread = reference_spread(&plane, 3, 3, Spread::Inflate);

    assert_eq!(spread[4], 9.0);
}

#[test]
fn deflate_lowers_a_peak_to_its_neighbours_mean() {
    // 8 neighbours of 1 average to 1, which is below the centre's 9.
    let mut plane = vec![1.0; 9];
    plane[4] = 9.0;

    let spread = reference_spread(&plane, 3, 3, Spread::Deflate);

    assert_eq!(spread[4], 1.0);
}

#[test]
fn deflate_leaves_a_dip_alone() {
    // 8 neighbours of 9 average to 9, which is above the centre's 1.
    let mut plane = vec![9.0; 9];
    plane[4] = 1.0;

    let spread = reference_spread(&plane, 3, 3, Spread::Deflate);

    assert_eq!(spread[4], 1.0);
}

/// A 3x3 plane whose 8 neighbours read, top-left to bottom-right skipping the centre:
/// 2, 4, 6, 9, 12, 15, 18, 3. Their sum is 69, for a mean of 69 / 8 = 8.625.
fn mixed_neighbourhood(centre: f32) -> Vec<f32> {
    vec![2.0, 4.0, 6.0, 9.0, centre, 12.0, 15.0, 18.0, 3.0]
}

#[test]
fn a_mixed_neighbourhood_moves_by_the_mean() {
    let plane = mixed_neighbourhood(5.0);

    let inflated = reference_spread(&plane, 3, 3, Spread::Inflate);
    let deflated = reference_spread(&plane, 3, 3, Spread::Deflate);

    // The centre's 5 sits below the mean of 8.625, so inflate lifts it to the mean.
    assert!((inflated[4] - 8.625).abs() < 1e-6, "got {}", inflated[4]);

    // The same 5 sits below the mean, so deflate keeps it, since taking the smaller of the two
    // keeps the centre rather than the larger mean.
    assert_eq!(deflated[4], 5.0);
}

/// A 3x3 plane of 1 to 9, row by row.
fn counted_plane() -> Vec<f32> {
    (1..=9).map(|value| value as f32).collect()
}

#[test]
fn clamp_repeats_the_edge_at_a_corner() {
    // The corner holds 1. Clamped its neighbours are 1, 1, 2, 1, 2, 4, 4, 5, a mean of 2.5.
    // Reflected they are 5, 4, 5, 2, 2, 5, 4, 5, a mean of 4.
    let plane = counted_plane();

    let clamped = reference_spread_with_edge(&plane, 3, 3, Spread::Inflate, Edge::Clamp);
    let reflected = reference_spread_with_edge(&plane, 3, 3, Spread::Inflate, Edge::Reflect);

    assert_eq!(clamped[0], 2.5);
    assert_eq!(reflected[0], 4.0);
    assert_eq!(reflected, reference_spread(&plane, 3, 3, Spread::Inflate));
}

#[test]
fn clamp_repeats_the_edge_on_a_side() {
    // The top middle holds 2. Clamped its neighbours are 1, 2, 3, 1, 3, 4, 5, 6, a mean of 3.125.
    // Reflected they are 4, 5, 6, 1, 3, 4, 5, 6, a mean of 4.25.
    let plane = counted_plane();

    let clamped = reference_spread_with_edge(&plane, 3, 3, Spread::Inflate, Edge::Clamp);
    let reflected = reference_spread_with_edge(&plane, 3, 3, Spread::Inflate, Edge::Reflect);

    assert_eq!(clamped[1], 3.125);
    assert_eq!(reflected[1], 4.25);
}
