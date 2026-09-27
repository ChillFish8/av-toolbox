use crate::testing::{reference_median, reference_prewitt};

#[test]
fn a_median_removes_a_lone_spike() {
    let mut plane = vec![10.0; 49];
    plane[24] = 200.0;

    for radius in 1..=3 {
        let filtered = reference_median(&plane, 7, 7, radius);

        assert_eq!(filtered[24], 10.0, "radius {radius}");
    }
}

#[test]
fn a_median_picks_the_middle_of_its_window() {
    // A 3x3 plane of 1 to 9 has 5 as the median at its centre.
    let plane: Vec<f32> = (1..=9).map(|value| value as f32).collect();

    let filtered = reference_median(&plane, 3, 3, 1);

    assert_eq!(filtered[4], 5.0);
}

#[test]
fn a_median_keeps_a_plane_of_repeats() {
    let plane = vec![7.0; 25];

    let filtered = reference_median(&plane, 5, 5, 2);

    assert_eq!(filtered, plane);
}

#[test]
fn prewitt_takes_the_strongest_compass_response() {
    // A vertical step from 0 to 10 gives 30 on the column kernel beside it.
    let row = [0.0, 0.0, 10.0, 10.0];
    let plane = row.repeat(3);

    let edges = reference_prewitt(&plane, 4, 3, 255.0);

    assert_eq!(edges[5], 30.0);
    assert_eq!(edges[6], 30.0);
}

#[test]
fn prewitt_clamps_at_the_peak() {
    let row = [0.0, 0.0, 255.0, 255.0];
    let plane = row.repeat(3);

    let edges = reference_prewitt(&plane, 4, 3, 255.0);

    assert_eq!(edges[5], 255.0);
}

#[test]
fn prewitt_finds_no_edge_in_a_flat_plane() {
    let plane = vec![90.0; 20];

    let edges = reference_prewitt(&plane, 5, 4, 255.0);

    assert_eq!(edges, vec![0.0; 20]);
}
