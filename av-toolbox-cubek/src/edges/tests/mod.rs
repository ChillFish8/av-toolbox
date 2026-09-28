use crate::edges::{EdgeLimits, Operator};
use crate::testing::{reference_add_capped, reference_binarize, reference_edges};

/// Limits that leave every magnitude alone.
fn open() -> EdgeLimits {
    EdgeLimits {
        multi: 1.0,
        low: 0.0,
        high: 1000.0,
        peak: 1000.0,
    }
}

#[test]
fn open_leaves_every_magnitude_alone() {
    let limits = EdgeLimits::open(1000.0);

    assert_eq!(limits, open());
}

#[test]
fn a_vertical_step_gives_each_operator_its_magnitude() {
    let plane = [0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0];
    let limits = open();

    let prewitt = reference_edges(&plane, 3, 3, Operator::Prewitt, limits);
    let sobel = reference_edges(&plane, 3, 3, Operator::Sobel, limits);
    let kirsch = reference_edges(&plane, 3, 3, Operator::Kirsch, limits);

    assert_eq!(prewitt[4], 27.0);
    assert_eq!(sobel[4], 36.0);
    assert_eq!(kirsch[4], 135.0);
}

#[test]
fn a_lone_corner_gives_each_operator_its_magnitude() {
    let mut plane = vec![0.0; 9];
    plane[0] = 9.0;
    let limits = open();

    let prewitt = reference_edges(&plane, 3, 3, Operator::Prewitt, limits);
    let sobel = reference_edges(&plane, 3, 3, Operator::Sobel, limits);
    let kirsch = reference_edges(&plane, 3, 3, Operator::Kirsch, limits);

    let expected = 162.0f32.sqrt();

    assert!((prewitt[4] - expected).abs() < 1e-4);
    assert!((sobel[4] - expected).abs() < 1e-4);
    assert_eq!(kirsch[4], 45.0);
}

#[test]
fn multi_scales_the_magnitude() {
    let plane = [0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0];
    let limits = EdgeLimits { multi: 2.0, ..open() };

    let prewitt = reference_edges(&plane, 3, 3, Operator::Prewitt, limits);

    assert_eq!(prewitt[4], 54.0);
}

#[test]
fn the_limits_cut_below_low_and_fill_above_high() {
    let plane = [0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0];
    let limits = EdgeLimits {
        multi: 1.0,
        low: 30.0,
        high: 100.0,
        peak: 255.0,
    };

    let prewitt = reference_edges(&plane, 3, 3, Operator::Prewitt, limits);
    let sobel = reference_edges(&plane, 3, 3, Operator::Sobel, limits);
    let kirsch = reference_edges(&plane, 3, 3, Operator::Kirsch, limits);

    assert_eq!(prewitt[4], 0.0);
    assert_eq!(sobel[4], 36.0);
    assert_eq!(kirsch[4], 255.0);
}

#[test]
fn equal_limits_keep_a_value_exactly_at_them() {
    let plane = [0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0];
    let limits = EdgeLimits {
        multi: 1.0,
        low: 27.0,
        high: 27.0,
        peak: 255.0,
    };

    let prewitt = reference_edges(&plane, 3, 3, Operator::Prewitt, limits);
    let sobel = reference_edges(&plane, 3, 3, Operator::Sobel, limits);
    let kirsch = reference_edges(&plane, 3, 3, Operator::Kirsch, limits);

    assert_eq!(prewitt[4], 27.0);
    assert_eq!(sobel[4], 255.0);
    assert_eq!(kirsch[4], 255.0);
}

#[test]
fn a_large_magnitude_clamps_to_peak() {
    let plane = [0.0, 0.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 9.0];
    let limits = EdgeLimits {
        multi: 1.0,
        low: 0.0,
        high: 1000.0,
        peak: 100.0,
    };

    let kirsch = reference_edges(&plane, 3, 3, Operator::Kirsch, limits);

    assert_eq!(kirsch[4], 100.0);
}

#[test]
fn the_border_reflects_onto_the_neighbour() {
    let plane = [0.0, 3.0, 9.0];
    let limits = open();

    let prewitt = reference_edges(&plane, 3, 1, Operator::Prewitt, limits);

    assert_eq!(prewitt, vec![0.0, 27.0, 0.0]);
}

#[test]
fn a_single_sample_plane_has_no_edges() {
    let plane = [7.0];
    let limits = open();

    for operator in [Operator::Prewitt, Operator::Sobel, Operator::Kirsch] {
        let edges = reference_edges(&plane, 1, 1, operator, limits);

        assert_eq!(edges, vec![0.0], "{operator:?}");
    }
}

#[test]
fn binarize_splits_at_the_threshold() {
    let input = [0.1, 0.5, 0.49999, 0.9];

    let binarized = reference_binarize(&input, 0.5, 1.0);

    assert_eq!(binarized, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn add_capped_sums_and_clamps_to_the_peak() {
    let first = [0.0, 100.0, 200.0, 250.0];
    let second = [0.0, 50.0, 100.0, 100.0];

    let summed = reference_add_capped(&first, &second, 255.0);

    assert_eq!(summed, vec![0.0, 150.0, 255.0, 255.0]);
}
