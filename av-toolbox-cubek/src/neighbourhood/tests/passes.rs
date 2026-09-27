use crate::neighbourhood::{Expansion, Extreme, Shape, Smooth, expansion_passes};
use crate::testing::{reference_extreme, reference_smooth};

#[test]
fn a_rectangle_is_square_passes_then_one_axis() {
    let wide = expansion_passes(2, 1, Expansion::Rectangle);
    let tall = expansion_passes(1, 3, Expansion::Rectangle);

    assert_eq!(wide, vec![Shape::Square, Shape::Horizontal]);
    assert_eq!(tall, vec![Shape::Square, Shape::Vertical, Shape::Vertical]);
}

#[test]
fn an_ellipse_alternates_by_the_steps_left() {
    let cases = [
        (1, vec![Shape::Square]),
        (2, vec![Shape::Plus, Shape::Square]),
        (3, vec![Shape::Plus, Shape::Plus, Shape::Square]),
        (4, vec![Shape::Square, Shape::Plus, Shape::Plus, Shape::Square]),
    ];

    for (radius, expected) in cases {
        let passes = expansion_passes(radius, radius, Expansion::Ellipse);

        assert_eq!(passes, expected, "radius {radius}");
    }
}

#[test]
fn no_steps_gives_no_passes() {
    let passes = expansion_passes(0, 0, Expansion::Ellipse);

    assert!(passes.is_empty());
}

/// A 5x5 plane of zeros with 9 in the centre.
fn impulse() -> Vec<f32> {
    let mut plane = vec![0.0; 25];
    plane[12] = 9.0;
    plane
}

#[test]
fn each_shape_spreads_the_maximum_to_its_own_neighbours() {
    let plane = impulse();

    let square = reference_extreme(&plane, 5, 5, Extreme::Maximum, Shape::Square);
    let plus = reference_extreme(&plane, 5, 5, Extreme::Maximum, Shape::Plus);
    let horizontal = reference_extreme(&plane, 5, 5, Extreme::Maximum, Shape::Horizontal);
    let vertical = reference_extreme(&plane, 5, 5, Extreme::Maximum, Shape::Vertical);

    assert_eq!(square.iter().filter(|value| **value == 9.0).count(), 9);
    assert_eq!(plus.iter().filter(|value| **value == 9.0).count(), 5);
    assert_eq!((horizontal[11], horizontal[7]), (9.0, 0.0));
    assert_eq!((vertical[7], vertical[11]), (9.0, 0.0));
}

#[test]
fn the_minimum_shrinks_a_bright_square() {
    let mut plane = vec![0.0; 25];
    for index in [6, 7, 8, 11, 12, 13, 16, 17, 18] {
        plane[index] = 5.0;
    }

    let shrunk = reference_extreme(&plane, 5, 5, Extreme::Minimum, Shape::Square);

    assert_eq!(shrunk[12], 5.0);
    assert_eq!(shrunk.iter().filter(|value| **value == 5.0).count(), 1);
}

#[test]
fn the_blurs_spread_an_impulse_by_their_weights() {
    let plane = impulse();

    let binomial = reference_smooth(&plane, 5, 5, Smooth::Binomial);
    let boxed = reference_smooth(&plane, 5, 5, Smooth::Box);

    assert_eq!(
        (binomial[12], binomial[7], binomial[6]),
        (9.0 * 4.0 / 16.0, 9.0 * 2.0 / 16.0, 9.0 / 16.0)
    );
    assert_eq!((boxed[12], boxed[6]), (1.0, 1.0));
}

#[test]
fn the_edges_read_through_reflection() {
    // Column 0 reads column 1 on its left, so the left edge of a ramp keeps its slope.
    let plane: Vec<f32> = (0..5).map(|column| column as f32).collect();

    let widest = reference_extreme(&plane, 5, 1, Extreme::Maximum, Shape::Horizontal);

    assert_eq!(widest, vec![1.0, 2.0, 3.0, 4.0, 4.0]);
}
