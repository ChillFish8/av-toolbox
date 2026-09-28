use crate::neighbourhood::{Extreme, Shape};
use crate::testing::reference_extreme;

#[test]
fn the_corners_minimum_ignores_the_plus_neighbours() {
    let mut plane = vec![5.0; 9];
    for index in [1, 3, 5, 7] {
        plane[index] = 1.0;
    }

    let minimum = reference_extreme(&plane, 3, 3, Extreme::Minimum, Shape::Corners);

    assert_eq!(minimum[4], 5.0);
}

#[test]
fn the_corners_maximum_takes_a_diagonal() {
    let mut plane = vec![0.0; 9];
    plane[8] = 7.0;

    let maximum = reference_extreme(&plane, 3, 3, Extreme::Maximum, Shape::Corners);

    assert_eq!(maximum[4], 7.0);
}
