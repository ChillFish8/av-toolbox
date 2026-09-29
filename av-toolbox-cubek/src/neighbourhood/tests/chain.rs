use crate::neighbourhood::{Extreme, Shape, Spread, Step};
use crate::testing::reference_chain;

/// A 5x5 plane of zeros with 9 in the centre.
fn impulse() -> Vec<f32> {
    let mut plane = vec![0.0; 25];
    plane[12] = 9.0;
    plane
}

#[test]
fn two_square_maximums_grow_an_impulse_by_two_samples() {
    // A 7x7 plane with 9 in the centre. Each maximum grows the 9 by one sample on every side, so
    // two of them leave a 5x5 block of 9 inside a border of 0.
    let mut plane = vec![0.0; 49];
    plane[24] = 9.0;
    let steps = [
        Step::Extreme(Extreme::Maximum, Shape::Square),
        Step::Extreme(Extreme::Maximum, Shape::Square),
    ];

    let chained = reference_chain(&plane, 7, 7, &steps);

    for (index, value) in chained.iter().enumerate() {
        let column = index % 7;
        let row = index / 7;
        let inside = (1..=5).contains(&column) && (1..=5).contains(&row);
        let expected = if inside { 9.0 } else { 0.0 };

        assert_eq!(*value, expected, "column {column}, row {row}");
    }
}

#[test]
fn the_steps_run_in_order() {
    // The plus spreads the 9 to the centre and its 4 arms. Deflating then lowers the centre to
    // its neighbours' mean of 4 * 9 / 8 = 4.5, and the arm above it to 3 * 9 / 8 = 3.375.
    let plane = impulse();
    let grow_then_deflate = [
        Step::Extreme(Extreme::Maximum, Shape::Plus),
        Step::Spread(Spread::Deflate),
    ];

    let chained = reference_chain(&plane, 5, 5, &grow_then_deflate);

    assert_eq!(chained[12], 4.5);
    assert_eq!(chained[7], 3.375);

    // Deflating first lowers the lone 9 to its neighbours' mean of 0, so nothing is left to grow.
    let deflate_then_grow = [
        Step::Spread(Spread::Deflate),
        Step::Extreme(Extreme::Maximum, Shape::Plus),
    ];

    let reversed = reference_chain(&plane, 5, 5, &deflate_then_grow);

    assert!(reversed.iter().all(|value| *value == 0.0), "{reversed:?}");
}
