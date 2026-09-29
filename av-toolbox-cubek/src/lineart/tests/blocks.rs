use crate::testing::{reference_merge, reference_vertical_121};

#[test]
fn merge_takes_the_weighted_mix() {
    let input = [0.0, 100.0, 200.0];
    let filtered = [100.0, 0.0, 200.0];
    let weight = [255.0, 0.0, 51.0];

    let merged = reference_merge(&input, &filtered, &weight, 255.0);

    assert_eq!(merged, vec![100.0, 100.0, 200.0]);
}

#[test]
fn a_full_weight_lands_on_a_fractional_filtered_value_far_from_the_input() {
    let input = [1000.0, 0.3];
    let filtered = [0.3, 1000.0];
    let weight = [255.0, 255.0];

    let merged = reference_merge(&input, &filtered, &weight, 255.0);

    assert_eq!(merged, vec![0.3, 1000.0]);
}

#[test]
fn vertical_121_blurs_down_the_column() {
    let column = [0.0, 8.0, 0.0];

    let blurred = reference_vertical_121(&column, 1, 3);

    assert_eq!(blurred, vec![4.0, 4.0, 4.0]);
}

#[test]
fn vertical_121_leaves_a_single_row_unchanged() {
    let row = [3.0, 9.0, 27.0];

    let blurred = reference_vertical_121(&row, 3, 1);

    assert_eq!(blurred, vec![3.0, 9.0, 27.0]);
}
