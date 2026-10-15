use crate::neighbourhood::median::{
    MEDIAN_OF_9,
    MEDIAN_OF_9_AT,
    MEDIAN_OF_25,
    MEDIAN_OF_25_AT,
    MEDIAN_OF_49,
    MEDIAN_OF_49_AT,
};
use crate::testing::{SplitMix, reference_median, reference_prewitt};

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

/// Runs a table of exchanges over `values` and returns the value at `at`.
fn select(values: &mut [f32], pairs: &[(usize, usize)], at: usize) -> f32 {
    for &(first, second) in pairs {
        let low = values[first].min(values[second]);
        let high = values[first].max(values[second]);
        values[first] = low;
        values[second] = high;
    }

    values[at]
}

/// Fails unless the exchanges select the median of every input of 0s and 1s.
///
/// By the 0-1 principle, a network of exchanges that does so selects the median of any input.
fn assert_selects_every_0_1_median(pairs: &[(usize, usize)], count: usize, at: usize) {
    let mut values = vec![0.0f32; count];

    for input in 0u32..1 << count {
        for (slot, value) in values.iter_mut().enumerate() {
            *value = ((input >> slot) & 1) as f32;
        }

        let ones = input.count_ones() as usize;
        let expected = if ones > count / 2 { 1.0 } else { 0.0 };
        let selected = select(&mut values, pairs, at);

        assert_eq!(selected, expected, "input {input:#b}");
    }
}

#[test]
fn the_9_value_network_selects_the_median_of_every_0_1_input() {
    assert_selects_every_0_1_median(&MEDIAN_OF_9, 9, MEDIAN_OF_9_AT);
}

#[test]
fn the_25_value_network_selects_the_median_of_every_0_1_input() {
    assert_selects_every_0_1_median(&MEDIAN_OF_25, 25, MEDIAN_OF_25_AT);
}

#[test]
fn the_49_value_selection_matches_a_sort() {
    let mut random = SplitMix(0x4949);

    for trial in 0..20_000 {
        let values: Vec<f32> = (0..49).map(|_| random.below(6) as f32).collect();
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);

        let mut working = values.clone();
        let selected = select(&mut working, &MEDIAN_OF_49, MEDIAN_OF_49_AT);

        assert_eq!(selected, sorted[24], "trial {trial}: {values:?}");
    }
}
