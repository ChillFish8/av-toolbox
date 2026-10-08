use crate::deblock::h264::thresholds;
use crate::matrix::MatrixSamples;
use crate::testing::{
    H264_FIXTURES,
    H264_FLOAT_TOLERANCE,
    assert_h264_plane,
    blocky_plane,
    reference_h264,
    reference_h264_padded,
};

/// One row of 8 samples repeated down 4 rows, deblocked at `quant`, giving the first row.
///
/// The plane is two blocks across and one down, so only the left edge at column 4 is filtered.
fn filtered_row(row: [f64; 8], quant: i32, samples: MatrixSamples) -> Vec<f64> {
    let mut plane = row.repeat(4);
    let found_thresholds = thresholds(quant, 0, 0, samples);

    reference_h264_padded(&mut plane, 8, 4, samples, &found_thresholds);

    let first = plane[..8].to_vec();
    let every_row_alike = plane.chunks(8).all(|other| other == first.as_slice());
    assert!(every_row_alike, "every row of the edge filters alike");

    first
}

#[test]
fn the_reference_matches_every_plugin_fixture() {
    let mut largest = 0.0f32;

    for fixture in H264_FIXTURES {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let expected = fixture.output_planes();
        let case_thresholds = case.thresholds();

        for (index, values) in inputs.iter().enumerate() {
            let plane = case.plane(index, values);
            let found = reference_h264(plane, case.samples, &case_thresholds);

            let label = format!("{} plane {index}", fixture.name);
            let difference = assert_h264_plane(&found, &expected[index], plane.width, case.samples, &label);
            largest = largest.max(difference);
        }
    }

    println!("the largest float difference from the plugin is {largest:e}");
}

#[test]
fn the_fixtures_with_a_zero_alpha_change_nothing() {
    let unchanged: Vec<_> = H264_FIXTURES
        .iter()
        .filter(|fixture| fixture.case().thresholds().alpha == 0.0)
        .collect();

    assert_eq!(unchanged.len(), 2, "two fixtures have an alpha index below 16");

    for fixture in unchanged {
        assert_eq!(
            fixture.input_planes(),
            fixture.output_planes(),
            "{}",
            fixture.name
        );
    }
}

#[test]
fn a_passing_edge_moves_two_samples_each_side_rounding_shifts_down() {
    let row = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0];

    let found = filtered_row(row, 51, MatrixSamples::U8);

    // `delta_q1` is `(110 + 105 - 220) >> 1`, which is -3 because the shift rounds down.
    assert_eq!(
        found,
        vec![100.0, 100.0, 102.0, 104.0, 106.0, 107.0, 110.0, 110.0]
    );
}

#[test]
fn every_move_stays_within_its_limit() {
    let row = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0];

    let found = filtered_row(row, 25, MatrixSamples::U8);

    // At quant 25, `c0` is 1 and `c` is 3, so `delta` 4 becomes 3 and both second samples move by 1.
    assert_eq!(
        found,
        vec![100.0, 100.0, 101.0, 103.0, 107.0, 109.0, 110.0, 110.0]
    );
}

#[test]
fn a_top_edge_filters_each_column_like_a_left_edge_filters_each_row() {
    let mut plane: Vec<f64> = (0..32)
        .map(|index| if index < 16 { 100.0 } else { 110.0 })
        .collect();
    let found_thresholds = thresholds(51, 0, 0, MatrixSamples::U8);

    reference_h264_padded(&mut plane, 4, 8, MatrixSamples::U8, &found_thresholds);

    let columns: Vec<f64> = plane.chunks(4).map(|row| row[0]).collect();
    assert_eq!(
        columns,
        vec![100.0, 100.0, 102.0, 104.0, 106.0, 107.0, 110.0, 110.0]
    );

    let every_column_alike = plane
        .chunks(4)
        .all(|row| row.iter().all(|value| *value == row[0]));
    assert!(every_column_alike);
}

#[test]
fn an_edge_failing_its_alpha_test_is_left_alone() {
    let row = [100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0];

    let found = filtered_row(row, 25, MatrixSamples::U8);

    assert_eq!(found, row.to_vec());
}

#[test]
fn an_edge_failing_its_beta_test_is_left_alone() {
    let row = [100.0, 100.0, 96.0, 100.0, 110.0, 110.0, 110.0, 110.0];

    let found = filtered_row(row, 25, MatrixSamples::U8);

    assert_eq!(found, row.to_vec());
}

#[test]
fn samples_clamp_at_zero_and_at_the_peak() {
    let near_zero = [26.0, 26.0, 26.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let near_peak = [229.0, 229.0, 229.0, 255.0, 255.0, 255.0, 255.0, 255.0];

    let low = filtered_row(near_zero, 60, MatrixSamples::U8);
    let high = filtered_row(near_peak, 60, MatrixSamples::U8);

    // Unclamped, `q0` would be -3 and 258.
    assert_eq!(low, vec![26.0, 26.0, 13.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(high, vec![229.0, 229.0, 242.0, 252.0, 255.0, 255.0, 255.0, 255.0]);
}

#[test]
fn an_index_below_16_changes_nothing() {
    let row = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0];

    let found = filtered_row(row, 15, MatrixSamples::U8);

    assert_eq!(found, row.to_vec());
}

#[test]
fn a_negative_offset_taking_the_index_below_16_changes_nothing() {
    let mut plane = blocky_plane(64, 48, MatrixSamples::U8, 3);
    let original = plane.clone();
    let lowered = thresholds(30, -15, 0, MatrixSamples::U8);

    reference_h264_padded(&mut plane, 64, 48, MatrixSamples::U8, &lowered);

    assert_eq!(plane, original);
}

#[test]
fn float_samples_move_without_rounding() {
    let codes = [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0];
    let row = codes.map(|code: f64| f64::from(code as f32 / 255.0));

    let found = filtered_row(row, 51, MatrixSamples::F32);

    let expected = [100.0, 100.0, 102.5, 103.75, 106.25, 107.5, 110.0, 110.0];
    for (value, code) in found.iter().zip(expected) {
        let wanted = code / 255.0;

        assert!(
            (value - wanted).abs() < f64::from(H264_FLOAT_TOLERANCE),
            "{value} against {wanted}"
        );
    }
}
