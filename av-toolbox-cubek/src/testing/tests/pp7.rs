use crate::deblock::pp7::{Pp7Mode, thresholds};
use crate::matrix::MatrixSamples;
use crate::testing::{
    PP7_FIXTURES,
    Pp7Plane,
    count_pp7_differences,
    plugin_output_at_peak,
    reference_coefficients,
    reference_extended_index,
    reference_integer_output,
    reference_kept_float,
    reference_kept_integer,
    reference_pp7,
    reference_pp7_plane,
    reference_transform,
};

/// The fixture whose plugin output rings past its 10-bit peak.
const RINGING_FIXTURE: &str = "grey_47x30_10_q10_hard_binary";

#[test]
fn the_reference_matches_every_plugin_fixture() {
    for fixture in PP7_FIXTURES {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let outputs = fixture.output_planes();
        let sizes = case.plane_sizes();

        let found = reference_pp7(&case, &inputs);

        let mut differing = 0;
        for (index, (width, _)) in sizes.iter().enumerate() {
            let label = format!("{} plane {index}", fixture.name);
            let (plugin, _) = plugin_output_at_peak(&outputs[index], case.samples);

            differing += count_pp7_differences(&found[index], &plugin, *width, case.samples, &label);
        }

        println!(
            "{}: {differing} samples past the float tolerance, {} allowed",
            fixture.name, fixture.allowed
        );
        assert!(
            differing <= fixture.allowed,
            "{}: {differing} samples past the float tolerance, {} allowed",
            fixture.name,
            fixture.allowed,
        );
    }
}

#[test]
fn only_the_ringing_fixture_has_plugin_samples_past_its_peak() {
    for fixture in PP7_FIXTURES {
        let case = fixture.case();
        let outputs = fixture.output_planes();

        let brought: usize = outputs
            .iter()
            .map(|plane| plugin_output_at_peak(plane, case.samples).1)
            .sum();

        let expected = if fixture.name == RINGING_FIXTURE { 95 } else { 0 };
        assert_eq!(brought, expected, "{}", fixture.name);
    }
}

#[test]
fn one_tap_transforms_to_its_column_of_the_folded_core_rows() {
    let rows: [[i64; 7]; 4] = [
        [1, 1, 1, 2, 1, 1, 1],
        [-2, -1, 1, 4, 1, -1, -2],
        [1, -1, -1, 2, -1, -1, 1],
        [-1, 2, -2, 2, -2, 2, -1],
    ];

    for tap in 0..7 {
        let mut taps = [0i64; 7];
        taps[tap] = 1;

        let found = reference_transform(taps);

        let expected = rows.map(|row| row[tap]);
        assert_eq!(found, expected, "tap {tap}");
    }
}

#[test]
fn an_odd_neighbourhood_has_no_coefficients() {
    let taps = [5i64, -3, 7, 0, -7, 3, -5];
    let square: [[i64; 7]; 7] =
        std::array::from_fn(|row| std::array::from_fn(|column| (row as i64 + 1) * taps[column]));

    let transformed = reference_transform(taps);
    let coefficients = reference_coefficients(&square);

    assert_eq!(transformed, [0; 4]);
    assert_eq!(coefficients, [0; 16]);
}

#[test]
fn a_flat_square_has_only_its_dc_coefficient() {
    let square = [[3i64; 7]; 7];

    let found = reference_coefficients(&square);

    assert_eq!(found[0], 192);
    assert!(found[1..].iter().all(|coefficient| *coefficient == 0));
}

#[test]
fn coefficients_take_ky_down_the_square_and_kx_across_it() {
    let down = [1i64, 0, 2, 5, 3, 1, 4];
    let across = [2i64, 7, 1, 0, 3, 3, 1];
    let square: [[i64; 7]; 7] =
        std::array::from_fn(|row| std::array::from_fn(|column| down[row] * across[column]));

    let found = reference_coefficients(&square);

    let rows = reference_transform(down);
    let columns = reference_transform(across);
    for ky in 0..4 {
        for kx in 0..4 {
            assert_eq!(
                found[ky * 4 + kx],
                rows[ky] * columns[kx],
                "coefficient {ky},{kx}"
            );
        }
    }
}

#[test]
fn integer_modes_keep_a_coefficient_only_past_its_threshold() {
    let cases = [
        (31, Pp7Mode::Hard, 0),
        (32, Pp7Mode::Hard, 32),
        (-32, Pp7Mode::Hard, -32),
        (31, Pp7Mode::Soft, 0),
        (32, Pp7Mode::Soft, 1),
        (-40, Pp7Mode::Soft, -9),
        (31, Pp7Mode::Medium, 0),
        (32, Pp7Mode::Medium, 2),
        (-33, Pp7Mode::Medium, -4),
        (62, Pp7Mode::Medium, 62),
        (63, Pp7Mode::Medium, 63),
    ];

    for (value, mode, expected) in cases {
        let kept = reference_kept_integer(value, 31, mode);

        assert_eq!(kept, expected, "{value} under {mode:?}");
    }
}

#[test]
fn float_modes_decide_on_the_magnitude_rounded_down() {
    let cases = [
        (31.9, Pp7Mode::Hard, 0.0),
        (32.0, Pp7Mode::Hard, 32.0),
        (-32.25, Pp7Mode::Soft, -1.25),
        (31.9, Pp7Mode::Soft, 0.0),
        (32.5, Pp7Mode::Medium, 3.0),
        (62.5, Pp7Mode::Medium, 63.0),
        (63.5, Pp7Mode::Medium, 63.5),
        (-62.5, Pp7Mode::Medium, -63.0),
    ];

    for (value, mode, expected) in cases {
        let kept = reference_kept_float(value, 31.0, mode);

        assert_eq!(kept, expected, "{value} under {mode:?}");
    }
}

#[test]
fn integer_outputs_round_half_up_and_stay_between_0_and_the_peak() {
    let limits = thresholds(2.0, MatrixSamples::U8);
    let dc_only = |value: i64| {
        let mut found = [0i64; 16];
        found[0] = value;
        found
    };

    let thirty_two = dc_only(32);
    let thirty_one = dc_only(31);
    let below_zero = dc_only(-200);
    let large = dc_only(64 * 300);

    // A DC of 32 weighs exactly half of one code, and 31 just under it.
    let half_code = reference_integer_output(&thirty_two, &limits, Pp7Mode::Hard, 255);
    let under_half = reference_integer_output(&thirty_one, &limits, Pp7Mode::Hard, 255);
    let negative = reference_integer_output(&below_zero, &limits, Pp7Mode::Hard, 255);
    let above_peak = reference_integer_output(&large, &limits, Pp7Mode::Hard, 255);
    let deeper_peak = reference_integer_output(&large, &limits, Pp7Mode::Hard, 1023);

    assert_eq!(half_code, 1);
    assert_eq!(under_half, 0);
    assert_eq!(negative, 0);
    assert_eq!(above_peak, 255);
    assert_eq!(deeper_peak, 300);
}

#[test]
fn the_border_mirrors_with_the_edge_sample_repeated() {
    let left: Vec<u32> = (-3..0)
        .map(|position| reference_extended_index(position, 45, 48))
        .collect();
    let right: Vec<u32> = (45..48)
        .map(|position| reference_extended_index(position, 45, 48))
        .collect();

    assert_eq!(left, vec![2, 1, 0]);
    assert_eq!(right, vec![44, 43, 42]);
}

#[test]
fn a_side_just_short_of_its_alignment_folds_back_again_past_it() {
    let fifteen: Vec<u32> = (47..50)
        .map(|position| reference_extended_index(position, 47, 48))
        .collect();
    let fourteen: Vec<u32> = (46..49)
        .map(|position| reference_extended_index(position, 46, 48))
        .collect();
    let chroma: Vec<u32> = (23..26)
        .map(|position| reference_extended_index(position, 23, 24))
        .collect();

    assert_eq!(fifteen, vec![46, 46, 46]);
    assert_eq!(fourteen, vec![45, 44, 44]);
    assert_eq!(chroma, vec![22, 22, 22]);
}

#[test]
fn a_tiny_side_mirrors_back_and_forth() {
    let two: Vec<u32> = (-3..5)
        .map(|position| reference_extended_index(position, 2, 16))
        .collect();
    let one: Vec<u32> = (-3..4)
        .map(|position| reference_extended_index(position, 1, 16))
        .collect();

    assert_eq!(two, vec![1, 1, 0, 0, 1, 1, 0, 0]);
    assert_eq!(one, vec![0; 7]);
}

#[test]
fn a_flat_integer_plane_comes_back_unchanged_in_every_mode() {
    let depths = [
        (MatrixSamples::U8, 200.0),
        (MatrixSamples::U16 { bits: 10 }, 1000.0),
        (MatrixSamples::U16 { bits: 16 }, 65535.0),
    ];
    let modes = [Pp7Mode::Hard, Pp7Mode::Soft, Pp7Mode::Medium];

    for (samples, level) in depths {
        let values = vec![level; 20 * 9];
        let plane = Pp7Plane {
            values: &values,
            width: 20,
            height: 9,
            aligned_width: 32,
            aligned_height: 16,
        };

        for mode in modes {
            let found = reference_pp7_plane(plane, samples, 63.0, mode);

            assert_eq!(found, values, "{samples:?} under {mode:?}");
        }
    }
}

#[test]
fn float_coefficients_are_summed_columns_first() {
    let square: [[f32; 7]; 7] = std::array::from_fn(|row| {
        std::array::from_fn(|column| 0.1 + (row * 7 + column) as f32 * 0.37 + (row * column) as f32 * 0.013)
    });

    let found = reference_coefficients(&square);

    let mut columns_first = [0.0f32; 16];
    let mut rows_first = [0.0f32; 16];
    let down: Vec<[f32; 4]> = (0..7)
        .map(|column| reference_transform(std::array::from_fn(|row| square[row][column])))
        .collect();
    let across: Vec<[f32; 4]> = (0..7).map(|row| reference_transform(square[row])).collect();
    for ky in 0..4 {
        let taps_down = std::array::from_fn(|column| down[column][ky]);
        let columns_then_rows = reference_transform(taps_down);
        columns_first[ky * 4..ky * 4 + 4].copy_from_slice(&columns_then_rows);

        for kx in 0..4 {
            let taps_across = std::array::from_fn(|row| across[row][kx]);
            let rows_then_columns = reference_transform(taps_across);
            rows_first[ky * 4 + kx] = rows_then_columns[ky];
        }
    }

    assert_ne!(
        columns_first, rows_first,
        "the square cannot tell the two orders apart"
    );
    assert_eq!(found, columns_first);
}
