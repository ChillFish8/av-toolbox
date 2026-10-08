use crate::matrix::MatrixSamples;
use crate::testing::{
    QED_FIXTURES,
    QedChroma,
    QedPlane,
    assert_h264_plane,
    count_qed_differences,
    half_lowpass_planes,
    is_border_sample,
    reference_lowpass,
    reference_lowpass_mirror,
    reference_qed,
    reference_qed_plane,
};

#[test]
fn the_reference_matches_every_plugin_fixture() {
    for fixture in QED_FIXTURES {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let normal = fixture.normal_planes();
        let strong = fixture.strong_planes();
        let changes = fixture.change_planes();
        let output = fixture.output_planes();
        let sizes = case.plane_sizes();
        let qed_planes = case.qed_planes();

        let found = reference_qed(&case, &inputs);

        let mut differing = 0;
        for (index, (width, _)) in sizes.iter().enumerate() {
            let label = format!("{} plane {index}", fixture.name);
            let normal_label = format!("{label} normal pass");
            let strong_label = format!("{label} strong pass");

            assert_h264_plane(
                &found.normal[index],
                &normal[index],
                *width,
                case.samples,
                &normal_label,
            );
            assert_h264_plane(
                &found.strong[index],
                &strong[index],
                *width,
                case.samples,
                &strong_label,
            );
            let found_output = &found.output[index];
            if !qed_planes.contains(&index) {
                assert_eq!(
                    *found_output, output[index],
                    "{label} takes a pass, so matches exactly"
                );
                continue;
            }

            for (place, value) in found_output.iter().enumerate() {
                let column = place as u32 % width;
                let row = place as u32 / width;

                if is_border_sample(column, row) {
                    assert_eq!(*value, output[index][place], "{label} border at {column},{row}");
                }
            }

            differing += count_qed_differences(found_output, &output[index], *width, case.samples, &label);
        }

        for (position, index) in qed_planes.iter().enumerate() {
            let (width, _) = sizes[*index];
            let label = format!("{} plane {index} change map", fixture.name);

            assert_h264_plane(
                &found.changes[position],
                &changes[position],
                width,
                case.samples,
                &label,
            );
        }

        println!(
            "{}: {differing} samples 1 from the plugin, {} allowed",
            fixture.name, fixture.allowed
        );
        assert!(
            differing <= fixture.allowed,
            "{}: {differing} samples differ from the plugin, {} allowed",
            fixture.name,
            fixture.allowed,
        );
    }
}

#[test]
fn a_strong_pass_below_index_16_leaves_every_interior_at_its_source() {
    let below: Vec<_> = QED_FIXTURES
        .iter()
        .filter(|fixture| fixture.case().quant[1] < 16)
        .collect();

    assert_eq!(below.len(), 1, "one fixture has a strong quant below 16");
    for fixture in below {
        let case = fixture.case();
        let inputs = fixture.input_planes();
        let normal = fixture.normal_planes();
        let output = fixture.output_planes();
        let sizes = case.plane_sizes();

        assert_eq!(
            fixture.strong_planes(),
            inputs,
            "{}: the strong pass changed something",
            fixture.name
        );
        for (index, (width, _)) in sizes.iter().enumerate() {
            for (place, value) in output[index].iter().enumerate() {
                let column = place as u32 % width;
                let row = place as u32 / width;
                let expected = if is_border_sample(column, row) {
                    normal[index][place]
                } else {
                    inputs[index][place]
                };

                assert_eq!(
                    *value, expected,
                    "{} plane {index} at {column},{row}",
                    fixture.name
                );
            }
        }
    }
}

#[test]
fn chroma_under_normal_or_strong_takes_that_pass() {
    for fixture in QED_FIXTURES {
        let case = fixture.case();
        let chosen = match case.chroma {
            QedChroma::Qed => continue,
            QedChroma::Normal => fixture.normal_planes(),
            QedChroma::Strong => fixture.strong_planes(),
        };
        let output = fixture.output_planes();

        assert_eq!(output[1..], chosen[1..], "{}", fixture.name);
    }
}

#[test]
fn the_border_is_columns_and_rows_0_and_7_of_each_block() {
    let border = [(0, 3), (7, 3), (3, 0), (3, 7), (8, 9), (15, 12), (71, 71)];
    let interior = [(1, 3), (6, 3), (3, 1), (3, 6), (9, 9), (14, 13), (65, 65)];

    for (column, row) in border {
        assert!(is_border_sample(column, row), "{column},{row} is border");
    }

    for (column, row) in interior {
        assert!(!is_border_sample(column, row), "{column},{row} is interior");
    }
}

#[test]
fn the_mirror_repeats_the_edge_then_reads_0() {
    let three: Vec<u32> = (0..8).map(|index| reference_lowpass_mirror(index, 3)).collect();
    let two: Vec<u32> = (0..8).map(|index| reference_lowpass_mirror(index, 2)).collect();
    let seventy: Vec<u32> = (68..80)
        .map(|index| reference_lowpass_mirror(index, 70))
        .collect();

    assert_eq!(three, vec![0, 1, 2, 2, 1, 0, 0, 0]);
    assert_eq!(two, vec![0, 1, 1, 0, 0, 0, 0, 0]);
    assert_eq!(seventy, vec![68, 69, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60]);
}

#[test]
fn a_tiny_plane_reads_its_mirror_clamped_to_0() {
    let change: Vec<f64> = (0..6).map(|value| f64::from(value * 10)).collect();
    let rows = [0usize, 1, 1, 0, 0, 0, 0, 0];
    let columns = [0usize, 1, 2, 2, 1, 0, 0, 0];
    let expanded: Vec<f64> = rows
        .iter()
        .flat_map(|row| columns.iter().map(move |column| row * 3 + column))
        .map(|index| change[index])
        .collect();

    let found = reference_lowpass(&change, 3, 2, (8, 8));
    let whole = reference_lowpass(&expanded, 8, 8, (8, 8));

    let cropped: Vec<f64> = whole
        .chunks(8)
        .take(2)
        .flat_map(|row| row[..3].to_vec())
        .collect();
    assert_eq!(found, cropped);
}

#[test]
fn a_constant_change_map_comes_back_unchanged() {
    let change = vec![1000.0; 70 * 46];

    let found = reference_lowpass(&change, 70, 46, (80, 48));

    assert!(found.iter().all(|value| *value == 1000.0));
}

#[test]
fn a_lowpass_of_exactly_half_rounds_away_from_zero() {
    let (source, strong) = half_lowpass_planes();
    let plane = QedPlane {
        source: &source,
        normal: &source,
        strong: &strong,
        width: 16,
        height: 16,
        lowpass_width: 16,
        lowpass_height: 16,
    };

    let found = reference_qed_plane(plane, MatrixSamples::U8);

    for (index, value) in found.iter().enumerate() {
        let column = index as u32 % 16;
        let row = index as u32 / 16;

        // Every low-pass is exactly 128.5, which rounds to 129, one above neutral.
        let expected = if is_border_sample(column, row) {
            100.0
        } else {
            99.0
        };
        assert_eq!(*value, expected, "{column},{row}");
    }
}
