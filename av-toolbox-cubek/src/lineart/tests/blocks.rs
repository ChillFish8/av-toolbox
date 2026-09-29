use crate::depth::Element;
use crate::lineart::ThinSettings;
use crate::testing::{reference_clamped_merge, reference_merge, reference_thin, reference_vertical_121};
use crate::warp::{BlurKind, WarpParams};

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

/// `60 + (0 - 60) * 51 / 255 = 48` and `200 + (255 - 200) * 51 / 255 = 211`.
#[test]
fn a_clamped_merge_holds_the_filtered_value_to_the_range_first() {
    let input = [60.0, 200.0];
    let filtered = [-67.0, 300.0];
    let weight = [51.0, 51.0];

    let merged = reference_clamped_merge(&input, &filtered, &weight, 255.0, 255.0);

    assert_eq!(merged, vec![48.0, 211.0]);
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

const RAMP_WIDTH: u32 = 64;

const RAMP_HEIGHT: u32 = 6;

fn thin_settings(threshold: f32, depth: i32, passes: u32) -> ThinSettings {
    let warp = WarpParams {
        element: Element::U8,
        bits: 8,
        thresh: 128,
        blur: BlurKind::R6,
        passes,
        depth,
    };

    ThinSettings {
        warp,
        threshold,
        radius: 1,
        peak: 255.0,
    }
}

/// A plane of 200 that falls 10 a column down to 40 in the middle and rises back, so its Prewitt
/// magnitude is `3 * 20 = 60` on every column of the slopes and less at the corners.
fn ramp_plane() -> Vec<f32> {
    (0..RAMP_WIDTH * RAMP_HEIGHT)
        .map(|index| {
            let column = (index % RAMP_WIDTH) as f32;
            let distance = (column - 32.0).abs();

            (40.0 + 10.0 * distance).min(200.0)
        })
        .collect()
}

#[test]
fn thin_leaves_a_flat_plane_unchanged() {
    let plane = vec![77.0f32; (RAMP_WIDTH * RAMP_HEIGHT) as usize];
    let settings = thin_settings(20.0, 24, 1);

    let thinned = reference_thin(&plane, RAMP_WIDTH, RAMP_HEIGHT, &settings);

    assert_eq!(thinned, plane);
}

#[test]
fn thin_hands_back_the_input_above_the_threshold_and_moves_it_at_the_threshold() {
    let plane = ramp_plane();
    let above = thin_settings(61.0, 24, 1);
    let on = thin_settings(60.0, 24, 1);

    let untouched = reference_thin(&plane, RAMP_WIDTH, RAMP_HEIGHT, &above);
    let thinned = reference_thin(&plane, RAMP_WIDTH, RAMP_HEIGHT, &on);

    let moved = thinned
        .iter()
        .zip(&plane)
        .filter(|(found, wanted)| found != wanted)
        .count();
    println!("{moved} samples move at a threshold of 60");
    assert_eq!(untouched, plane);
    assert!(moved > 0);
}

/// A plane of 200 with a 4-wide line of 40 at columns 30 to 33, whose middle samples sit beside the mask.
fn wide_line_plane() -> Vec<f32> {
    (0..RAMP_WIDTH * RAMP_HEIGHT)
        .map(|index| {
            if (30..34).contains(&(index % RAMP_WIDTH)) {
                40.0
            } else {
                200.0
            }
        })
        .collect()
}

/// The samples the reference moves differently at a radius of 0 and of 2, so `mrad` has a say.
#[test]
fn mrad_changes_the_thinned_scene() {
    let plane = wide_line_plane();
    let narrow = ThinSettings {
        radius: 0,
        ..thin_settings(60.0, 24, 1)
    };
    let wide = ThinSettings {
        radius: 2,
        ..thin_settings(60.0, 24, 1)
    };

    let without = reference_thin(&plane, RAMP_WIDTH, RAMP_HEIGHT, &narrow);
    let with = reference_thin(&plane, RAMP_WIDTH, RAMP_HEIGHT, &wide);

    let differing = without
        .iter()
        .zip(&with)
        .filter(|(left, right)| left != right)
        .count();
    println!("{differing} samples differ between mrad 0 and 2");
    assert!(differing > 0);
}
