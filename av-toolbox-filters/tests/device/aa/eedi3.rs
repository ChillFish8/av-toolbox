use av_toolbox_cubek::testing::{
    CloseDecisionTally,
    Eedi3Antialias,
    Eedi3Clips,
    Eedi3Parameters,
    Eedi3PlaneKind,
    SplitMix,
    random_eedi3_plane,
    reference_eedi3_antialias,
    transposed_plane,
};
use av_toolbox_gpu::{Depth, FrameFormat, SampleFormat};
use serde_json::json;

use super::{
    DIRECTIONS,
    assert_f32_matches_sixteen_bit,
    decoded,
    grey,
    largest_vertical_step,
    noisy_diagonals,
    one_sample_diagonal,
    scramble,
    sixteen_bit,
    yuv420,
};
use crate::support::{
    ROUND_TRIP,
    assert_codes_within,
    code_bits,
    encoded,
    run_masked_into_codes,
    run_once,
    shared_planes,
    ten_bit,
    uniform_mask,
    widened,
};

const STAGE: &str = "aa/eedi3";

const WIDTH: u32 = 96;

const HEIGHT: u32 = 64;

/// A shallow and a steep aliased line, each two samples thick, over a faint noisy texture, scaled to
/// `sample`'s depth.
///
/// The shallow line falls one row every 3 columns and the steep line moves one column every 3 rows, so
/// each tests the pass that runs across it.
fn aliased_lines(width: u32, height: u32, sample: SampleFormat) -> Vec<u16> {
    let bits = code_bits(sample);
    let scale = 1 << (bits - 8);

    (0..width * height)
        .map(|index| {
            let column = index % width;
            let row = index / width;
            let shallow_row = height / 4 + column / 3;
            let steep_column = width / 4 + row / 3;
            let on_shallow = row >= shallow_row && row < shallow_row + 2;
            let on_steep = column >= steep_column && column < steep_column + 2;
            let texture = scramble(column, row) % 12;
            let value = match (on_shallow, on_steep) {
                (true, _) => 220,
                (false, true) => 180,
                (false, false) => 40 + texture,
            };

            (value * scale) as u16
        })
        .collect()
}

/// Every plane of `format`, each holding aliased lines at its own size.
fn line_planes(format: &FrameFormat) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            aliased_lines(width, height, format.sample)
        })
        .collect()
}

/// The default arguments at `sample`'s depth.
fn default_parameters(sample: SampleFormat) -> Eedi3Parameters {
    Eedi3Parameters {
        bits: code_bits(sample),
        ..Eedi3Parameters::default()
    }
}

/// The arguments of `parameters` as `aa/eedi3` takes them, along `direction` on every plane of `format`.
fn stage_arguments(parameters: &Eedi3Parameters, format: &FrameFormat, direction: &str) -> serde_json::Value {
    let planes: Vec<String> = format
        .channels()
        .iter()
        .map(|channel| channel.to_string().to_lowercase())
        .collect();

    json!({
        "alpha": parameters.alpha,
        "beta": parameters.beta,
        "gamma": parameters.gamma,
        "nrad": parameters.nrad,
        "mdis": parameters.mdis,
        "vcheck": parameters.vcheck,
        "vthresh": parameters.vthresh,
        "planes": planes,
        "direction": direction,
    })
}

/// Runs every plane of a `format` clip of `values` through `aa/eedi3` with `parameters` along `direction`.
fn run_stage(
    format: FrameFormat,
    values: &[Vec<u16>],
    parameters: &Eedi3Parameters,
    direction: &str,
) -> Vec<Vec<u16>> {
    let planes = encoded(&format, values);
    let args = stage_arguments(parameters, &format, direction);

    let filtered = run_once(STAGE, format, format, args, &planes);

    decoded(&format, &filtered)
}

fn vertical_reference(
    values: &[f32],
    width: u32,
    height: u32,
    parameters: &Eedi3Parameters,
) -> Eedi3Antialias {
    let clips = Eedi3Clips::default();

    reference_eedi3_antialias(values, width, height, parameters, clips)
}

/// The reference run on the transposed plane, with its output and touched samples transposed back.
fn horizontal_reference(
    values: &[f32],
    width: u32,
    height: u32,
    parameters: &Eedi3Parameters,
) -> Eedi3Antialias {
    let swapped = transposed_plane(values, width, height);
    let filtered = vertical_reference(&swapped, height, width, parameters);

    let output = transposed_plane(&filtered.output, height, width);
    let touched = transposed_plane(&filtered.touched, height, width);

    Eedi3Antialias { output, touched }
}

/// The reference's answer on every plane of a `format` clip of `values` along `direction`.
///
/// For "both", the horizontal reference runs on the stage's own vertical output, so it never inherits the
/// vertical pass's close decisions.
fn expected_planes(
    format: FrameFormat,
    values: &[Vec<u16>],
    parameters: &Eedi3Parameters,
    direction: &str,
) -> Vec<Eedi3Antialias> {
    let mut inputs = values.to_vec();
    if direction == "both" {
        inputs = run_stage(format, values, parameters, "vertical");
    }

    let channels = format.channels();
    channels
        .iter()
        .zip(&inputs)
        .map(|(channel, plane)| {
            let (width, height) = format.channel_dimensions(*channel).expect("a listed channel");
            let plane_values = widened(plane);

            match direction {
                "vertical" => vertical_reference(&plane_values, width, height, parameters),
                _ => horizontal_reference(&plane_values, width, height, parameters),
            }
        })
        .collect()
}

/// Adds every plane the stage gave for `found` to `tally` against `expected`.
fn compare_planes(
    tally: &mut CloseDecisionTally,
    format: &FrameFormat,
    found: &[Vec<u16>],
    expected: &[Eedi3Antialias],
    label: &str,
) {
    let channels = format.channels();
    for (index, channel) in channels.iter().enumerate() {
        let (width, _) = format.channel_dimensions(*channel).expect("a listed channel");
        let found_values = widened(&found[index]);
        let plane_label = format!("{label} {channel:?}");

        tally.compare(
            &plane_label,
            &found_values,
            &expected[index].output,
            &expected[index].touched,
            width,
        );
    }
}

/// Runs a `format` clip of `values` along `direction` and adds it to `tally` against the reference.
fn tally_run(
    tally: &mut CloseDecisionTally,
    format: FrameFormat,
    values: &[Vec<u16>],
    parameters: &Eedi3Parameters,
    direction: &str,
) {
    let found = run_stage(format, values, parameters, direction);
    let expected = expected_planes(format, values, parameters, direction);

    let label = format!("{} {} {direction}", format.layout, format.sample);
    for (index, plane) in expected.iter().enumerate() {
        let widened_input = widened(&values[index]);
        assert_ne!(
            plane.output, widened_input,
            "{label} leaves plane {index} unchanged"
        );
    }

    compare_planes(tally, &format, &found, &expected, &label);
}

/// Runs a `Grey` clip of aliased lines at `sample`'s depth along each of `directions`, against the
/// reference in one tally.
fn assert_grey_matches(sample: SampleFormat, directions: &[&str]) {
    let format = grey(WIDTH, HEIGHT, sample);
    let values = [aliased_lines(WIDTH, HEIGHT, sample)];
    let parameters = default_parameters(sample);

    let mut tally = CloseDecisionTally::new();
    for direction in directions {
        tally_run(&mut tally, format, &values, &parameters, direction);
    }

    let label = format!("{sample} {directions:?}");
    assert_tally(&tally, &label);
}

/// Prints the tally's counts, then asserts its bounds.
fn assert_tally(tally: &CloseDecisionTally, label: &str) {
    println!("{label}: {tally:?}");

    tally.assert_bounds(label);
}

#[test]
fn vertical_matches_the_reference() {
    assert_grey_matches(SampleFormat::U8, &["vertical"]);
}

#[test]
fn horizontal_matches_the_transposed_reference() {
    assert_grey_matches(SampleFormat::U8, &["horizontal"]);
}

#[test]
fn both_matches_the_horizontal_reference_on_the_vertical_output() {
    assert_grey_matches(SampleFormat::U8, &["both"]);
}

#[test]
fn ten_bit_matches_the_reference() {
    let sample = ten_bit();

    assert_grey_matches(sample, &DIRECTIONS);
}

#[test]
fn sixteen_bit_matches_the_reference() {
    let sample = SampleFormat::U16 {
        depth: Depth::SIXTEEN,
    };

    assert_grey_matches(sample, &DIRECTIONS);
}

#[test]
fn every_plane_of_a_420_clip_matches_at_its_own_size() {
    let format = yuv420(WIDTH, HEIGHT, SampleFormat::U8);
    let values = line_planes(&format);
    let parameters = default_parameters(format.sample);

    let mut tally = CloseDecisionTally::new();
    for direction in DIRECTIONS {
        tally_run(&mut tally, format, &values, &parameters, direction);
    }

    assert_tally(&tally, "every plane of a 4:2:0 clip");
}

/// The settings `based_aa` runs EEDI3 with, at `bits` per sample.
fn based_aa_parameters(bits: u32) -> Eedi3Parameters {
    Eedi3Parameters {
        alpha: 0.125,
        gamma: 40.0,
        vthresh: [12.0, 24.0, 4.0],
        bits,
        ..Eedi3Parameters::default()
    }
}

/// Aliased lines over a noisy texture, then random sloped edges with and without noise, at `sample`'s
/// depth.
fn based_aa_planes(width: u32, height: u32, sample: SampleFormat) -> Vec<Vec<u16>> {
    let bits = code_bits(sample);
    let kinds = [
        Eedi3PlaneKind::NoisyEdge,
        Eedi3PlaneKind::Edge,
        Eedi3PlaneKind::NoisyEdge,
        Eedi3PlaneKind::Edge,
    ];
    let mut random = SplitMix(0x5eed_0bad);

    let mut planes = vec![aliased_lines(width, height, sample)];
    for kind in kinds {
        let plane = random_eedi3_plane(&mut random, width, height, bits, &[kind]);
        let samples = plane.iter().map(|value| *value as u16).collect();
        planes.push(samples);
    }

    planes
}

/// Runs `based_aa`'s settings on textured lines and random edges at 8 and 10 bit, in every direction.
#[test]
fn based_aa_settings_match_the_reference() {
    let width = 128;
    let height = 64;

    let mut tally = CloseDecisionTally::new();
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = grey(width, height, sample);
        let bits = code_bits(sample);
        let parameters = based_aa_parameters(bits);

        for plane in based_aa_planes(width, height, sample) {
            let values = [plane];
            for direction in DIRECTIONS {
                tally_run(&mut tally, format, &values, &parameters, direction);
            }
        }
    }

    assert_tally(&tally, "based_aa settings");
}

/// The arguments the sensitivity test moves off one at a time.
///
/// `alpha` and `beta` differ, so swapping them changes the output. Every `vthresh` is a power of two, whose
/// exact reciprocal keeps the reliability check's blends clear of close decisions.
const SENSITIVITY_BASE: Eedi3Parameters = Eedi3Parameters {
    alpha: 0.3,
    beta: 0.35,
    gamma: 30.0,
    nrad: 1,
    mdis: 12,
    vcheck: 2,
    vthresh: [32.0, 64.0, 4.0],
    bits: 8,
};

/// Each argument moved alone off [SENSITIVITY_BASE](crate::aa::eedi3::SENSITIVITY_BASE), by name.
fn moved_arguments() -> [(&'static str, Eedi3Parameters); 9] {
    let base = SENSITIVITY_BASE;

    [
        ("alpha", Eedi3Parameters { alpha: 0.6, ..base }),
        ("beta", Eedi3Parameters { beta: 0.65, ..base }),
        ("gamma", Eedi3Parameters { gamma: 2.0, ..base }),
        ("nrad", Eedi3Parameters { nrad: 3, ..base }),
        ("mdis", Eedi3Parameters { mdis: 4, ..base }),
        ("vcheck", Eedi3Parameters { vcheck: 3, ..base }),
        (
            "vthresh[0]",
            Eedi3Parameters {
                vthresh: [8.0, 64.0, 4.0],
                ..base
            },
        ),
        (
            "vthresh[1]",
            Eedi3Parameters {
                vthresh: [32.0, 16.0, 4.0],
                ..base
            },
        ),
        (
            "vthresh[2]",
            Eedi3Parameters {
                vthresh: [32.0, 64.0, 1.0],
                ..base
            },
        ),
    ]
}

/// Noisy stripes above shallow noisy diagonals, 8-bit.
fn sensitive_plane(width: u32, height: u32) -> Vec<u16> {
    let mut plane = noisy_diagonals(width, height / 2, 9, 70, 130, 20);
    let diagonals = noisy_diagonals(width, height / 2, 3, 60, 140, 24);
    plane.extend(diagonals);

    plane
}

/// The plane of [sensitive_plane](crate::aa::eedi3::sensitive_plane), transposed for a horizontal pass so
/// that it meets the same rows a vertical pass does.
fn sensitive_plane_along(width: u32, height: u32, direction: &str) -> Vec<u16> {
    if direction != "horizontal" {
        return sensitive_plane(width, height);
    }

    let turned = sensitive_plane(height, width);
    let widened_values = widened(&turned);
    let swapped = transposed_plane(&widened_values, height, width);

    swapped.iter().map(|value| *value as u16).collect()
}

/// The samples where `moved` and `base` differ and neither marks the sample as touched.
fn clear_changes(moved: &Eedi3Antialias, base: &Eedi3Antialias) -> usize {
    let outputs = moved.output.iter().zip(&base.output);
    let touched = moved.touched.iter().zip(&base.touched);

    outputs
        .zip(touched)
        .filter(|((moved_value, base_value), (moved_touched, base_touched))| {
            moved_value != base_value && !**moved_touched && !**base_touched
        })
        .count()
}

#[test]
fn each_argument_alone_reaches_the_device() {
    let width = 128;
    let height = 64;
    let format = grey(width, height, SampleFormat::U8);

    let mut tally = CloseDecisionTally::new();
    for direction in DIRECTIONS {
        let values = [sensitive_plane_along(width, height, direction)];
        let base = expected_planes(format, &values, &SENSITIVITY_BASE, direction);
        let found = run_stage(format, &values, &SENSITIVITY_BASE, direction);
        let label = format!("the base {direction}");
        compare_planes(&mut tally, &format, &found, &base, &label);

        for (name, parameters) in moved_arguments() {
            let expected = expected_planes(format, &values, &parameters, direction);
            let found = run_stage(format, &values, &parameters, direction);

            let label = format!("{name} moved {direction}");
            let clear_changes = clear_changes(&expected[0], &base[0]);
            assert!(
                clear_changes > 0,
                "{label} changes no sample outside the touched sets"
            );
            compare_planes(&mut tally, &format, &found, &expected, &label);
        }
    }

    assert_tally(&tally, "each argument alone");
}

#[test]
fn a_one_sample_diagonal_comes_back_with_smaller_steps() {
    let width = 32;
    let height = 24;
    let format = grey(width, height, SampleFormat::U8);
    let values = one_sample_diagonal(width, height);
    let plane_values = [values.clone()];
    let planes = encoded(&format, &plane_values);
    let before = largest_vertical_step(&values, width);

    for direction in ["vertical", "both"] {
        let args = json!({ "direction": direction });

        let filtered = run_once(STAGE, format, format, args, &planes);

        let found = decoded(&format, &filtered);
        let after = largest_vertical_step(&found[0], width);
        assert!(
            after < before,
            "{direction} leaves a step of {after}, the input's was {before}"
        );
    }
}

#[test]
fn only_the_selected_planes_change() {
    let format = yuv420(WIDTH, HEIGHT, SampleFormat::U8);

    let shared = shared_planes(STAGE, format, json!({ "planes": ["y"] }));

    assert_eq!(shared, vec![false, true, true]);
}

#[test]
fn a_zero_mask_hands_back_the_input() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let values = line_planes(&format);
        let planes = encoded(&format, &values);
        let mask = uniform_mask(&format, 0);
        let args = json!({ "planes": ["y", "u", "v"] });
        let bits = code_bits(sample);

        let found = run_masked_into_codes(STAGE, format, args, &planes, mask, bits);

        for (plane, codes) in found.iter().enumerate() {
            let input = widened(&values[plane]);
            let label = format!("{sample} plane {plane}");
            assert_codes_within(codes, &input, ROUND_TRIP, &label);
        }
    }
}

#[test]
fn a_flat_frame_is_unchanged() {
    for sample in [SampleFormat::U8, ten_bit()] {
        let format = yuv420(WIDTH, HEIGHT, sample);
        let flat = if sample == SampleFormat::U8 { 128u16 } else { 512 };
        let values: Vec<Vec<u16>> = format
            .channels()
            .iter()
            .map(|channel| {
                let samples = format.channel_samples(*channel).expect("a listed channel");
                vec![flat; samples]
            })
            .collect();
        let planes = encoded(&format, &values);
        let args = json!({ "planes": ["y", "u", "v"] });

        let filtered = run_once(STAGE, format, format, args, &planes);

        assert_eq!(filtered, planes, "{sample}");
    }
}

#[test]
fn eedi3_f32_matches_sixteen_bit() {
    let input = yuv420(WIDTH, HEIGHT, sixteen_bit());
    let values = line_planes(&input);
    let args = json!({ "planes": ["y", "u", "v"] });

    assert_f32_matches_sixteen_bit(STAGE, input, args, &values, 0.01, 0.0);
}
