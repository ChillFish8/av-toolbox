use crate::retinex::{RecursiveGaussian, sections, solve_q};
use crate::testing::{reference_recursive_gaussian, reference_recursive_gaussian_cascade};

/// The sigmas the design is checked across, between 0.5 and 1024.
const SIGMAS: [f32; 7] = [0.5, 1.0, 2.0, 5.0, 50.0, 350.0, 1024.0];

#[test]
fn q_solves_the_variance_equation() {
    let expected = [
        0.402206, 0.575892, 1.000002, 2.381052, 23.578445, 165.032947, 482.838379,
    ];

    for (sigma, want) in SIGMAS.into_iter().zip(expected) {
        let sigma = f64::from(sigma);
        let q = solve_q(sigma);

        assert!(
            (q - want).abs() < 1e-5,
            "sigma {sigma} gave q {q}, expected {want}"
        );
    }
}

#[test]
fn the_sections_match_the_scaled_poles() {
    let (pole, pair_a, pair_b) = sections(5.0);

    assert!((pole - 0.769622).abs() < 1e-6, "p {pole}");
    assert!((pair_a - 1.532192).abs() < 1e-6, "a {pair_a}");
    assert!((pair_b - (-0.628372)).abs() < 1e-6, "b {pair_b}");
}

#[test]
fn each_section_has_unit_gain() {
    for sigma in SIGMAS {
        let filter = RecursiveGaussian::new(sigma);

        let real_total = filter.real_gain + filter.real_pole;
        let pair_total = filter.pair_gain + filter.pair[0] + filter.pair[1];

        assert!(
            (real_total - 1.0).abs() < 1e-6,
            "sigma {sigma} real total {real_total}"
        );
        assert!(
            (pair_total - 1.0).abs() < 1e-6,
            "sigma {sigma} pair total {pair_total}"
        );
    }
}

#[test]
fn the_reference_has_the_requested_spread() {
    for sigma in [0.5f32, 1.0, 5.0, 50.0, 350.0] {
        let length = (40.0 * sigma) as u32 + 1;
        let centre = (length / 2) as usize;
        let mut line = vec![0.0f32; length as usize];
        line[centre] = 1.0;

        let response = reference_recursive_gaussian(&line, length, 1, sigma);

        let total: f64 = response.iter().map(|value| f64::from(*value)).sum();
        let mean = response
            .iter()
            .enumerate()
            .map(|(index, value)| index as f64 * f64::from(*value))
            .sum::<f64>()
            / total;
        let variance = response
            .iter()
            .enumerate()
            .map(|(index, value)| (index as f64 - mean).powi(2) * f64::from(*value))
            .sum::<f64>()
            / total;
        let spread = variance.sqrt();
        let target = f64::from(sigma);
        let relative = (spread / target - 1.0).abs();

        let peak = response.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let most_negative = response.iter().copied().fold(f32::INFINITY, f32::min);
        let dip = -most_negative / peak;

        assert!(relative < 1e-4, "sigma {sigma} spread {spread}");
        assert!(
            dip <= 0.016,
            "sigma {sigma} dips to {most_negative} of peak {peak}"
        );
    }
}

#[test]
fn the_edges_act_as_replicated() {
    for sigma in [2.0f32, 5.0, 50.0, 350.0] {
        let width = (6.0 * sigma).max(200.0) as u32;
        let step = width - sigma as u32;
        let mut line = vec![0.0f32; width as usize];
        for value in line.iter_mut().skip(step as usize) {
            *value = 1.0;
        }

        let filter = RecursiveGaussian::new(sigma);
        let cascade = reference_recursive_gaussian_cascade(&line, width, 1, filter);
        let reference = reference_recursive_gaussian(&line, width, 1, sigma);

        let worst = cascade
            .iter()
            .zip(reference.iter())
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);

        assert!(worst < 2e-4, "sigma {sigma} missed by {worst}");
    }
}

#[test]
fn a_flat_plane_stays_flat() {
    let width = 7;
    let height = 5;
    let plane = vec![42.0; (width * height) as usize];

    for sigma in [1.0, 50.0, 1024.0] {
        let filter = RecursiveGaussian::new(sigma);
        let output = reference_recursive_gaussian(&plane, width, height, sigma);
        let cascade = reference_recursive_gaussian_cascade(&plane, width, height, filter);

        for value in output.into_iter().chain(cascade) {
            assert!((value - 42.0).abs() < 1e-5, "sigma {sigma} gave {value}");
        }
    }
}

#[test]
fn a_single_sample_plane_is_unchanged() {
    let plane = [9.0];

    let filter = RecursiveGaussian::new(1.0);
    let output = reference_recursive_gaussian(&plane, 1, 1, 1.0);
    let cascade = reference_recursive_gaussian_cascade(&plane, 1, 1, filter);

    assert!((output[0] - 9.0).abs() < 1e-5, "{}", output[0]);
    assert_eq!(cascade[0], 9.0);
}

#[test]
fn an_impulse_spreads_both_ways() {
    let plane = [0.0, 0.0, 10.0, 0.0, 0.0];

    let output = reference_recursive_gaussian(&plane, 5, 1, 1.0);

    assert!((output[0] - output[4]).abs() < 1e-6, "{output:?}");
    assert!((output[1] - output[3]).abs() < 1e-6, "{output:?}");
    assert!(output[2] > output[1] && output[1] > output[0], "{output:?}");
    assert!(output[0] > 0.0, "{output:?}");
}

#[test]
fn the_f32_cascade_matches_the_f64_reference_on_a_wide_step() {
    let width = 1920u32;
    let height = 1u32;
    let midpoint = width as usize / 2;
    let mut plane = vec![0.0f32; width as usize];
    for value in plane.iter_mut().skip(midpoint) {
        *value = 1.0;
    }

    for sigma in [200.0, 350.0] {
        let filter = RecursiveGaussian::new(sigma);
        let reference = reference_recursive_gaussian(&plane, width, height, sigma);
        let cascade = reference_recursive_gaussian_cascade(&plane, width, height, filter);

        for (actual, expected) in cascade.iter().zip(reference.iter()) {
            assert!(
                (actual - expected).abs() < 1e-3,
                "sigma {sigma}: {actual} vs {expected}"
            );
        }
    }
}
