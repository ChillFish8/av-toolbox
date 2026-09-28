use crate::retinex::{RecursiveGaussian, direct_form_coefficients, factor};
use crate::testing::{reference_recursive_gaussian, reference_recursive_gaussian_cascade};

/// The sigmas the factoring is checked across, widening from 0.5 to 350.
const SIGMAS: [f32; 7] = [0.5, 1.0, 2.5, 5.0, 50.0, 200.0, 350.0];

#[test]
fn the_coefficients_below_2_5_follow_the_small_sigma_formula() {
    let (gain, f1, f2, f3) = direct_form_coefficients(1.0);

    assert!((gain - 0.541422).abs() < 1e-5);
    assert!((f1 - 0.570436).abs() < 1e-5);
    assert!((f2 - (-0.123131)).abs() < 1e-5);
    assert!((f3 - 0.011274).abs() < 1e-5);
}

#[test]
fn the_coefficients_above_2_5_follow_the_large_sigma_formula() {
    let (gain, f1, f2, f3) = direct_form_coefficients(5.0);

    assert!((gain - 0.026183).abs() < 1e-5);
    assert!((f1 - 2.225551).abs() < 1e-5);
    assert!((f2 - (-1.690706)).abs() < 1e-5);
    assert!((f3 - 0.438972).abs() < 1e-5);
}

#[test]
fn the_factoring_reproduces_the_direct_form_feedback() {
    for sigma in SIGMAS {
        let (_, f1, f2, f3) = direct_form_coefficients(sigma);
        let (pole, pair_a, pair_b) = factor(f1, f2, f3);

        let recovered_f1 = pole + pair_a;
        let recovered_f2 = pair_b - pole * pair_a;
        let recovered_f3 = -(pole * pair_b);

        assert!(
            (recovered_f1 - f1).abs() < 1e-9,
            "sigma {sigma} f1 {recovered_f1} vs {f1}"
        );
        assert!(
            (recovered_f2 - f2).abs() < 1e-9,
            "sigma {sigma} f2 {recovered_f2} vs {f2}"
        );
        assert!(
            (recovered_f3 - f3).abs() < 1e-9,
            "sigma {sigma} f3 {recovered_f3} vs {f3}"
        );
    }
}

#[test]
fn the_real_pole_lies_between_zero_and_one() {
    for sigma in SIGMAS {
        let (_, f1, f2, f3) = direct_form_coefficients(sigma);
        let (pole, _, _) = factor(f1, f2, f3);

        assert!(pole > 0.0 && pole < 1.0, "sigma {sigma} gave pole {pole}");
    }
}

#[test]
fn each_sections_gains_sum_to_one() {
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
fn a_flat_plane_stays_flat() {
    let width = 7;
    let height = 5;
    let plane = vec![42.0; (width * height) as usize];

    for sigma in [1.0, 50.0, 350.0] {
        let output = reference_recursive_gaussian(&plane, width, height, sigma);

        for value in output {
            assert!((value - 42.0).abs() < 1e-4, "sigma {sigma} gave {value}");
        }
    }
}

#[test]
fn a_single_sample_plane_is_unchanged() {
    let plane = [9.0];

    let output = reference_recursive_gaussian(&plane, 1, 1, 1.0);

    assert!((output[0] - 9.0).abs() < 1e-5, "{}", output[0]);
}

#[test]
fn an_impulse_spreads_both_ways() {
    let plane = [0.0, 0.0, 10.0, 0.0, 0.0];

    let output = reference_recursive_gaussian(&plane, 5, 1, 1.0);

    let expected = [0.6978, 2.0544, 4.0492, 2.1744, 1.0951];
    for (actual, want) in output.iter().zip(expected) {
        assert!((actual - want).abs() < 1e-3, "{actual} vs {want}");
    }
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
