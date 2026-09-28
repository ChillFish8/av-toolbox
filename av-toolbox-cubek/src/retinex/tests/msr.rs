use crate::retinex::RetinexSettings;
use crate::testing::{reference_log_response, reference_percentile, reference_retinex};

#[test]
fn a_flat_plane_gives_zero() {
    let width = 6;
    let height = 4;
    let input = vec![100.0f32; (width * height) as usize];
    let settings = RetinexSettings::new(vec![1.0, 3.0], 0.01, 0.01, 255.0);

    let output = reference_retinex(&input, width, height, &settings);

    for value in output {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, 0.0);
    }
}

#[test]
fn the_output_spans_zero_to_scale() {
    let width = 16;
    let height = 12;
    let samples = (width * height) as usize;
    let scale = 255.0;
    let input: Vec<f32> = (0..samples)
        .map(|index| index as f32 * scale / (samples - 1) as f32)
        .collect();
    let settings = RetinexSettings::new(vec![1.0, 3.0], 0.01, 0.01, scale);

    let output = reference_retinex(&input, width, height, &settings);

    let smallest = output.iter().copied().fold(f32::INFINITY, f32::min);
    let largest = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    assert_eq!(smallest, 0.0);
    assert_eq!(largest, scale);
    for value in output {
        assert!((0.0..=scale).contains(&value), "{value}");
    }
}

#[test]
fn crossed_percentiles_give_zero() {
    let width = 9;
    let height = 7;
    let samples = (width * height) as usize;
    let input: Vec<f32> = (0..samples).map(|index| index as f32).collect();
    let settings = RetinexSettings::new(vec![2.0], 0.6, 0.6, 255.0);

    let output = reference_retinex(&input, width, height, &settings);

    for value in output {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, 0.0);
    }
}

/// Bin 1365 holds the value 1, from `floor(1 / 3 * 4096)`.
#[test]
fn reference_percentile_takes_the_bin_centre() {
    let values = [0.0, 1.0, 2.0, 3.0];

    let found = reference_percentile(&values, 50.0);

    let expected = (1365.0 + 0.5) * 3.0 / 4096.0;
    assert!((found - expected).abs() < 1e-6, "{found} vs {expected}");
}

#[test]
fn reference_log_response_of_a_flat_plane_is_zero_and_finite() {
    let width = 6;
    let height = 4;
    let normalised = vec![0.0f32; (width * height) as usize];
    let sigmas = vec![1.0, 3.0];

    let response = reference_log_response(&normalised, width, height, &sigmas);

    for value in response {
        assert!(value.is_finite(), "{value}");
        assert_eq!(value, 0.0);
    }
}

/// An isolated bright sample against a dark field pushes `intensity / gaussian` into the millions
/// at a sigma this wide, which overflows f32 across 8 sigmas when the response multiplies a factor
/// per sigma instead of summing its log.
#[test]
fn reference_log_response_of_wide_sigmas_on_an_isolated_sample_stays_finite() {
    let width = 512;
    let height = 512;
    let mut normalised = vec![0.0f32; (width * height) as usize];
    normalised[(width * height / 2) as usize] = 1.0;
    let sigmas = vec![1024.0; 8];

    let response = reference_log_response(&normalised, width, height, &sigmas);

    for value in &response {
        assert!(value.is_finite(), "{value}");
    }
    assert!(response.iter().any(|value| *value != 0.0));
}
