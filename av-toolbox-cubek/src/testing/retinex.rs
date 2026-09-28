use crate::histogram::BINS;
use crate::retinex::{RecursiveGaussian, RetinexSettings, direct_form_coefficients};

/// Runs the direct-form recursive Gaussian along one line in f64, forward then backward, in place.
fn direct_form_forward_backward(values: &mut [f64], gain: f64, feedback: [f64; 3]) {
    let first = values[0];
    let mut previous = [first, first, first];
    let mut forward = vec![0.0f64; values.len()];

    for (index, value) in values.iter().enumerate() {
        let filtered =
            gain * value + feedback[0] * previous[0] + feedback[1] * previous[1] + feedback[2] * previous[2];

        forward[index] = filtered;
        previous = [filtered, previous[0], previous[1]];
    }

    let last = forward[forward.len() - 1];
    let mut previous = [last, last, last];

    for index in (0..forward.len()).rev() {
        let filtered = gain * forward[index]
            + feedback[0] * previous[0]
            + feedback[1] * previous[1]
            + feedback[2] * previous[2];

        values[index] = filtered;
        previous = [filtered, previous[0], previous[1]];
    }
}

/// Runs the direct-form recursive Gaussian of `sigma` in f64, mirroring
/// [recursive_gaussian](crate::retinex::recursive_gaussian).
///
/// Each row runs forward then backward, then each column runs forward then backward over the
/// result.
pub fn reference_recursive_gaussian(input: &[f32], width: u32, height: u32, sigma: f32) -> Vec<f32> {
    let (gain, f1, f2, f3) = direct_form_coefficients(sigma);
    let feedback = [f1, f2, f3];

    let mut output: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();

    for row in 0..height {
        let start = (row * width) as usize;
        let end = start + width as usize;

        direct_form_forward_backward(&mut output[start..end], gain, feedback);
    }

    let mut column_values = vec![0.0f64; height as usize];
    for column in 0..width {
        for row in 0..height {
            column_values[row as usize] = output[(row * width + column) as usize];
        }

        direct_form_forward_backward(&mut column_values, gain, feedback);

        for row in 0..height {
            output[(row * width + column) as usize] = column_values[row as usize];
        }
    }

    output.iter().map(|value| *value as f32).collect()
}

/// Runs the cascade sections forward then backward along one line in f32, in place, mirroring the
/// kernel's per-row and per-column passes.
fn cascade_forward_backward(values: &mut [f32], filter: RecursiveGaussian) {
    let mut forward_first_previous = values[0];
    let mut first_stage = vec![0.0f32; values.len()];

    for (index, value) in values.iter().enumerate() {
        let filtered = filter.real_gain * value + filter.real_pole * forward_first_previous;

        first_stage[index] = filtered;
        forward_first_previous = filtered;
    }

    let forward_seed = first_stage[0];
    let mut forward_second_previous_1 = forward_seed;
    let mut forward_second_previous_2 = forward_seed;
    let mut forward = vec![0.0f32; values.len()];

    for (index, value) in first_stage.iter().enumerate() {
        let filtered = filter.pair_gain * value
            + filter.pair[0] * forward_second_previous_1
            + filter.pair[1] * forward_second_previous_2;

        forward[index] = filtered;
        forward_second_previous_2 = forward_second_previous_1;
        forward_second_previous_1 = filtered;
    }

    let backward_seed = forward[forward.len() - 1];
    let mut backward_first_previous = backward_seed;
    let mut backward_first_stage = vec![0.0f32; values.len()];

    for index in (0..forward.len()).rev() {
        let filtered = filter.real_gain * forward[index] + filter.real_pole * backward_first_previous;

        backward_first_stage[index] = filtered;
        backward_first_previous = filtered;
    }

    let backward_second_seed = backward_first_stage[backward_first_stage.len() - 1];
    let mut backward_second_previous_1 = backward_second_seed;
    let mut backward_second_previous_2 = backward_second_seed;

    for index in (0..values.len()).rev() {
        let filtered = filter.pair_gain * backward_first_stage[index]
            + filter.pair[0] * backward_second_previous_1
            + filter.pair[1] * backward_second_previous_2;

        values[index] = filtered;
        backward_second_previous_2 = backward_second_previous_1;
        backward_second_previous_1 = filtered;
    }
}

/// Runs the device's cascade-form recursive Gaussian on the CPU in f32, mirroring
/// [recursive_gaussian](crate::retinex::recursive_gaussian) exactly, section by section and pass
/// by pass.
pub fn reference_recursive_gaussian_cascade(
    input: &[f32],
    width: u32,
    height: u32,
    filter: RecursiveGaussian,
) -> Vec<f32> {
    let mut output = input.to_vec();

    for row in 0..height {
        let start = (row * width) as usize;
        let end = start + width as usize;

        cascade_forward_backward(&mut output[start..end], filter);
    }

    let mut column_values = vec![0.0f32; height as usize];
    for column in 0..width {
        for row in 0..height {
            column_values[row as usize] = output[(row * width + column) as usize];
        }

        cascade_forward_backward(&mut column_values, filter);

        for row in 0..height {
            output[(row * width + column) as usize] = column_values[row as usize];
        }
    }

    output
}

/// The value `percentile` percent of the way through `values` on the CPU, mirroring
/// [percentile](crate::histogram::percentile) bin for bin.
///
/// The values are split into [BINS](crate::histogram::BINS) bins between their extremes, and the
/// result is the centre of the first bin whose running count reaches `percentile / 100 * len`, or
/// the largest value when none does.
pub fn reference_percentile(values: &[f32], percentile: f32) -> f32 {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = high - low;

    let mut bins = vec![0u32; BINS as usize];
    for value in values {
        let mut bin = 0usize;
        if span > 0.0 {
            let scaled = (value - low) / span * BINS as f32;
            bin = (scaled.max(0.0) as usize).min(BINS as usize - 1);
        }

        bins[bin] += 1;
    }

    let target = percentile / 100.0 * values.len() as f32;
    let bin_width = span / BINS as f32;

    let mut running = 0u32;
    for (bin, count) in bins.iter().enumerate() {
        running += count;
        if running as f32 >= target {
            return low + (bin as f32 + 0.5) * bin_width;
        }
    }

    high
}

/// The multi-scale log response of a normalised plane on the CPU, mirroring the retinex's `fill`,
/// `accumulate` and `finish` kernels.
///
/// The response starts at 1, gathers a factor from each of `sigmas` and finishes as its log
/// divided by the sigma count. A sigma whose blur reaches 0 at a sample contributes a factor of 1
/// there rather than dividing by 0.
pub fn reference_log_response(normalised: &[f32], width: u32, height: u32, sigmas: &[f32]) -> Vec<f32> {
    let mut msr = vec![1.0f32; normalised.len()];
    for sigma in sigmas {
        let blurred = reference_recursive_gaussian(normalised, width, height, *sigma);

        for index in 0..msr.len() {
            let intensity = normalised[index];
            let gaussian = blurred[index];
            let factor = if gaussian > 0.0 {
                intensity / gaussian + 1.0
            } else {
                1.0
            };

            msr[index] *= factor;
        }
    }

    let count = sigmas.len() as f32;
    msr.iter().map(|value| value.ln() / count).collect()
}

/// Runs the multi-scale retinex on the CPU, mirroring [retinex](crate::retinex::retinex) in the
/// kernels' order.
pub fn reference_retinex(input: &[f32], width: u32, height: u32, settings: &RetinexSettings) -> Vec<f32> {
    let low = input.iter().copied().fold(f32::INFINITY, f32::min);
    let high = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let normalised: Vec<f32> = input
        .iter()
        .map(|value| {
            if high > low {
                (value - low) / (high - low)
            } else {
                0.0
            }
        })
        .collect();

    let finished = reference_log_response(&normalised, width, height, &settings.sigmas);

    let lower_percentile = settings.lower * 100.0;
    let upper_percentile = (1.0 - settings.upper) * 100.0;
    let msr_low = reference_percentile(&finished, lower_percentile);
    let msr_high = reference_percentile(&finished, upper_percentile);

    finished
        .iter()
        .map(|value| {
            if msr_high > msr_low {
                let fraction = (value - msr_low) / (msr_high - msr_low);
                fraction.clamp(0.0, 1.0) * settings.scale
            } else {
                0.0
            }
        })
        .collect()
}
