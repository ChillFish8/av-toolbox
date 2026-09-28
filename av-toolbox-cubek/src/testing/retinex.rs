use crate::histogram::BINS;
use crate::retinex::{RecursiveGaussian, RetinexSettings, sections, settling_length};

/// Runs the recursive Gaussian's sections along `values` in f64, first-order then second-order,
/// walking backward when `backward` is set, in place.
///
/// Each section starts its history at its first input sample in the walking direction.
fn cascade_pass(values: &mut [f64], feedback: (f64, f64, f64), backward: bool) {
    let (pole, pair_a, pair_b) = feedback;
    let real_gain = 1.0 - pole;
    let pair_gain = 1.0 - pair_a - pair_b;

    if backward {
        values.reverse();
    }

    let mut first = values[0];
    for value in values.iter_mut() {
        first = real_gain * *value + pole * first;
        *value = first;
    }

    let mut last = values[0];
    let mut before_last = values[0];
    for value in values.iter_mut() {
        let filtered = pair_gain * *value + pair_a * last + pair_b * before_last;

        before_last = last;
        last = filtered;
        *value = filtered;
    }

    if backward {
        values.reverse();
    }
}

/// Runs the recursive Gaussian forward then backward along `line` in f64, in place, as if the
/// line carried on `padding` copies of its edge samples past each end.
fn padded_forward_backward(
    line: &mut [f64],
    padded: &mut Vec<f64>,
    feedback: (f64, f64, f64),
    padding: usize,
) {
    let first = line[0];
    let last = line[line.len() - 1];

    padded.clear();
    padded.resize(padding, first);
    padded.extend_from_slice(line);
    padded.resize(padded.len() + padding, last);

    cascade_pass(padded, feedback, false);
    cascade_pass(padded, feedback, true);

    let kept = &padded[padding..padding + line.len()];
    line.copy_from_slice(kept);
}

/// Runs the recursive Gaussian of `sigma` in f64 as if the plane were extended by repeating its
/// edge samples.
///
/// Each row runs forward then backward, then each column runs forward then backward over the
/// result. Each line is extended on both ends by enough copies of its edge samples for the
/// filter's response to die away before it reaches the line.
pub fn reference_recursive_gaussian(input: &[f32], width: u32, height: u32, sigma: f32) -> Vec<f32> {
    let sigma = f64::from(sigma);
    let feedback = sections(sigma);
    let padding = settling_length(sigma);

    let mut output: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let mut padded = Vec::new();

    for row in 0..height {
        let start = (row * width) as usize;
        let end = start + width as usize;

        padded_forward_backward(&mut output[start..end], &mut padded, feedback, padding);
    }

    let mut column_values = vec![0.0f64; height as usize];
    for column in 0..width {
        for row in 0..height {
            column_values[row as usize] = output[(row * width + column) as usize];
        }

        padded_forward_backward(&mut column_values, &mut padded, feedback, padding);

        for row in 0..height {
            output[(row * width + column) as usize] = column_values[row as usize];
        }
    }

    output.iter().map(|value| *value as f32).collect()
}

/// The backward pass's starting state `(h, y1, y1 - y2)` for a line whose last input is `edge`
/// and whose forward pass ended in the state `forward`, in the kernel's operation order.
fn backward_seed(filter: &RecursiveGaussian, forward: [f32; 3], edge: f32) -> [f32; 3] {
    let deviation_first = forward[0] - edge;
    let deviation_last = forward[1] - edge;
    let deviation_difference = forward[2];
    let matrix = filter.edge;

    let offset_first =
        matrix[0][0] * deviation_first + matrix[0][1] * deviation_last + matrix[0][2] * deviation_difference;
    let offset_last =
        matrix[1][0] * deviation_first + matrix[1][1] * deviation_last + matrix[1][2] * deviation_difference;
    let difference =
        matrix[2][0] * deviation_first + matrix[2][1] * deviation_last + matrix[2][2] * deviation_difference;

    [edge + offset_first, edge + offset_last, difference]
}

/// Runs the cascade sections forward then backward along one line in f32, in place, mirroring the
/// kernel's per-row and per-column passes.
///
/// Each section updates its last output by a small step, and the backward pass starts from
/// [backward_seed].
fn cascade_forward_backward(values: &mut [f32], filter: RecursiveGaussian) {
    let start = values[0];
    let edge = values[values.len() - 1];

    let mut first = start;
    for value in values.iter_mut() {
        first += filter.real_gain * (*value - first);
        *value = first;
    }

    let mut last = start;
    let mut difference = 0.0f32;
    for value in values.iter_mut() {
        let step = filter.pair_gain * (*value - last) - filter.pair[1] * difference;

        last += step;
        difference = step;
        *value = last;
    }

    let forward = [first, last, difference];
    let [seed_first, seed_last, seed_difference] = backward_seed(&filter, forward, edge);

    let mut first = seed_first;
    for value in values.iter_mut().rev() {
        first += filter.real_gain * (*value - first);
        *value = first;
    }

    let mut last = seed_last;
    let mut difference = seed_difference;
    for value in values.iter_mut().rev() {
        let step = filter.pair_gain * (*value - last) - filter.pair[1] * difference;

        last += step;
        difference = step;
        *value = last;
    }
}

/// Runs the device's recursive Gaussian on the CPU in f32, mirroring
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

/// The multi-scale log response of a normalised plane on the CPU.
///
/// Each of `sigmas` blurs the plane and contributes a factor at every sample, and the response is
/// the mean of that factor's log across `sigmas`. A sigma whose blur reaches 0 at a sample
/// contributes a factor of 1 there rather than dividing by 0. Summing the logs, rather than
/// multiplying the factors and taking one final log, keeps a wide sigma's huge factor from
/// overflowing f32 before the mean is taken.
pub fn reference_log_response(normalised: &[f32], width: u32, height: u32, sigmas: &[f32]) -> Vec<f32> {
    let mut msr = vec![0.0f32; normalised.len()];
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

            msr[index] += factor.ln();
        }
    }

    let count = sigmas.len() as f32;
    msr.iter().map(|value| value / count).collect()
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

    let finished = reference_log_response(&normalised, width, height, settings.sigmas());

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
