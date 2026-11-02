// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The number of sample offsets `sigma_s` calls for, and the gap between them, following vszip's
/// truncated bilateral.
fn sampling(sigma_s: f32) -> (u32, u32) {
    let scaled = 2.0 * sigma_s + 0.5;
    let truncated = scaled.trunc() as i64;
    let reach = i64::max(truncated, 1);
    let step: i64 = if reach < 4 {
        1
    } else if reach < 8 {
        2
    } else {
        3
    };

    let mut count: i64 = 1;
    let mut radius: i64 = 1;
    while reach * 2 > radius * 3 {
        count += 1;
        radius = 1 + (count - 1) * step;
    }

    if radius >= reach && count > 2 {
        count -= 1;
    }

    (count as u32, step as u32)
}

/// The sample offsets vszip's truncated bilateral reads for `sigma_s`, from 1 outward.
pub fn positions(sigma_s: f32) -> Vec<u32> {
    let (count, step) = sampling(sigma_s);

    (0..count).map(|index| 1 + index * step).collect()
}

/// The difference past which the range weight stops falling, from `sigma_r` and `peak`.
fn range_cutoff(sigma_r: f32, peak: f32) -> f32 {
    let scaled = sigma_r * 8.0 * peak + 0.5;
    let truncated = scaled.trunc();

    truncated.min(peak)
}

#[cube]
fn clamped_index(width: u32, height: u32, column: i32, row: i32) -> u32 {
    let last_column = width as i32 - 1;
    let last_row = height as i32 - 1;
    let x = i32::clamp(column, 0, last_column);
    let y = i32::clamp(row, 0, last_row);
    let row_start = y as u32 * width;

    row_start + x as u32
}

/// `exp(-((min(diff, cutoff) / peak) / sigma_r)^2 / 2)`, vszip's range weight past a cutoff.
#[cube]
fn range_weight(diff: f32, cutoff: f32, sigma_r: f32, peak: f32) -> f32 {
    let limited = f32::min(diff, cutoff);
    let normalised = limited / peak;
    let scaled = normalised / sigma_r;
    let exponent = -(scaled * scaled) / 2.0;

    f32::exp(exponent)
}

#[cube(launch_unchecked)]
fn bilateral_kernel(
    input: &[f32],
    reference: &[f32],
    output: &mut [f32],
    width: u32,
    height: u32,
    count: u32,
    step: u32,
    sigma_s: f32,
    sigma_r: f32,
    cutoff: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;

    let centre_reference = reference[ABSOLUTE_POS];
    let centre_input = input[ABSOLUTE_POS];

    let mut weight_sum = 1.0f32;
    let mut value_sum = centre_input;
    let spatial_denominator = 2.0 * sigma_s * sigma_s;

    for row_index in 0..count {
        let offset_y = 1i32 + (row_index * step) as i32;

        for column_index in 0..count {
            let offset_x = 1i32 + (column_index * step) as i32;

            let squared_offset = offset_x * offset_x + offset_y * offset_y;
            let squared_offset_f32 = squared_offset as f32;
            let spatial_exponent = -squared_offset_f32 / spatial_denominator;
            let spatial = f32::exp(spatial_exponent);

            for quadrant in 0..4u32 {
                let sign_x = (quadrant % 2u32) as i32 * 2 - 1;
                let sign_y = (quadrant / 2u32) as i32 * 2 - 1;
                let sample_x = x + sign_x * offset_x;
                let sample_y = y + sign_y * offset_y;

                let sample_index = clamped_index(width, height, sample_x, sample_y);
                let sample_input = input[sample_index as usize];
                let sample_reference = reference[sample_index as usize];
                let difference = sample_reference - centre_reference;
                let distance = f32::abs(difference);
                let range = range_weight(distance, cutoff, sigma_r, peak);
                let weight = spatial * range;

                weight_sum += weight;
                value_sum += weight * sample_input;
            }
        }
    }

    output[ABSOLUTE_POS] = value_sum / weight_sum;
}

/// The planes one bilateral pass reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct BilateralPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub reference: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The spatial and range spread of one bilateral pass, and the code value scale it runs at.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BilateralSettings {
    pub sigma_s: f32,
    pub sigma_r: f32,
    pub peak: f32,
}

/// Averages `planes.input`, weighing each sample by its distance and by how far `planes.reference`
/// differs from the centre's, following vszip's truncated bilateral.
pub fn bilateral(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: BilateralPlanes<'_>,
    settings: BilateralSettings,
) {
    let (count, step) = sampling(settings.sigma_s);
    let cutoff = range_cutoff(settings.sigma_r, settings.peak);
    let width = planes.output.width();
    let height = planes.output.height();
    let samples = planes.output.samples();
    let input = planes.input.array();
    let reference = planes.reference.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values, and every read clamps into the plane.
    unsafe {
        bilateral_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            reference,
            output,
            width,
            height,
            count,
            step,
            settings.sigma_s,
            settings.sigma_r,
            cutoff,
            settings.peak,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{reference_bilateral, reference_range_weight};

    #[test]
    fn positions_follow_vszips_sampling() {
        assert_eq!(positions(1.0), vec![1, 2]);
        assert_eq!(positions(2.0), vec![1, 3]);
        assert_eq!(positions(5.0), vec![1, 4, 7]);
        assert_eq!(positions(10.0), vec![1, 4, 7, 10, 13, 16]);
        assert_eq!(positions(16.0), vec![1, 4, 7, 10, 13, 16, 19, 22]);
    }

    #[test]
    fn a_flat_plane_is_unchanged() {
        let width = 9;
        let height = 9;
        let plane = vec![100.0; (width * height) as usize];
        let reference = plane.clone();
        let settings = BilateralSettings {
            sigma_s: 2.0,
            sigma_r: 0.02,
            peak: 255.0,
        };

        let output = reference_bilateral(&plane, &reference, width, height, settings);

        for value in output {
            assert!((value - 100.0).abs() < 1e-4, "{value} was not 100");
        }
    }

    #[test]
    fn the_centre_row_and_column_are_not_sampled() {
        let width = 15;
        let height = 15;
        let mut plane = vec![0.0f32; (width * height) as usize];
        let bright_index = 8 * width + 7;
        plane[bright_index as usize] = 100.0;
        let reference = vec![0.0f32; (width * height) as usize];
        let settings = BilateralSettings {
            sigma_s: 1.0,
            sigma_r: 0.02,
            peak: 255.0,
        };

        let output = reference_bilateral(&plane, &reference, width, height, settings);
        let centre_index = 7 * width + 7;

        assert_eq!(output[centre_index as usize], 0.0);
    }

    /// At `(7, 7)`, only the offset `(+1, +1)` reaches the 100 at `(8, 8)`. With `sigma_s = 1`,
    /// that offset's spatial weight is `exp(-(1^2 + 1^2) / (2 * 1^2)) = exp(-1)`. The weight sum
    /// adds the centre's 1 to the four quadrant reads at each of positions 1 and 2 in both axes,
    /// `1 + 4 * (exp(-1) + 2 * exp(-2.5) + exp(-4))`, from `(1, 1)`, `(1, 2)` and `(2, 1)` (equal
    /// to each other), and `(2, 2)`. The reference is flat, so every range weight is 1, and the
    /// expected value is `100 * exp(-1)` over that sum.
    #[test]
    fn the_spatial_weight_reaches_a_diagonal_neighbour() {
        let width = 15;
        let height = 15;
        let mut plane = vec![0.0f32; (width * height) as usize];
        let bright_index = 8 * width + 8;
        plane[bright_index as usize] = 100.0;
        let reference = vec![0.0f32; (width * height) as usize];
        let settings = BilateralSettings {
            sigma_s: 1.0,
            sigma_r: 0.02,
            peak: 255.0,
        };

        let output = reference_bilateral(&plane, &reference, width, height, settings);
        let centre_index = 7 * width + 7;

        let one_one = (-1.0f64).exp();
        let one_two_or_two_one = (-2.5f64).exp();
        let two_two = (-4.0f64).exp();
        let weight_sum = 1.0 + 4.0 * (one_one + 2.0 * one_two_or_two_one + two_two);
        let expected = 100.0 * one_one / weight_sum;

        let actual = f64::from(output[centre_index as usize]);
        assert!((actual - expected).abs() < 1e-4, "{actual} was not {expected}");
    }

    /// `cutoff` is `trunc(0.02 * 8 * 255 + 0.5) = 41`, so a difference of `peak` and one of
    /// `0.5 * peak` both clip to it and weigh the same.
    #[test]
    fn the_range_weight_holds_past_its_cutoff() {
        let peak = 255.0;
        let sigma_r = 0.02;
        let cutoff = range_cutoff(sigma_r, peak);

        assert_eq!(cutoff, 41.0);

        let full = reference_range_weight(peak as f64, cutoff as f64, sigma_r as f64, peak as f64);
        let half = reference_range_weight(0.5 * peak as f64, cutoff as f64, sigma_r as f64, peak as f64);

        assert_eq!(full, half);
    }

    #[test]
    fn an_edge_in_the_reference_is_kept() {
        let width = 16;
        let height = 16;
        let mut reference = vec![0.0f32; (width * height) as usize];
        for row in 0..height {
            for column in 8..width {
                reference[(row * width + column) as usize] = 255.0;
            }
        }

        let input = reference.clone();
        let settings = BilateralSettings {
            sigma_s: 2.0,
            sigma_r: 0.02,
            peak: 255.0,
        };

        let output = reference_bilateral(&input, &reference, width, height, settings);
        let row = 8;
        let before_index = row * width + 7;
        let after_index = row * width + 8;

        assert!(output[before_index as usize] < 0.01 * 255.0);
        assert!((output[after_index as usize] - 255.0).abs() < 0.01 * 255.0);
    }
}
