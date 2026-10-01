use crate::bilateral::{BilateralSettings, positions};

/// The index of `column`, `row` clamped into a `width` by `height` plane.
pub(super) fn clamped_index(width: u32, height: u32, column: i32, row: i32) -> usize {
    let last_column = width as i32 - 1;
    let last_row = height as i32 - 1;
    let x = column.clamp(0, last_column);
    let y = row.clamp(0, last_row);
    let row_start = y as u32 * width;

    (row_start + x as u32) as usize
}

/// `exp(-((min(distance, cutoff) / peak) / sigma_r)^2 / 2)` on the CPU in f64, mirroring the range
/// weight [bilateral](crate::bilateral::bilateral) applies on the device.
pub fn reference_range_weight(distance: f64, cutoff: f64, sigma_r: f64, peak: f64) -> f64 {
    let limited = distance.min(cutoff);
    let normalised = limited / peak;
    let scaled = normalised / sigma_r;
    let exponent = -(scaled * scaled) / 2.0;

    exponent.exp()
}

/// Averages `input` on the CPU in f64, weighing each sample by its distance and by how far
/// `reference` differs from the centre's, mirroring [bilateral](crate::bilateral::bilateral).
pub fn reference_bilateral(
    input: &[f32],
    reference: &[f32],
    width: u32,
    height: u32,
    settings: BilateralSettings,
) -> Vec<f32> {
    let sample_positions = positions(settings.sigma_s);
    let sigma_s = f64::from(settings.sigma_s);
    let sigma_r = f64::from(settings.sigma_r);
    let peak = f64::from(settings.peak);
    let cutoff_raw = (sigma_r * 8.0 * peak + 0.5).trunc();
    let cutoff = cutoff_raw.min(peak);

    let mut output = Vec::with_capacity(input.len());

    for row in 0..height as i32 {
        for column in 0..width as i32 {
            let centre_index = clamped_index(width, height, column, row);
            let centre_reference = f64::from(reference[centre_index]);
            let centre_input = f64::from(input[centre_index]);

            let mut weight_sum = 1.0f64;
            let mut value_sum = centre_input;

            for &offset_y in &sample_positions {
                for &offset_x in &sample_positions {
                    let squared_offset = f64::from(offset_x * offset_x + offset_y * offset_y);
                    let spatial_denominator = 2.0 * sigma_s * sigma_s;
                    let spatial_exponent = -squared_offset / spatial_denominator;
                    let spatial = spatial_exponent.exp();

                    let signed_x = offset_x as i32;
                    let signed_y = offset_y as i32;
                    let quadrant_offsets = [
                        (-signed_x, -signed_y),
                        (signed_x, -signed_y),
                        (-signed_x, signed_y),
                        (signed_x, signed_y),
                    ];

                    for (offset_across, offset_down) in quadrant_offsets {
                        let sample_column = column + offset_across;
                        let sample_row = row + offset_down;
                        let sample_index = clamped_index(width, height, sample_column, sample_row);
                        let sample_input = f64::from(input[sample_index]);
                        let sample_reference = f64::from(reference[sample_index]);
                        let difference = sample_reference - centre_reference;
                        let distance = difference.abs();
                        let range = reference_range_weight(distance, cutoff, sigma_r, peak);
                        let weight = spatial * range;

                        weight_sum += weight;
                        value_sum += weight * sample_input;
                    }
                }
            }

            let averaged = value_sum / weight_sum;
            output.push(averaged as f32);
        }
    }

    output
}
