use super::edges::prewitt_gradients;
use super::resize::reference_resize_float;
use crate::neighbourhood::median_of_three_sample;
use crate::resize::{Kernel, centred_taps};

/// Binarizes the Prewitt gradients on the CPU in 64-bit integers, mirroring
/// [binarize_prewitt](crate::aa::binarize_prewitt).
pub fn reference_binarize_prewitt(
    input: &[f32],
    width: u32,
    height: u32,
    threshold: u32,
    peak: f32,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let (across, down) = prewitt_gradients(input, width, height, x, y);
            let gx = across as i64;
            let gy = down as i64;
            let sum = gx * gx + gy * gy;
            let value = if sum >= i64::from(threshold) { peak } else { 0.0 };

            output.push(value);
        }
    }

    output
}

/// The two samples a doubled sample at `doubled` reads on one axis, clamped to `length`, and the
/// first one's weight.
fn doubling_taps(doubled: u32, length: u32) -> (usize, usize, f32) {
    let half = doubled / 2;
    let odd = doubled % 2;
    let start = (half + odd) as i32 - 1;
    let last = length as i32 - 1;
    let first = start.clamp(0, last) as usize;
    let second = (start + 1).clamp(0, last) as usize;
    let weight = if odd == 1 { 0.75 } else { 0.25 };

    (first, second, weight)
}

/// Doubles `mask` into EEDI3's 8-bit clip on the CPU, mirroring [double_mclip](crate::aa::double_mclip).
pub fn reference_double_mclip(mask: &[f32], width: u32, height: u32, mask_peak: f32) -> Vec<f32> {
    let columns = width as usize;
    let mut output = Vec::with_capacity(4 * mask.len());

    for row in 0..2 * height {
        for column in 0..2 * width {
            let (top, bottom, top_weight) = doubling_taps(row, height);
            let (left, right, left_weight) = doubling_taps(column, width);
            let right_weight = 1.0 - left_weight;
            let bottom_weight = 1.0 - top_weight;

            let upper = left_weight * mask[top * columns + left] + right_weight * mask[top * columns + right];
            let lower =
                left_weight * mask[bottom * columns + left] + right_weight * mask[bottom * columns + right];
            let value = top_weight * upper + bottom_weight * lower;
            let clipped = if 510.0 * value > mask_peak { 255.0 } else { 0.0 };

            output.push(clipped);
        }
    }

    output
}

/// Averages each 2x2 block of a plane of even width and height on the CPU, through
/// [Kernel::Box](crate::resize::Kernel::Box) taps.
pub fn reference_box_halve(input: &[f32], width: u32, height: u32) -> Vec<f32> {
    assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "a halved plane is {width}x{height}, expected even sizes",
    );

    let across = centred_taps(width, width / 2, &Kernel::Box);
    let down = centred_taps(height, height / 2, &Kernel::Box);

    reference_resize_float(input, width, height, &across, &down)
}

/// Writes `aa + keep * (luma - down)` on the CPU, mirroring [pscale_blend](crate::aa::pscale_blend).
pub fn reference_pscale_blend(aa: &[f32], luma: &[f32], down: &[f32], keep: f32) -> Vec<f32> {
    aa.iter()
        .zip(luma)
        .zip(down)
        .map(|((aa_value, luma_value), down_value)| {
            let difference = luma_value - down_value;
            aa_value + keep * difference
        })
        .collect()
}

/// Takes the median of three planes on the CPU, mirroring
/// [median_of_three](crate::neighbourhood::median_of_three).
pub fn reference_median_of_three(first: &[f32], second: &[f32], third: &[f32]) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .zip(third)
        .map(|((first_value, second_value), third_value)| {
            median_of_three_sample(*first_value, *second_value, *third_value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENTRE: usize = 12;

    /// A 5 by 5 plane whose rows hold `top`, `top`, `middle`, `bottom` and `bottom`.
    fn rows_plane(top: f32, middle: f32, bottom: f32) -> Vec<f32> {
        let levels = [top, top, middle, bottom, bottom];

        levels.iter().flat_map(|level| [*level; 5]).collect()
    }

    #[test]
    fn binarize_sets_an_exact_tie_at_8_bit() {
        let plane = rows_plane(40.0, 30.0, 20.0);
        let threshold = 60 * 60;

        let at = reference_binarize_prewitt(&plane, 5, 5, threshold, 255.0);
        let above = reference_binarize_prewitt(&plane, 5, 5, threshold + 1, 255.0);

        assert_eq!(at[CENTRE], 255.0);
        assert_eq!(above[CENTRE], 0.0);
    }

    #[test]
    fn binarize_sets_an_exact_tie_at_the_16_bit_threshold() {
        let plane = rows_plane(10240.0, 7000.0, 5120.0);
        let scaled = 60 << 8;
        let threshold = scaled * scaled;
        let next = (scaled + 1) * (scaled + 1);

        let at = reference_binarize_prewitt(&plane, 5, 5, threshold, 65535.0);
        let above = reference_binarize_prewitt(&plane, 5, 5, next, 65535.0);

        assert_eq!(at[CENTRE], 65535.0);
        assert_eq!(above[CENTRE], 0.0);
    }

    #[test]
    fn binarize_sets_a_gradient_whose_square_passes_2_to_the_32() {
        let plane = [
            65535.0, 30000.0, 0.0, //
            65535.0, 30000.0, 0.0, //
            65535.0, 30000.0, 0.0,
        ];
        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 65535.0);
    }

    #[test]
    fn binarize_sets_gradients_whose_squares_sum_past_2_to_the_32() {
        let mut plane = Vec::new();
        for row in 0..3 {
            for column in 0..3 {
                let value = 65535 - 8350 * (row + column);
                plane.push(value as f32);
            }
        }

        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 65535.0);
    }

    /// The window's gradients are 20947 and 61828, whose squares sum to 7 under the threshold. An f32
    /// square root rounds that sum onto the threshold.
    #[test]
    fn binarize_leaves_a_16_bit_sum_just_under_the_threshold_unset() {
        let plane = [
            0.0, 61828.0, 0.0, //
            20947.0, 0.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let scaled = 255 << 8;
        let threshold = scaled * scaled;

        let binarized = reference_binarize_prewitt(&plane, 3, 3, threshold, 65535.0);

        assert_eq!(binarized[4], 0.0);
    }

    /// A 5 by 5 mask holding `value` at its centre and 0 elsewhere.
    fn lone_sample(value: f32) -> Vec<f32> {
        let mut mask = vec![0.0; 25];
        mask[CENTRE] = value;

        mask
    }

    /// The doubled positions `mclip` sets.
    fn set_positions(clip: &[f32], width: usize) -> Vec<(usize, usize)> {
        clip.iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .map(|(index, _)| (index % width, index / width))
            .collect()
    }

    /// The 4 by 4 block of doubled samples that read the centre of a 5 by 5 mask.
    fn centre_block() -> Vec<(usize, usize)> {
        let mut block = Vec::new();
        for row in 3..=6 {
            for column in 3..=6 {
                block.push((column, row));
            }
        }

        block
    }

    #[test]
    fn mclip_leaves_an_exact_half_unset() {
        let mask = lone_sample(8.0);

        let clip = reference_double_mclip(&mask, 5, 5, 255.0);

        let corners = [(3, 3), (6, 3), (3, 6), (6, 6)];
        let expected: Vec<(usize, usize)> = centre_block()
            .into_iter()
            .filter(|position| !corners.contains(position))
            .collect();
        assert_eq!(set_positions(&clip, 10), expected);
    }

    #[test]
    fn mclip_sets_just_past_half() {
        let mask = lone_sample(9.0);

        let clip = reference_double_mclip(&mask, 5, 5, 255.0);

        assert_eq!(set_positions(&clip, 10), centre_block());
    }

    #[test]
    fn mclip_repeats_the_edge_sample() {
        let mut inner = vec![255.0; 25];
        let mut ring = vec![0.0; 25];
        for row in 0..5 {
            for column in 0..5 {
                let on_ring = row == 0 || row == 4 || column == 0 || column == 4;
                if on_ring {
                    inner[row * 5 + column] = 0.0;
                    ring[row * 5 + column] = 255.0;
                }
            }
        }

        let inner_clip = reference_double_mclip(&inner, 5, 5, 255.0);
        let ring_clip = reference_double_mclip(&ring, 5, 5, 255.0);

        for row in 0..10 {
            for column in 0..10 {
                let on_outer_ring = row == 0 || row == 9 || column == 0 || column == 9;
                if on_outer_ring {
                    let index = row * 10 + column;
                    assert_eq!(inner_clip[index], 0.0, "column {column}, row {row}");
                    assert_eq!(ring_clip[index], 255.0, "column {column}, row {row}");
                }
            }
        }
    }

    const AA: [f32; 3] = [10.0, 20.25, 100.5];
    const LUMA: [f32; 3] = [12.0, 30.0, 90.0];
    const DOWN: [f32; 3] = [11.0, 25.0, 95.75];

    #[test]
    fn blend_at_pscale_0_adds_the_whole_difference() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 1.0);

        assert_eq!(blended, [11.0, 25.25, 94.75]);
    }

    #[test]
    fn blend_at_pscale_half_adds_half_the_difference() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 0.5);

        assert_eq!(blended, [10.5, 22.75, 97.625]);
    }

    #[test]
    fn blend_at_pscale_1_keeps_aa() {
        let blended = reference_pscale_blend(&AA, &LUMA, &DOWN, 0.0);

        assert_eq!(blended, AA);
    }

    /// A plane holding `column + 10 * row`.
    fn ramp(width: u32, height: u32) -> Vec<f32> {
        let mut plane = Vec::new();
        for row in 0..height {
            for column in 0..width {
                let value = column + 10 * row;
                plane.push(value as f32);
            }
        }

        plane
    }

    #[test]
    fn box_halving_averages_each_block_to_an_odd_size() {
        let plane = ramp(6, 2);

        let halved = reference_box_halve(&plane, 6, 2);

        assert_eq!(halved, [5.5, 7.5, 9.5]);
    }

    #[test]
    fn box_halving_averages_each_block_to_an_even_size() {
        let plane = ramp(8, 4);

        let halved = reference_box_halve(&plane, 8, 4);

        assert_eq!(halved, [5.5, 7.5, 9.5, 11.5, 25.5, 27.5, 29.5, 31.5]);
    }

    #[test]
    fn median_of_three_picks_the_middle_value() {
        let first = [1.0, 5.0, 9.0, 4.0];
        let second = [2.0, 9.0, 1.0, 4.0];
        let third = [3.0, 1.0, 5.0, 7.0];

        let middle = reference_median_of_three(&first, &second, &third);

        assert_eq!(middle, [2.0, 5.0, 5.0, 4.0]);
    }
}
