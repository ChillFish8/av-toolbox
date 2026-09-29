mod bilateral;
mod dehalo;
mod dering;
mod edges;
mod hysteresis;
mod lineart;
mod neighbourhood;
mod padding;
mod resize;
mod retinex;
mod rgvs;

pub use self::bilateral::{reference_bilateral, reference_range_weight};
pub use self::dehalo::{
    reference_dehalo_alpha,
    reference_edge_clean,
    reference_fine_dehalo,
    reference_merge_masked,
    reference_padded_warp,
    reference_yahr,
};
pub use self::dering::{reference_hq_dering, reference_ring_mask};
pub use self::edges::{reference_add_capped, reference_binarize, reference_edges};
pub use self::hysteresis::reference_hysteresis;
pub use self::lineart::{reference_darken, reference_merge, reference_vertical_121, reference_vinverse};
pub use self::neighbourhood::{
    reference_chain,
    reference_extreme,
    reference_median,
    reference_min_blur,
    reference_prewitt,
    reference_smooth,
    reference_spread,
};
pub use self::padding::{reference_crop, reference_pad};
pub use self::resize::{reference_resize, reference_resize_float, reference_resize_wide};
pub use self::retinex::{
    reference_log_response,
    reference_percentile,
    reference_recursive_gaussian,
    reference_recursive_gaussian_cascade,
    reference_retinex,
};
pub use self::rgvs::{reference_remove_grain, reference_repair};
use crate::deband::{DebandParams, advance, unit};
use crate::depth::{Conversion, Shift, Thresholds, convert_sample};
use crate::warp::{BlurKind, WarpParams};

/// Mirrors [reflect](crate::reflect) on the CPU.
pub fn reflect(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    if folded >= length { period - folded } else { folded }
}

/// A separable blur on the CPU with `weights` along each axis and reflect-101 borders.
pub fn reference_blur(input: &[f32], width: u32, height: u32, weights: &[f32]) -> Vec<f32> {
    let radius = (weights.len() / 2) as i32;
    let width = width as i32;
    let height = height as i32;

    let mut rows = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(column + tap as i32 - radius, width);
                total += input[(row * width + source) as usize] * weight;
            }

            rows[(row * width + column) as usize] = total;
        }
    }

    let mut output = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(row + tap as i32 - radius, height);
                total += rows[(source * width + column) as usize] * weight;
            }

            output[(row * width + column) as usize] = total;
        }
    }

    output
}

/// Converts a plane `width` samples across on the CPU, mirroring [convert](crate::depth::convert).
pub fn reference_depth(
    input: &[u16],
    width: u32,
    conversion: &Conversion,
    thresholds: &Thresholds,
    shift: Shift,
) -> Vec<u16> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let column = index as u32 % width;
            let row = index as u32 / width;
            let threshold = thresholds.at(column, row, shift);

            convert_sample(*value, conversion.multiplier, threshold, conversion.largest)
        })
        .collect()
}

/// The sample nearest to `offset_x`, `offset_y` from the centre of `column`, `row`, clamped to the
/// plane.
fn nearest(
    input: &[u16],
    width: u32,
    height: u32,
    column: u32,
    row: u32,
    offset_x: f32,
    offset_y: f32,
) -> f32 {
    let x = (column as f32 + 0.5 + offset_x).floor();
    let y = (row as f32 + 0.5 + offset_y).floor();
    let cast_x = x.clamp(0.0, (width - 1) as f32) as u32;
    let cast_y = y.clamp(0.0, (height - 1) as f32) as u32;
    let clamped_x = u32::min(cast_x, width - 1);
    let clamped_y = u32::min(cast_y, height - 1);

    f32::from(input[(clamped_y * width + clamped_x) as usize])
}

/// Debands a plane on the CPU, mirroring [deband](crate::deband::deband).
pub fn reference_deband(
    input: &[u16],
    width: u32,
    height: u32,
    params: &DebandParams,
    salt: u32,
) -> Vec<u16> {
    let threshold = params.threshold / 1000.0;
    let grain = params.grain / 1000.0;
    let neutral = params.neutral / params.peak;
    let draws = params.draws();

    let mut output = Vec::with_capacity(input.len());
    for row in 0..height {
        for column in 0..width {
            let centre = f32::from(input[(row * width + column) as usize]);
            let mut state = [column, row, salt];
            let mut result = centre / params.peak;

            for draw in 0..draws {
                advance(&mut state);
                let first = unit(state[0]);
                let second = unit(state[1]);

                if draw < params.iterations {
                    let step = (draw + 1) as f32;
                    let distance = first * step * params.radius;
                    let angle = second * std::f32::consts::TAU;
                    let offset_x = distance * angle.cos();
                    let offset_y = distance * angle.sin();

                    let total = nearest(input, width, height, column, row, offset_x, offset_y)
                        + nearest(input, width, height, column, row, -offset_x, offset_y)
                        + nearest(input, width, height, column, row, -offset_x, -offset_y)
                        + nearest(input, width, height, column, row, offset_x, -offset_y);
                    let average = total / (4.0 * params.peak);
                    let bound = threshold / step;

                    if (result - average).abs() <= bound {
                        result = average;
                    }
                } else {
                    let strength = (result - neutral).abs().min(grain);
                    result += strength * (first - 0.5);
                }
            }

            let scaled = (result * params.peak + 0.5).floor();
            let clamped = scaled.clamp(0.0, params.peak);
            output.push(clamped as u16);
        }
    }

    output
}

/// The rounded mean of two values, the way aWarpSharp2 averages.
fn rounded_mean(first: i32, second: i32) -> i32 {
    (first + second + 1) >> 1
}

fn sample_at(input: &[u16], width: i32, column: i32, row: i32) -> i32 {
    let index = row * width + column;

    i32::from(input[index as usize])
}

/// Builds aWarpSharp2's edge mask on the CPU, mirroring [edge_mask](crate::warp::edge_mask).
///
/// Every plane is one size. A plane under 3 samples across or down gets an all-zero mask.
pub fn reference_edge_mask(input: &[u16], width: u32, height: u32, params: &WarpParams) -> Vec<u32> {
    if width < 3 || height < 3 {
        return vec![0; input.len()];
    }

    let peak = params.peak();
    let thresh = params.scaled_thresh();
    let columns = width as i32;
    let rows = height as i32;

    let mut mask = Vec::with_capacity(input.len());
    for row in 0..rows {
        for column in 0..columns {
            let x = column.clamp(1, columns - 2);
            let y = row.clamp(1, rows - 2);

            let top_left = sample_at(input, columns, x - 1, y - 1);
            let top = sample_at(input, columns, x, y - 1);
            let top_right = sample_at(input, columns, x + 1, y - 1);
            let left = sample_at(input, columns, x - 1, y);
            let right = sample_at(input, columns, x + 1, y);
            let bottom_left = sample_at(input, columns, x - 1, y + 1);
            let bottom = sample_at(input, columns, x, y + 1);
            let bottom_right = sample_at(input, columns, x + 1, y + 1);

            let top_corners = rounded_mean(top_left, top_right);
            let bottom_corners = rounded_mean(bottom_left, bottom_right);
            let left_corners = rounded_mean(bottom_left, top_left);
            let right_corners = rounded_mean(bottom_right, top_right);
            let above = rounded_mean(top, top_corners);
            let below = rounded_mean(bottom, bottom_corners);
            let before = rounded_mean(left, left_corners);
            let after = rounded_mean(right, right_corners);

            let vertical = (above - below).abs();
            let horizontal = (before - after).abs();
            let strongest = horizontal.max(vertical);
            let summed = (vertical + horizontal).min(peak);
            let boosted = (summed + strongest).min(peak);
            let doubled = (boosted * 2).min(peak);
            let tripled = (doubled + boosted).min(peak);
            let edge = (tripled * 2).min(peak);
            let capped = edge.min(thresh);

            mask.push(capped as u32);
        }
    }

    mask
}

/// The value `offset` steps from `position` along one line of `values`, clamped into the line.
fn along_line(values: &[u32], base: u32, step: u32, position: i32, offset: i32, length: i32) -> i32 {
    let reached = (position + offset).clamp(0, length - 1);
    let index = base + reached as u32 * step;

    values[index as usize] as i32
}

/// One r6 blur step. `start_wins` picks the start zone where the start and end zones overlap.
fn r6_step(values: &[u32], base: u32, step: u32, position: i32, length: i32, start_wins: bool) -> u32 {
    let centre = along_line(values, base, step, position, 0, length);
    let near_start = position < 6;
    let near_end = position >= length - 6;

    let direction = if near_start && (start_wins || !near_end) {
        1
    } else if near_end {
        -1
    } else {
        0
    };

    let (near, middle, far) = if direction == 0 {
        let pairs: Vec<i32> = (1..=6)
            .map(|reach| {
                let before = along_line(values, base, step, position, -reach, length);
                let after = along_line(values, base, step, position, reach, length);

                rounded_mean(before, after)
            })
            .collect();

        let near = rounded_mean(pairs[0], pairs[1]);
        let middle = rounded_mean(pairs[2], pairs[3]);
        let far = rounded_mean(pairs[4], pairs[5]);
        (near, middle, far)
    } else {
        let taps: Vec<i32> = (1..=6)
            .map(|reach| along_line(values, base, step, position, direction * reach, length))
            .collect();

        let near = rounded_mean(taps[0], taps[1]);
        let middle = rounded_mean(taps[2], taps[3]);
        let far = rounded_mean(taps[4], taps[5]);
        (near, middle, far)
    };

    let with_centre = rounded_mean(centre, near);
    let outer = rounded_mean(middle, far);
    let everything = rounded_mean(with_centre, outer);
    let blurred = rounded_mean(with_centre, everything);

    blurred as u32
}

/// One r2 blur step.
fn r2_step(values: &[u32], base: u32, step: u32, position: i32, length: i32) -> u32 {
    let centre = along_line(values, base, step, position, 0, length);
    let before_one = along_line(values, base, step, position, -1, length);
    let after_one = along_line(values, base, step, position, 1, length);
    let before_two = along_line(values, base, step, position, -2, length);
    let after_two = along_line(values, base, step, position, 2, length);

    let inner = rounded_mean(before_one, after_one);
    let outer = rounded_mean(before_two, after_two);
    let first = rounded_mean(outer, centre);
    let second = rounded_mean(first, centre);
    let blurred = rounded_mean(second, inner);

    blurred as u32
}

fn blur_step(
    values: &[u32],
    base: u32,
    step: u32,
    position: i32,
    length: i32,
    kind: BlurKind,
    start_wins: bool,
) -> u32 {
    match kind {
        BlurKind::R6 => r6_step(values, base, step, position, length, start_wins),
        BlurKind::R2 => r2_step(values, base, step, position, length),
    }
}

/// Blurs an edge mask once on the CPU, rows then columns, mirroring [blur_mask](crate::warp::blur_mask).
///
/// Across a row the end zone wins where both r6 zones reach, and down a column the start zone wins, as the
/// plugin's loop order has it.
pub fn reference_blur_mask(mask: &[u32], width: u32, height: u32, kind: BlurKind) -> Vec<u32> {
    let columns = width as i32;
    let rows = height as i32;

    let mut across = vec![0u32; mask.len()];
    for row in 0..height {
        let base = row * width;
        for column in 0..width {
            let blurred = blur_step(mask, base, 1, column as i32, columns, kind, false);
            across[(base + column) as usize] = blurred;
        }
    }

    let mut output = vec![0u32; mask.len()];
    for row in 0..height {
        for column in 0..width {
            let blurred = blur_step(&across, column, width, row as i32, rows, kind, true);
            output[(row * width + column) as usize] = blurred;
        }
    }

    output
}

fn mask_at(mask: &[u32], width: i32, column: i32, row: i32) -> i32 {
    let index = row * width + column;

    mask[index as usize] as i32
}

/// Moves each sample along the mask's slope on the CPU, mirroring [warp_plane](crate::warp::warp_plane).
pub fn reference_warp(input: &[u16], mask: &[u32], width: u32, height: u32, params: &WarpParams) -> Vec<u16> {
    let columns = width as i32;
    let rows = height as i32;
    let last_column = columns - 1;
    let last_row = rows - 1;
    let extra_bits = params.bits as i32 - 8;
    let scaled_depth = params.depth << 8;
    let peak = params.peak();

    let mut output = Vec::with_capacity(input.len());
    for y in 0..rows {
        for x in 0..columns {
            let row_above = (y - 1).max(0);
            let row_below = (y + 1).min(last_row);
            let column_before = (x - 1).max(0);
            let column_after = (x + 1).min(last_column);
            let above = mask_at(mask, columns, x, row_above);
            let below = mask_at(mask, columns, x, row_below);
            let before = mask_at(mask, columns, column_before, y);
            let after = mask_at(mask, columns, column_after, y);

            let horizontal_shifted = (before - after) >> extra_bits;
            let vertical_shifted = (above - below) >> extra_bits;
            let horizontal_fixed = ((horizontal_shifted << 7) * scaled_depth) >> 16;
            let vertical_fixed = ((vertical_shifted << 7) * scaled_depth) >> 16;

            let lowest = -y * 128;
            let highest = (rows - y) * 128 - 129;
            let vertical_limited = vertical_fixed.max(lowest).min(highest);

            let fraction_across = horizontal_fixed & 127;
            let fraction_down = vertical_limited & 127;
            let column_unclamped = x + (horizontal_fixed >> 7);
            let row_unclamped = y + (vertical_limited >> 7);

            let weight_across = if column_unclamped >= last_column || column_unclamped < 0 {
                0
            } else {
                fraction_across
            };

            let column = column_unclamped.clamp(0, last_column);
            let next_column = (column + 1).min(last_column);
            let row = row_unclamped.clamp(0, last_row);
            let next_row = (row_unclamped + 1).clamp(0, last_row);

            let top_left = sample_at(input, columns, column, row);
            let top_right = sample_at(input, columns, next_column, row);
            let bottom_left = sample_at(input, columns, column, next_row);
            let bottom_right = sample_at(input, columns, next_column, next_row);

            let top_mix = (top_left * (128 - weight_across) + top_right * weight_across + 64) >> 7;
            let bottom_mix = (bottom_left * (128 - weight_across) + bottom_right * weight_across + 64) >> 7;
            let blended = (top_mix * (128 - fraction_down) + bottom_mix * fraction_down + 64) >> 7;

            let clamped = blended.clamp(0, peak);
            output.push(clamped as u16);
        }
    }

    output
}

/// The whole aWarpSharp2 warp on the CPU, mirroring [warp_sharp](crate::warp::warp_sharp).
pub fn reference_warp_sharp(input: &[u16], width: u32, height: u32, params: &WarpParams) -> Vec<u16> {
    let mut mask = reference_edge_mask(input, width, height, params);
    for _ in 0..params.passes {
        mask = reference_blur_mask(&mask, width, height, params.blur);
    }

    reference_warp(input, &mask, width, height, params)
}

/// Rounds f32 code values on the CPU, mirroring [narrow](crate::values::narrow).
pub fn reference_narrow(values: &[f32], peak: f32) -> Vec<u16> {
    values
        .iter()
        .map(|value| {
            let rounded = (value + 0.5).floor();
            rounded.clamp(0.0, peak) as u16
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrowing_rounds_half_up_and_clamps() {
        let values = [-3.0, 0.49, 0.5, 1.5, 254.5, 300.0];

        let narrowed = reference_narrow(&values, 255.0);

        assert_eq!(narrowed, vec![0, 0, 1, 2, 255, 255]);
    }
}
