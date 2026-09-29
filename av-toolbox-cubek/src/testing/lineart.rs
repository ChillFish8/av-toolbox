use super::{reference_binarize, reference_chain, reference_edges, reference_padded_warp, reflect};
use crate::edges::{EdgeLimits, Operator};
use crate::lineart::{DarkenSettings, ThinSettings, ToonSettings};
use crate::neighbourhood::{Extreme, Shape, Spread, Step};

/// Merges on the CPU in f64, mirroring [merge](crate::lineart::merge).
pub fn reference_merge(input: &[f32], filtered: &[f32], weight: &[f32], weight_peak: f32) -> Vec<f32> {
    let peak = f64::from(weight_peak);

    input
        .iter()
        .zip(filtered)
        .zip(weight)
        .map(|((value, target), sample_weight)| {
            let value = f64::from(*value);
            let change = f64::from(*target) - value;
            let merged = value + change * f64::from(*sample_weight) / peak;

            merged as f32
        })
        .collect()
}

/// The vertical `1 2 1` blur on the CPU, mirroring [vertical_121](crate::lineart::vertical_121).
pub fn reference_vertical_121(input: &[f32], width: u32, height: u32) -> Vec<f32> {
    let rows = height as i32;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..rows {
        let above = reflect(row - 1, rows) as u32;
        let below = reflect(row + 1, rows) as u32;

        for column in 0..width {
            let top = f64::from(input[(above * width + column) as usize]);
            let centre = f64::from(input[(row as u32 * width + column) as usize]);
            let bottom = f64::from(input[(below * width + column) as usize]);
            let blurred = (top + 2.0 * centre + bottom) / 4.0;

            output.push(blurred as f32);
        }
    }

    output
}

/// Runs vinverse on the CPU in f64, mirroring [vinverse](crate::lineart::vinverse).
///
/// `amnt` is already scaled to the code values of `input`.
pub fn reference_vinverse(
    input: &[f32],
    width: u32,
    height: u32,
    contra_str: f32,
    scl: f32,
    amnt: Option<f32>,
) -> Vec<f32> {
    let values: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let blurred = blur_121(&values, width, height);
    let diff = subtract(&values, &blurred);
    let diff_blurred = blur_121(&diff, width, height);

    let sbr: Vec<f64> = values
        .iter()
        .zip(&diff)
        .zip(&diff_blurred)
        .map(|((value, comb), comb_blurred)| remove_comb(*value, *comb, *comb_blurred))
        .collect();
    let sbr_blurred = blur_121(&sbr, width, height);

    let contra_str = f64::from(contra_str);
    let scl = f64::from(scl);
    let limit = amnt.map_or(f64::MAX, f64::from);

    values
        .iter()
        .zip(&sbr)
        .zip(&sbr_blurred)
        .map(|((value, base), base_blurred)| {
            let contra = (base - base_blurred) * contra_str;
            let removed = value - base;
            let smaller = if contra.abs() < removed.abs() {
                contra
            } else {
                removed
            };
            let step = if contra * removed < 0.0 {
                smaller * scl
            } else {
                smaller
            };
            let result = base + step;
            let capped = result.min(value + limit).max(value - limit);

            capped as f32
        })
        .collect()
}

fn remove_comb(value: f64, comb: f64, comb_blurred: f64) -> f64 {
    let residual = comb - comb_blurred;
    if residual * comb < 0.0 {
        return value;
    }

    let removed = if residual.abs() < comb.abs() {
        residual
    } else {
        comb
    };

    value - removed
}

fn subtract(first: &[f64], second: &[f64]) -> Vec<f64> {
    first
        .iter()
        .zip(second)
        .map(|(left, right)| left - right)
        .collect()
}

fn blur_121(input: &[f64], width: u32, height: u32) -> Vec<f64> {
    let rows = height as i32;
    let mut output = Vec::with_capacity(input.len());

    for row in 0..rows {
        let above = reflect(row - 1, rows) as u32;
        let below = reflect(row + 1, rows) as u32;

        for column in 0..width {
            let top = input[(above * width + column) as usize];
            let centre = input[(row as u32 * width + column) as usize];
            let bottom = input[(below * width + column) as usize];

            output.push((top + 2.0 * centre + bottom) / 4.0);
        }
    }

    output
}

/// Darkens lines on the CPU in f64, mirroring [darken](crate::lineart::darken).
pub fn reference_darken(input: &[f32], width: u32, height: u32, settings: &DarkenSettings) -> Vec<f32> {
    let values: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let s = f64::from(settings.s);
    let limit = f64::from(settings.limit);
    let cap = f64::from(settings.cap);
    let thr = f64::from(settings.thr);
    let t = f64::from(settings.t);
    let n = f64::from(settings.n);
    let peak = f64::from(settings.peak);

    let maxed = window_extreme(&values, width, height, f64::max);
    let limited: Vec<f64> = maxed
        .iter()
        .zip(&values)
        .map(|(grown, value)| grown.min(value + limit))
        .collect();
    let closed = window_extreme(&limited, width, height, f64::min);

    let mut sharp = Vec::with_capacity(values.len());
    let mut diff = Vec::with_capacity(values.len());
    for (value, closing) in values.iter().zip(&closed) {
        let closing = closing.min(cap);
        let is_line = closing > value + thr;

        sharp.push(
            if is_line {
                value + (value - closing) * s
            } else {
                *value
            },
        );
        diff.push(if is_line { value - closing + n } else { n }.clamp(0.0, peak));
    }

    if t <= 0.0 {
        return sharp.iter().map(|value| *value as f32).collect();
    }

    let darkest = window_extreme(&diff, width, height, f64::min);
    let weights: Vec<f64> = darkest
        .iter()
        .map(|lowest| ((lowest - n) * t + peak).clamp(0.0, peak))
        .collect();
    let line_mask = window_box(&weights, width, height);

    let mut output = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let thin = maxed[index] + (diff[index] - n) * (s + 1.0);
        let merged = thin + (sharp[index] - thin) * line_mask[index] / peak;

        output.push(merged as f32);
    }

    output
}

/// The largest or smallest value in each 3x3 window, chosen by `pick`.
fn window_extreme(input: &[f64], width: u32, height: u32, pick: fn(f64, f64) -> f64) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    for row in 0..height as i32 {
        for column in 0..width as i32 {
            let mut best = input[(row as u32 * width + column as u32) as usize];

            for offset_row in -1..=1 {
                for offset_column in -1..=1 {
                    let value =
                        read_reflected(input, width, height, column + offset_column, row + offset_row);
                    best = pick(best, value);
                }
            }

            output.push(best);
        }
    }

    output
}

/// The mean of each 3x3 window.
fn window_box(input: &[f64], width: u32, height: u32) -> Vec<f64> {
    let mut output = Vec::with_capacity(input.len());

    for row in 0..height as i32 {
        for column in 0..width as i32 {
            let mut total = 0.0;

            for offset_row in -1..=1 {
                for offset_column in -1..=1 {
                    total += read_reflected(input, width, height, column + offset_column, row + offset_row);
                }
            }

            output.push(total / 9.0);
        }
    }

    output
}

fn read_reflected(input: &[f64], width: u32, height: u32, column: i32, row: i32) -> f64 {
    let x = reflect(column, width as i32) as u32;
    let y = reflect(row, height as i32) as u32;

    input[(y * width + x) as usize]
}

/// Deepens lines on the CPU in f64, mirroring [toon_difference](crate::lineart::toon_difference) and
/// [toon_finish](crate::lineart::toon_finish).
pub fn reference_toon(input: &[f32], width: u32, height: u32, settings: &ToonSettings) -> Vec<f32> {
    let values: Vec<f64> = input.iter().map(|value| f64::from(*value)).collect();
    let strength = f64::from(settings.strength);
    let l = f64::from(settings.l);
    let u = f64::from(settings.u);
    let n = f64::from(settings.n);
    let peak = f64::from(settings.peak);

    let maxed = window_extreme(&values, width, height, f64::max);
    let closed = window_extreme(&maxed, width, height, f64::min);
    let diff: Vec<f64> = closed
        .iter()
        .zip(&values)
        .map(|(closing, value)| (closing - value).min(peak - n))
        .collect();

    let shifted: Vec<f32> = diff.iter().map(|value| (value + n) as f32).collect();
    let warped = reference_padded_warp(&shifted, width, height, &settings.warp);
    let limited: Vec<f64> = diff
        .iter()
        .zip(&warped)
        .map(|(darkness, moved)| darkness.min(f64::from(*moved) - n))
        .collect();
    let widest = window_extreme(&limited, width, height, f64::max);

    let mut output = Vec::with_capacity(values.len());
    for index in 0..values.len() {
        let darkness = limited[index];
        let around = widest[index];
        let dark = if around <= l {
            0.0
        } else if around >= u {
            darkness
        } else {
            darkness * (around - l) / (u - l)
        };

        output.push((values[index] - dark * strength) as f32);
    }

    output
}

/// Warp-sharpens the lines of a plane on the CPU, mirroring the thin pass.
///
/// The warped plane is merged back over the input by a line mask. The mask is the Prewitt magnitude
/// binarized at `threshold`, grown by `radius` square maximum passes, then inflated once.
pub fn reference_thin(input: &[f32], width: u32, height: u32, settings: &ThinSettings) -> Vec<f32> {
    let warped = reference_padded_warp(input, width, height, &settings.warp);
    let limits = EdgeLimits::open(settings.peak);
    let magnitude = reference_edges(input, width, height, Operator::Prewitt, limits);
    let binary = reference_binarize(&magnitude, settings.threshold, settings.peak);

    let mut steps = vec![Step::Extreme(Extreme::Maximum, Shape::Square); settings.radius as usize];
    let inflate = Step::Spread(Spread::Inflate);
    steps.push(inflate);
    let mask = reference_chain(&binary, width, height, &steps);

    reference_merge(input, &warped, &mask, settings.peak)
}
