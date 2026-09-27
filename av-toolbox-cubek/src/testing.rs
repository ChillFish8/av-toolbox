use crate::deband::{DebandParams, advance, unit};
use crate::depth::{Conversion, Shift, Thresholds, convert_sample};
use crate::resize::Taps;

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

/// Applies `across` then `down` on the CPU, rounding to U8 at the end, mirroring
/// [apply](crate::resize::apply).
pub fn reference_resize(input: &[u8], width: u32, height: u32, across: &Taps, down: &Taps) -> Vec<u8> {
    let wide: Vec<u16> = input.iter().map(|sample| u16::from(*sample)).collect();
    let resized = reference_resize_wide(&wide, width, height, across, down, u16::from(u8::MAX));

    resized.iter().map(|sample| *sample as u8).collect()
}

/// Applies `across` then `down` on the CPU, rounding and clamping to `0..=largest` at the end.
pub fn reference_resize_wide(
    input: &[u16],
    width: u32,
    height: u32,
    across: &Taps,
    down: &Taps,
    largest: u16,
) -> Vec<u16> {
    let out_width = across.starts.len() as u32;
    let out_height = down.starts.len() as u32;

    let mut rows = vec![0.0f32; (out_width * height) as usize];
    for row in 0..height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..across.count {
                let tap = across.starts[column as usize] + step as i32;
                let source = reflect(tap, width as i32) as u32;
                let weight = across.weights[(column * across.count + step) as usize];
                let sample = input[(row * width + source) as usize];
                total += f32::from(sample) * weight;
            }

            rows[(row * out_width + column) as usize] = total;
        }
    }

    let ceiling = f32::from(largest);
    let mut output = Vec::with_capacity((out_width * out_height) as usize);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..down.count {
                let tap = down.starts[row as usize] + step as i32;
                let source = reflect(tap, height as i32) as u32;
                let weight = down.weights[(row * down.count + step) as usize];
                total += rows[(source * out_width + column) as usize] * weight;
            }

            let rounded = total.round().clamp(0.0, ceiling) as u16;
            output.push(rounded);
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
    let clamped_x = x.clamp(0.0, (width - 1) as f32) as u32;
    let clamped_y = y.clamp(0.0, (height - 1) as f32) as u32;

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
            output.push(scaled.clamp(0.0, params.peak) as u16);
        }
    }

    output
}
