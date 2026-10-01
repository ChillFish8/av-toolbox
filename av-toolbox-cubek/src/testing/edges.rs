use super::neighbourhood::read_reflected;
use crate::edges::{EdgeLimits, Operator};

/// The 8 neighbours of `x`, `y`, clockwise from the top left.
fn ring(input: &[f32], width: u32, height: u32, x: i32, y: i32) -> [f32; 8] {
    let top_left = read_reflected(input, width, height, x - 1, y - 1);
    let top = read_reflected(input, width, height, x, y - 1);
    let top_right = read_reflected(input, width, height, x + 1, y - 1);
    let right = read_reflected(input, width, height, x + 1, y);
    let bottom_right = read_reflected(input, width, height, x + 1, y + 1);
    let bottom = read_reflected(input, width, height, x, y + 1);
    let bottom_left = read_reflected(input, width, height, x - 1, y + 1);
    let left = read_reflected(input, width, height, x - 1, y);

    [
        top_left,
        top,
        top_right,
        right,
        bottom_right,
        bottom,
        bottom_left,
        left,
    ]
}

/// The Prewitt gradients across and down the window around `x`, `y`.
pub(super) fn prewitt_gradients(input: &[f32], width: u32, height: u32, x: i32, y: i32) -> (f32, f32) {
    let top_left = read_reflected(input, width, height, x - 1, y - 1);
    let top = read_reflected(input, width, height, x, y - 1);
    let top_right = read_reflected(input, width, height, x + 1, y - 1);
    let left = read_reflected(input, width, height, x - 1, y);
    let right = read_reflected(input, width, height, x + 1, y);
    let bottom_left = read_reflected(input, width, height, x - 1, y + 1);
    let bottom = read_reflected(input, width, height, x, y + 1);
    let bottom_right = read_reflected(input, width, height, x + 1, y + 1);

    let gx = top_left + left + bottom_left - top_right - right - bottom_right;
    let gy = top_left + top + top_right - bottom_left - bottom - bottom_right;

    (gx, gy)
}

fn prewitt_magnitude(input: &[f32], width: u32, height: u32, x: i32, y: i32) -> f32 {
    let (gx, gy) = prewitt_gradients(input, width, height, x, y);

    (gx * gx + gy * gy).sqrt()
}

fn sobel_magnitude(input: &[f32], width: u32, height: u32, x: i32, y: i32) -> f32 {
    let top_left = read_reflected(input, width, height, x - 1, y - 1);
    let top = read_reflected(input, width, height, x, y - 1);
    let top_right = read_reflected(input, width, height, x + 1, y - 1);
    let left = read_reflected(input, width, height, x - 1, y);
    let right = read_reflected(input, width, height, x + 1, y);
    let bottom_left = read_reflected(input, width, height, x - 1, y + 1);
    let bottom = read_reflected(input, width, height, x, y + 1);
    let bottom_right = read_reflected(input, width, height, x + 1, y + 1);

    let gx = top_left + 2.0 * left + bottom_left - top_right - 2.0 * right - bottom_right;
    let gy = top_left + 2.0 * top + top_right - bottom_left - 2.0 * bottom - bottom_right;

    (gx * gx + gy * gy).sqrt()
}

/// Kirsch's compass magnitude, the largest `|8 * run - 3 * ring|` over the 8 runs of 3 that walk the ring.
fn kirsch_magnitude(input: &[f32], width: u32, height: u32, x: i32, y: i32) -> f32 {
    let neighbours = ring(input, width, height, x, y);
    let total: f32 = neighbours.iter().sum();

    let mut strongest = 0.0f32;
    for start in 0..8 {
        let first = neighbours[start];
        let second = neighbours[(start + 1) % 8];
        let third = neighbours[(start + 2) % 8];
        let run = first + second + third;
        let value = 8.0 * run - 3.0 * total;
        let size = value.abs();

        strongest = strongest.max(size);
    }

    strongest
}

fn limit(magnitude: f32, limits: EdgeLimits) -> f32 {
    let scaled = magnitude * limits.multi;
    let limited = if scaled < limits.low {
        0.0
    } else if scaled > limits.high {
        limits.peak
    } else {
        scaled
    };

    limited.clamp(0.0, limits.peak)
}

/// Writes the edge strength of each sample on the CPU, mirroring [edges](crate::edges::edges).
pub fn reference_edges(
    input: &[f32],
    width: u32,
    height: u32,
    operator: Operator,
    limits: EdgeLimits,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let magnitude = match operator {
                Operator::Prewitt => prewitt_magnitude(input, width, height, x, y),
                Operator::Sobel => sobel_magnitude(input, width, height, x, y),
                Operator::Kirsch => kirsch_magnitude(input, width, height, x, y),
            };

            let limited = limit(magnitude, limits);
            output.push(limited);
        }
    }

    output
}

/// Writes `peak` where a sample is at least `threshold` on the CPU, mirroring
/// [binarize](crate::edges::binarize).
pub fn reference_binarize(input: &[f32], threshold: f32, peak: f32) -> Vec<f32> {
    input
        .iter()
        .map(|value| if *value < threshold { 0.0 } else { peak })
        .collect()
}

/// Writes `min(first + second, peak)` on the CPU, mirroring [add_capped](crate::edges::add_capped).
pub fn reference_add_capped(first: &[f32], second: &[f32], peak: f32) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .map(|(first_value, second_value)| (first_value + second_value).min(peak))
        .collect()
}
