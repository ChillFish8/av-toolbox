use super::reflect;
use crate::neighbourhood::{Extreme, Shape, Smooth};

/// The value at `column`, `row`, read through reflection.
pub(super) fn read_reflected(input: &[f32], width: u32, height: u32, column: i32, row: i32) -> f32 {
    let x = reflect(column, width as i32);
    let y = reflect(row, height as i32);
    let index = y as u32 * width + x as u32;

    input[index as usize]
}

fn in_shape(shape: Shape, offset_x: i32, offset_y: i32) -> bool {
    match shape {
        Shape::Square => true,
        Shape::Plus => offset_x == 0 || offset_y == 0,
        Shape::Horizontal => offset_y == 0,
        Shape::Vertical => offset_x == 0,
    }
}

/// Takes the largest or smallest neighbour on the CPU, mirroring [extreme](crate::neighbourhood::extreme).
pub fn reference_extreme(input: &[f32], width: u32, height: u32, extreme: Extreme, shape: Shape) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut best = input[(y as u32 * width + x as u32) as usize];

            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    if !in_shape(shape, offset_x, offset_y) {
                        continue;
                    }

                    let value = read_reflected(input, width, height, x + offset_x, y + offset_y);
                    best = match extreme {
                        Extreme::Maximum => best.max(value),
                        Extreme::Minimum => best.min(value),
                    };
                }
            }

            output.push(best);
        }
    }

    output
}

/// Blurs a plane with a 3x3 kernel on the CPU, mirroring [smooth](crate::neighbourhood::smooth).
pub fn reference_smooth(input: &[f32], width: u32, height: u32, smooth: Smooth) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut total = 0.0f32;

            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    let value = read_reflected(input, width, height, x + offset_x, y + offset_y);
                    let weight = match smooth {
                        Smooth::Binomial => ((2 - offset_x.abs()) * (2 - offset_y.abs())) as f32,
                        Smooth::Box => 1.0,
                    };

                    total += value * weight;
                }
            }

            let divisor = match smooth {
                Smooth::Binomial => 16.0,
                Smooth::Box => 9.0,
            };

            output.push(total / divisor);
        }
    }

    output
}
