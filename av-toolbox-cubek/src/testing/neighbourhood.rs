use super::reflect;
use crate::neighbourhood::{Edge, Extreme, Shape, Smooth, Spread, Step, median_of_three_sample};

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
        Shape::Corners => offset_x != 0 && offset_y != 0,
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

/// The value at `column`, `row`, with the edge sample repeated past the plane.
fn read_clamped(input: &[f32], width: u32, height: u32, column: i32, row: i32) -> f32 {
    let x = column.clamp(0, width as i32 - 1);
    let y = row.clamp(0, height as i32 - 1);

    input[(y as u32 * width + x as u32) as usize]
}

/// Moves each sample toward its 8 neighbours' mean on the CPU, mirroring
/// [spread](crate::neighbourhood::spread).
pub fn reference_spread(input: &[f32], width: u32, height: u32, spread: Spread) -> Vec<f32> {
    reference_spread_with_edge(input, width, height, spread, Edge::Reflect)
}

/// The same as [reference_spread], mirroring [spread_with_edge](crate::neighbourhood::spread_with_edge).
pub fn reference_spread_with_edge(
    input: &[f32],
    width: u32,
    height: u32,
    spread: Spread,
    edge: Edge,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let centre = read_reflected(input, width, height, x, y);
            let mut total = 0.0f32;

            for offset_y in -1..=1 {
                for offset_x in -1..=1 {
                    if offset_x == 0 && offset_y == 0 {
                        continue;
                    }

                    total += match edge {
                        Edge::Reflect => read_reflected(input, width, height, x + offset_x, y + offset_y),
                        Edge::Clamp => read_clamped(input, width, height, x + offset_x, y + offset_y),
                    };
                }
            }

            let mean = total / 8.0;
            let picked = match spread {
                Spread::Inflate => centre.max(mean),
                Spread::Deflate => centre.min(mean),
            };

            output.push(picked);
        }
    }

    output
}

/// Runs each of `steps` in turn on the CPU, mirroring [chain](crate::neighbourhood::chain).
pub fn reference_chain(input: &[f32], width: u32, height: u32, steps: &[Step]) -> Vec<f32> {
    reference_chain_with_edge(input, width, height, steps, Edge::Reflect)
}

/// The same as [reference_chain], mirroring [chain_with_edge](crate::neighbourhood::chain_with_edge).
pub fn reference_chain_with_edge(
    input: &[f32],
    width: u32,
    height: u32,
    steps: &[Step],
    edge: Edge,
) -> Vec<f32> {
    let mut plane = input.to_vec();

    for step in steps {
        plane = match *step {
            Step::Extreme(extreme, shape) => reference_extreme(&plane, width, height, extreme, shape),
            Step::Spread(spread) => reference_spread_with_edge(&plane, width, height, spread, edge),
        };
    }

    plane
}

/// Takes the median of each sample's window on the CPU, mirroring [median](crate::neighbourhood::median).
pub fn reference_median(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    let reach = radius as i32;
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let window_size = ((2 * radius + 1) * (2 * radius + 1)) as usize;
            let mut window = Vec::with_capacity(window_size);

            for offset_y in -reach..=reach {
                for offset_x in -reach..=reach {
                    let value = read_reflected(input, width, height, x + offset_x, y + offset_y);
                    window.push(value);
                }
            }

            window.sort_by(f32::total_cmp);
            let middle = window.len() / 2;
            output.push(window[middle]);
        }
    }

    output
}

/// Runs MinBlur on the CPU, mirroring [min_blur](crate::neighbourhood::min_blur).
pub fn reference_min_blur(input: &[f32], width: u32, height: u32, radius: u32) -> Vec<f32> {
    debug_assert!((1..=3).contains(&radius));

    let blurred = match radius {
        1 => reference_smooth(input, width, height, Smooth::Binomial),
        2 => {
            let binomial = reference_smooth(input, width, height, Smooth::Binomial);
            reference_smooth(&binomial, width, height, Smooth::Box)
        },
        3 => {
            let binomial = reference_smooth(input, width, height, Smooth::Binomial);
            let boxed_once = reference_smooth(&binomial, width, height, Smooth::Box);
            reference_smooth(&boxed_once, width, height, Smooth::Box)
        },
        _ => unreachable!("radius is 1, 2 or 3"),
    };
    let median_plane = reference_median(input, width, height, radius);

    let mut output = Vec::with_capacity(input.len());
    for index in 0..input.len() {
        let picked = median_of_three_sample(input[index], blurred[index], median_plane[index]);

        output.push(picked);
    }

    output
}

/// havsfunc's `AvsPrewitt` on the CPU, mirroring [prewitt](crate::neighbourhood::prewitt).
pub fn reference_prewitt(input: &[f32], width: u32, height: u32, peak: f32) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let top_left = read_reflected(input, width, height, x - 1, y - 1);
            let top = read_reflected(input, width, height, x, y - 1);
            let top_right = read_reflected(input, width, height, x + 1, y - 1);
            let left = read_reflected(input, width, height, x - 1, y);
            let right = read_reflected(input, width, height, x + 1, y);
            let bottom_left = read_reflected(input, width, height, x - 1, y + 1);
            let bottom = read_reflected(input, width, height, x, y + 1);
            let bottom_right = read_reflected(input, width, height, x + 1, y + 1);

            let diagonal = (top_left + top + left - right - bottom - bottom_right).abs();
            let vertical = (top_left + top + top_right - bottom_left - bottom - bottom_right).abs();
            let horizontal = (top_left - top_right + left - right + bottom_left - bottom_right).abs();
            let anti_diagonal = (left - top - top_right - right + bottom_left + bottom).abs();

            let strongest = diagonal.max(vertical).max(horizontal).max(anti_diagonal);
            output.push(strongest.min(peak));
        }
    }

    output
}
