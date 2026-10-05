use super::reflect;
use crate::neighbourhood::Edge;
use crate::resize::Taps;

/// Applies `across` then `down` on the CPU, rounding to U8 at the end, mirroring
/// [apply](crate::resize::apply).
pub fn reference_resize(input: &[u8], width: u32, height: u32, across: &Taps, down: &Taps) -> Vec<u8> {
    let wide: Vec<u16> = input.iter().map(|sample| u16::from(*sample)).collect();
    let largest = u16::from(u8::MAX);
    let resized = reference_resize_wide(&wide, width, height, across, down, largest);

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
    let wide: Vec<f32> = input.iter().map(|sample| f32::from(*sample)).collect();
    let resized = reference_resize_float(&wide, width, height, across, down);
    let ceiling = f32::from(largest);

    resized
        .iter()
        .map(|value| value.round().clamp(0.0, ceiling) as u16)
        .collect()
}

/// The tap position brought inside `length` samples as `edge` says.
fn fold(tap: i32, length: u32, edge: Edge) -> u32 {
    match edge {
        Edge::Reflect => reflect(tap, length as i32) as u32,
        Edge::Clamp => tap.clamp(0, length as i32 - 1) as u32,
    }
}

/// Applies `across` then `down` on the CPU in f32, mirroring [apply_float](crate::resize::apply_float).
pub fn reference_resize_float(
    input: &[f32],
    width: u32,
    height: u32,
    across: &Taps,
    down: &Taps,
) -> Vec<f32> {
    reference_resize_float_with_edge(input, width, height, across, down, Edge::Reflect)
}

/// The same as [reference_resize_float], mirroring
/// [apply_float_with_edge](crate::resize::apply_float_with_edge).
pub fn reference_resize_float_with_edge(
    input: &[f32],
    width: u32,
    height: u32,
    across: &Taps,
    down: &Taps,
    edge: Edge,
) -> Vec<f32> {
    let out_width = across.starts.len() as u32;
    let out_height = down.starts.len() as u32;

    let mut rows = vec![0.0f32; (out_width * height) as usize];
    for row in 0..height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..across.count {
                let tap = across.starts[column as usize] + step as i32;
                let source = fold(tap, width, edge);
                let weight = across.weights[(column * across.count + step) as usize];
                total += input[(row * width + source) as usize] * weight;
            }

            rows[(row * out_width + column) as usize] = total;
        }
    }

    let mut output = Vec::with_capacity((out_width * out_height) as usize);
    for row in 0..out_height {
        for column in 0..out_width {
            let mut total = 0.0f32;
            for step in 0..down.count {
                let tap = down.starts[row as usize] + step as i32;
                let source = fold(tap, height, edge);
                let weight = down.weights[(row * down.count + step) as usize];
                total += rows[(source * out_width + column) as usize] * weight;
            }

            output.push(total);
        }
    }

    output
}
