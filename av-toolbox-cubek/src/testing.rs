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
