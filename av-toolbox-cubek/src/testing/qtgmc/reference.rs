use crate::deinterlace::qtgmc::{Parity, bob_taps};

/// Bobs the field of `parity` in `frame` to a full-height plane on the CPU, mirroring
/// [bob](crate::deinterlace::qtgmc::bob).
///
/// The result is not rounded.
pub fn reference_bob(frame: &[f32], (width, height): (usize, usize), parity: Parity) -> Vec<f32> {
    let field_height = height / 2;
    let table = bob_taps(field_height as u32, parity);
    let count = table.count as usize;
    let last = field_height as i32 - 1;

    let mut output = Vec::with_capacity(width * height);
    for row in 0..height {
        let start = table.starts[row];
        let weights = &table.weights[row * count..(row + 1) * count];

        for column in 0..width {
            let mut total = 0.0f32;
            for (step, weight) in weights.iter().enumerate() {
                let field_row = (start + step as i32).clamp(0, last) as usize;
                let frame_row = field_row * 2 + parity as usize;
                total += frame[frame_row * width + column] * weight;
            }

            output.push(total);
        }
    }

    output
}
