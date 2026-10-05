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

/// The bob at `index`, or the nearest bob when `index` falls outside `bobs`.
fn clamped_bob(bobs: &[Vec<f32>], index: isize) -> &[f32] {
    let last = bobs.len() as isize - 1;

    &bobs[index.clamp(0, last) as usize]
}

/// The mean of the bobs within `radius` of `centre`.
fn average_frames(bobs: &[Vec<f32>], centre: usize, radius: isize, length: usize) -> Vec<f32> {
    let count = (2 * radius + 1) as f32;

    let mut total = vec![0.0f32; length];
    for offset in -radius..=radius {
        let bob = clamped_bob(bobs, centre as isize + offset);
        for (sum, sample) in total.iter_mut().zip(bob) {
            *sum += sample;
        }
    }

    total.iter().map(|sum| sum / count).collect()
}

/// `first * (1 - weight) + second * weight`.
fn merge(first: &[f32], second: &[f32], weight: f32) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .map(|(first_sample, second_sample)| first_sample * (1.0 - weight) + second_sample * weight)
        .collect()
}

/// QTGMC's TR0 temporal average of the bob at `centre`, as the two averages and merges it is built from.
///
/// Bobs before the first or after the last repeat the nearest, and the result is not rounded.
pub fn reference_tr0(bobs: &[Vec<f32>], centre: usize, tr0: u32, width: usize, height: usize) -> Vec<f32> {
    let length = width * height;
    let bob = &bobs[centre];
    if tr0 == 0 {
        return bob.clone();
    }

    let ts1 = average_frames(bobs, centre, 1, length);
    if tr0 == 1 {
        return merge(&ts1, bob, 0.25);
    }

    let ts2 = average_frames(bobs, centre, 2, length);
    let smoothed = merge(&ts1, &ts2, 0.357);

    merge(&smoothed, bob, 0.125)
}
