use crate::deinterlace::qtgmc::{
    NEGATIVE_STEPS,
    POSITIVE_STEPS,
    Parity,
    bob_taps,
    fmtc_gauss_taps,
    restore_limits,
    search_levels,
};
use crate::neighbourhood::{Edge, Extreme, Shape, Smooth};
use crate::testing::{
    reference_chain_with_edge,
    reference_extreme,
    reference_resize_float_with_edge,
    reference_smooth_with_edge,
};

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

/// Keeps only thin horizontal differences between `input` and `reference` on the CPU, mirroring
/// [shimmer](crate::deinterlace::qtgmc::shimmer).
///
/// The result is not rounded.
pub fn reference_shimmer(
    input: &[f32],
    reference: &[f32],
    width: usize,
    height: usize,
    bits: u32,
) -> Vec<f32> {
    let neutral = (1u32 << (bits - 1)) as f32;
    let peak = ((1u32 << bits) - 1) as f32;
    let (above, below) = restore_limits(bits);

    let difference: Vec<f32> = reference
        .iter()
        .zip(input)
        .map(|(wanted, found)| (wanted - found).clamp(-neutral, peak - neutral))
        .collect();
    let (columns, rows) = (width as u32, height as u32);
    let positive = reference_chain_with_edge(&difference, columns, rows, &POSITIVE_STEPS, Edge::Clamp);
    let negative = reference_chain_with_edge(&difference, columns, rows, &NEGATIVE_STEPS, Edge::Clamp);

    let mut output = Vec::with_capacity(input.len());
    for position in 0..input.len() {
        let original = difference[position];

        let mut restored = original;
        if original >= above {
            restored = positive[position].max(0.0);
        }

        let mut result = restored;
        if restored <= below {
            result = negative[position].min(0.0);
        }

        output.push((input[position] + result).clamp(0.0, peak));
    }

    output
}

/// Blurs a plane with fmtc's gaussian of `a1 = sharpness` on both axes.
fn gaussian(input: &[f32], width: usize, height: usize, sharpness: f32) -> Vec<f32> {
    let across = fmtc_gauss_taps(width as u32, sharpness);
    let down = fmtc_gauss_taps(height as u32, sharpness);

    reference_resize_float_with_edge(input, width as u32, height as u32, &across, &down, Edge::Clamp)
}

/// The two clips the search clip's final expressions compare, as `(blur, tweaked)`.
///
/// `blur` is `repaired` smoothed and merged with itself, and `tweaked` is `bob` held within a few codes of
/// `repaired`. Neither is rounded.
pub fn reference_search_inputs(
    repaired: &[f32],
    bob: &[f32],
    width: usize,
    height: usize,
    bits: u32,
) -> (Vec<f32>, Vec<f32>) {
    let peak = ((1u32 << bits) - 1) as f32;
    let (tolerance, _, _) = search_levels(bits);

    let blurred = reference_smooth_with_edge(
        repaired,
        width as u32,
        height as u32,
        Smooth::Binomial,
        Edge::Clamp,
    );
    let spread = gaussian(&blurred, width, height, 2.0);
    let blur = merge(&spread, repaired, 0.1);

    let tweaked = repaired
        .iter()
        .zip(bob)
        .map(|(centre, bob_sample)| {
            bob_sample
                .clamp(centre - tolerance, centre + tolerance)
                .clamp(0.0, peak)
        })
        .collect();

    (blur, tweaked)
}

/// The search clip's final expressions on the CPU, mirroring
/// [search_clip](crate::deinterlace::qtgmc::search_clip) after QTGMC's blur of `repaired`.
///
/// The result is not rounded.
pub fn reference_search_clip(
    repaired: &[f32],
    bob: &[f32],
    width: usize,
    height: usize,
    bits: u32,
) -> Vec<f32> {
    let peak = ((1u32 << bits) - 1) as f32;
    let (_, far, near) = search_levels(bits);
    let (blur, tweaked) = reference_search_inputs(repaired, bob, width, height, bits);

    let mut output = Vec::with_capacity(repaired.len());
    for (smoothed, wanted) in blur.iter().zip(&tweaked) {
        let mut result = (smoothed * 51.0 + wanted * 49.0) / 100.0;
        if smoothed + far < *wanted {
            result = smoothed + near;
        }
        if smoothed - far > *wanted {
            result = smoothed - near;
        }

        output.push(result.clamp(0.0, peak));
    }

    output
}

/// QTGMC's resharpen on the CPU, mirroring [resharpen](crate::deinterlace::qtgmc::resharpen) after the
/// blur of [vertical_midpoint](crate::deinterlace::qtgmc::vertical_midpoint), which rounds its midpoint.
///
/// The result is not rounded.
pub fn reference_resharpen(input: &[f32], width: usize, height: usize, strength: f32, bits: u32) -> Vec<f32> {
    let peak = ((1u32 << bits) - 1) as f32;
    let (columns, rows) = (width as u32, height as u32);

    let largest = reference_extreme(input, columns, rows, Extreme::Maximum, Shape::Vertical);
    let smallest = reference_extreme(input, columns, rows, Extreme::Minimum, Shape::Vertical);
    let midpoint: Vec<f32> = merge(&largest, &smallest, 0.5)
        .iter()
        .map(|sample| (sample + 0.5).floor())
        .collect();
    let blurred = reference_smooth_with_edge(&midpoint, columns, rows, Smooth::Binomial, Edge::Clamp);

    input
        .iter()
        .zip(&blurred)
        .map(|(sample, blurred_sample)| (sample + (sample - blurred_sample) * strength).clamp(0.0, peak))
        .collect()
}

/// QTGMC's luma back-blend on the CPU, mirroring
/// [subtract](crate::deinterlace::qtgmc::subtract) after its blurs.
///
/// The difference between `sharpened` and `unsharpened` is clamped as an integer `MakeDiff` stores it. The
/// result is not rounded.
pub fn reference_back_blend(
    sharpened: &[f32],
    unsharpened: &[f32],
    width: usize,
    height: usize,
    bits: u32,
) -> Vec<f32> {
    let peak = ((1u32 << bits) - 1) as f32;
    let (lowest, highest) = ((peak + 1.0) / -2.0, (peak - 1.0) / 2.0);

    let difference: Vec<f32> = sharpened
        .iter()
        .zip(unsharpened)
        .map(|(sharp, plain)| (sharp - plain).clamp(lowest, highest))
        .collect();
    let blurred = reference_smooth_with_edge(
        &difference,
        width as u32,
        height as u32,
        Smooth::Binomial,
        Edge::Clamp,
    );
    let spread = gaussian(&blurred, width, height, 5.0);

    sharpened
        .iter()
        .zip(&spread)
        .map(|(sharp, back)| (sharp - back).clamp(0.0, peak))
        .collect()
}

/// Clamps `input` between `lower` and `upper` on the CPU, mirroring
/// [clamp_between](crate::deinterlace::qtgmc::clamp_between).
pub fn reference_clamp(input: &[f32], upper: &[f32], lower: &[f32]) -> Vec<f32> {
    input
        .iter()
        .zip(upper)
        .zip(lower)
        .map(|((sample, high), low)| sample.min(*high).max(*low))
        .collect()
}
