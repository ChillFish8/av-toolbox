use crate::deinterlace::qtgmc::{
    NEGATIVE_STEPS,
    POSITIVE_STEPS,
    Parity,
    assert_bits,
    bob_taps,
    fmtc_gauss_taps,
    restore_limits,
    search_levels,
};
use crate::motion::{
    BlockGrid,
    WEIGHT_HALF,
    WEIGHT_MAX,
    WEIGHT_SHIFT,
    covered_area,
    sad_byte,
    simple_resize_tables,
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
                let frame_row = field_row * 2 + parity.row_offset() as usize;
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
pub(super) fn merge(first: &[f32], second: &[f32], weight: f32) -> Vec<f32> {
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
    assert_bits(bits);

    let peak = ((1u32 << bits) - 1) as f32;
    let (above, below) = restore_limits(bits);

    let difference = reference_clamped_difference(reference, input, bits);
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

/// Blurs a plane with the 3x3 binomial and then fmtc's gaussian of `a1 = sharpness` on the CPU, mirroring
/// [blur_and_spread](crate::deinterlace::qtgmc::blur_and_spread).
///
/// Both blurs clamp at the border.
pub fn reference_blur_and_spread(input: &[f32], width: usize, height: usize, sharpness: f32) -> Vec<f32> {
    let (columns, rows) = (width as u32, height as u32);
    let blurred = reference_smooth_with_edge(input, columns, rows, Smooth::Binomial, Edge::Clamp);
    let across = fmtc_gauss_taps(columns, sharpness);
    let down = fmtc_gauss_taps(rows, sharpness);

    reference_resize_float_with_edge(&blurred, columns, rows, &across, &down, Edge::Clamp)
}

/// The two clips the search clip's final expressions compare, as `(blur, tweaked)`.
///
/// `blur` is `repaired` smoothed, then merged back with `repaired`, and `tweaked` is `bob` held within a few
/// codes of `repaired`. Neither is rounded.
pub fn reference_search_inputs(
    repaired: &[f32],
    bob: &[f32],
    width: usize,
    height: usize,
    bits: u32,
) -> (Vec<f32>, Vec<f32>) {
    assert_bits(bits);

    let peak = ((1u32 << bits) - 1) as f32;
    let (tolerance, _, _) = search_levels(bits);

    let spread = reference_blur_and_spread(repaired, width, height, 2.0);
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
    assert_bits(bits);

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
    assert_bits(bits);

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
    assert_bits(bits);

    let peak = ((1u32 << bits) - 1) as f32;

    let difference = reference_clamped_difference(sharpened, unsharpened, bits);
    let spread = reference_blur_and_spread(&difference, width, height, 5.0);

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

/// `first − second` clamped as `std.MakeDiff` stores it on the CPU, mirroring
/// [clamped_difference](crate::deinterlace::qtgmc::clamped_difference).
pub fn reference_clamped_difference(first: &[f32], second: &[f32], bits: u32) -> Vec<f32> {
    assert_bits(bits);

    let peak = ((1u32 << bits) - 1) as f32;
    let (lowest, highest) = ((peak + 1.0) / -2.0, (peak - 1.0) / 2.0);

    first
        .iter()
        .zip(second)
        .map(|(minuend, subtrahend)| (minuend - subtrahend).clamp(lowest, highest))
        .collect()
}

/// The per-sample maximum and minimum of `centre` and `neighbours` on the CPU, mirroring
/// [temporal_limits](crate::deinterlace::qtgmc::temporal_limits).
pub fn reference_temporal_limits(centre: &[f32], neighbours: &[&[f32]]) -> (Vec<f32>, Vec<f32>) {
    let mut upper = centre.to_vec();
    let mut lower = centre.to_vec();
    for neighbour in neighbours {
        for (position, sample) in neighbour.iter().enumerate() {
            upper[position] = upper[position].max(*sample);
            lower[position] = lower[position].min(*sample);
        }
    }

    (upper, lower)
}

/// `(first × (max − weight) + second × weight + half) >> shift`, one of `SimpleResize`'s blends.
fn simple_blend(first: u32, second: u32, weight: u32) -> u32 {
    let first_weight = WEIGHT_MAX - weight;

    (first * first_weight + second * weight + WEIGHT_HALF) >> WEIGHT_SHIFT
}

/// MVTools' `Mask(kind=1)` of a field's `sads` on the CPU, mirroring [sad_mask](crate::motion::sad_mask).
///
/// The lines cited are vapoursynth-mvtools' `src/` at master.
///
/// - A `flagged` field is not usable, so every sample is `ysc = 0` (`MVMask.cpp` 193 to 200).
/// - Each block's SAD becomes a byte through `ByteNorm` with `dSADNormFactor = 4 × (1 / ml) / blksize²`
///   (`MVMask.cpp` 143 and 304, `MaskFun.cpp` 128 to 132). At `time = 100`, `MakeSADMaskTime` reads every
///   block's own SAD (`MaskFun.cpp` 135 to 159).
/// - The bytes are resized to the covered area, `nWidthB` by `nHeightB` (`MVMask.cpp` 309, 310 and 331),
///   through `InitTables` (`SimpleResize.cpp` 26 to 55). Each row blends two block rows into a byte, then
///   each sample blends two of those, both in fixed point with 14 bits of weight (`SimpleResize.cpp` 76 to
///   117, `SimpleResize.h` 27 to 31).
/// - Columns and rows past the covered area repeat its last column and row (`MVMask.cpp` 164 to 170).
///
/// A byte is stored as its value times `1 / 255`.
pub fn reference_sad_mask(
    sads: &[f32],
    grid: &BlockGrid,
    flagged: bool,
    ml: f32,
    (width, height): (usize, usize),
) -> Vec<f32> {
    if flagged {
        return vec![0.0; width * height];
    }

    let across = grid.across as usize;
    let down = grid.down as usize;
    let small_mask: Vec<u32> = sads.iter().map(|sad| sad_byte(*sad, ml, grid.blksize)).collect();

    let (covered_width, covered_height) = covered_area(grid);
    let (horizontal_offsets, horizontal_weights) = simple_resize_tables(covered_width, grid.across);
    let (vertical_offsets, vertical_weights) = simple_resize_tables(covered_height, grid.down);
    let (covered_width, covered_height) = (covered_width as usize, covered_height as usize);

    // A read clamped into the grid only moves one that weighs 0.
    let block_row = |offset: i32| offset.clamp(0, down as i32 - 1) as usize;
    let block_column = |offset: i32| offset.clamp(0, across as i32 - 1) as usize;

    let mut covered = vec![0u32; covered_width * covered_height];
    for y in 0..covered_height {
        let weight_bottom = vertical_weights[y];
        let first_row = block_row(vertical_offsets[y]);
        let second_row = block_row(vertical_offsets[y] + 1);

        let work: Vec<u32> = (0..across)
            .map(|x| {
                let top = small_mask[first_row * across + x];
                let bottom = small_mask[second_row * across + x];
                simple_blend(top, bottom, weight_bottom)
            })
            .collect();

        for x in 0..covered_width {
            let weight_right = horizontal_weights[x];
            let left = work[block_column(horizontal_offsets[x])];
            let right = work[block_column(horizontal_offsets[x] + 1)];

            covered[y * covered_width + x] = simple_blend(left, right, weight_right);
        }
    }

    let inverse_peak = 1.0f32 / 255.0;
    let mut output = Vec::with_capacity(width * height);
    for y in 0..height {
        let source_row = y.min(covered_height - 1);

        for x in 0..width {
            let source_column = x.min(covered_width - 1);
            let byte = covered[source_row * covered_width + source_column];

            output.push(byte as f32 * inverse_peak);
        }
    }

    output
}

/// `(above + 2 × centre + below) / 4` with the border clamped on the CPU, mirroring
/// [vertical_blur](crate::deinterlace::qtgmc::vertical_blur).
pub fn reference_vertical_blur(input: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(input.len());
    for row in 0..height {
        let above = row.saturating_sub(1);
        let below = (row + 1).min(height - 1);

        for column in 0..width {
            let top = input[above * width + column];
            let centre = input[row * width + column];
            let bottom = input[below * width + column];

            output.push((top + 2.0 * centre + bottom) * 0.25);
        }
    }

    output
}

/// `original + (rebuilt − original) × mask` on the CPU, mirroring
/// [repair_blend](crate::deinterlace::qtgmc::repair_blend).
pub fn reference_repair_blend(original: &[f32], rebuilt: &[f32], mask: &[f32]) -> Vec<f32> {
    original
        .iter()
        .zip(rebuilt)
        .zip(mask)
        .map(|((kept, rebuild), weight)| kept + (rebuild - kept) * weight)
        .collect()
}
