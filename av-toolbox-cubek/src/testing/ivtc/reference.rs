/// One plane of integer code values, `width` samples across.
#[derive(Debug, Clone, Copy)]
pub struct IntPlane<'a> {
    pub samples: &'a [i32],
    pub width: usize,
    pub height: usize,
}

impl IntPlane<'_> {
    pub fn at(&self, x: usize, y: usize) -> i32 {
        self.samples[y * self.width + x]
    }
}

/// The rows of parity `matching_parity` from `matching` and the rest from `other`.
pub fn reference_weave(
    matching: &[i32],
    other: &[i32],
    width: usize,
    height: usize,
    matching_parity: usize,
) -> Vec<i32> {
    let mut output = Vec::with_capacity(width * height);
    for row in 0..height {
        let source = if row % 2 == matching_parity {
            matching
        } else {
            other
        };
        let start = row * width;
        output.extend_from_slice(&source[start..start + width]);
    }

    output
}

/// VFM's weave for `matched` of `frames` (previous, current, next) with VFM's `field`.
pub fn reference_match_weave(
    frames: [&[i32]; 3],
    width: usize,
    height: usize,
    matched: usize,
    field: usize,
) -> Vec<i32> {
    let matching_parity = 1 - field;
    let [previous, current, next] = frames;
    let (matching, other) = match matched {
        0 => (current, previous),
        1 => (current, current),
        2 => (current, next),
        3 => (previous, current),
        4 => (next, current),
        _ => panic!("VFM has five matches, not {matched}"),
    };

    reference_weave(matching, other, width, height, matching_parity)
}

/// The sum of absolute differences of two planes.
pub fn reference_difference(first: &[i32], second: &[i32]) -> u64 {
    first
        .iter()
        .zip(second)
        .map(|(first_sample, second_sample)| u64::from(first_sample.abs_diff(*second_sample)))
        .sum()
}

/// VapourSynth's PlaneStats difference of a luma plane from `sum`, its absolute difference total.
pub fn scene_difference(sum: u64, width: usize, height: usize, bits: u32) -> f64 {
    let peak = (1i64 << bits) - 1;
    let scale = width as i64 * height as i64 * peak;

    sum as f64 / scale as f64
}
