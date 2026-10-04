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

/// Whether sample (`x`, `y`) of `plane` is combed, the way `calcMI` marks it.
fn combed_at(plane: IntPlane<'_>, x: usize, y: usize, cthresh: i32) -> bool {
    let cthresh6 = cthresh * 6;
    let last = plane.height - 1;
    let here = plane.at(x, y);

    if y == 0 {
        let below = plane.at(x, 1);
        let first = here - below;
        if first <= cthresh && first >= -cthresh {
            return false;
        }

        let two_below = plane.at(x, 2);
        let curve = two_below + here * 4 + two_below - 6 * below;
        return curve.abs() > cthresh6;
    }

    if y == last {
        let above = plane.at(x, y - 1);
        let first = here - above;
        if first <= cthresh && first >= -cthresh {
            return false;
        }

        let two_above = plane.at(x, y - 2);
        let curve = 2 * two_above + here * 4 - 6 * above;
        return curve.abs() > cthresh6;
    }

    let above = plane.at(x, y - 1);
    let below = plane.at(x, y + 1);
    let first = here - above;
    let second = here - below;
    let peak = first > cthresh && second > cthresh;
    let trough = first < -cthresh && second < -cthresh;
    if !peak && !trough {
        return false;
    }

    // Row 1 mirrors its missing upper tap from below, and row `height - 2` its lower tap from above.
    let (outer_above, outer_below) = if y == 1 {
        (plane.at(x, 3), plane.at(x, 3))
    } else if y == last - 1 {
        (plane.at(x, y - 2), plane.at(x, y - 2))
    } else {
        (plane.at(x, y - 2), plane.at(x, y + 2))
    };
    let curve = outer_above + here * 4 + outer_below - 3 * (above + below);

    curve.abs() > cthresh6
}

/// `calcMI`'s comb mask of one plane, 255 where combed and 0 elsewhere.
pub fn reference_comb_mask(plane: IntPlane<'_>, cthresh: i32) -> Vec<u8> {
    let samples = plane.width * plane.height;
    if cthresh < 0 {
        return vec![255; samples];
    }

    let mut mask = vec![0; samples];
    for y in 0..plane.height {
        for x in 0..plane.width {
            if combed_at(plane, x, y, cthresh) {
                mask[y * plane.width + x] = 255;
            }
        }
    }

    mask
}

/// Whether sample (`x`, `y`) of `mask` is set along with at least one of its 8 neighbours.
fn clustered(mask: &[u8], width: usize, x: usize, y: usize) -> bool {
    let set = |column: usize, row: usize| mask[row * width + column] == 255;
    if !set(x, y) {
        return false;
    }

    for row in y - 1..=y + 1 {
        for column in x - 1..=x + 1 {
            let centre = row == y && column == x;
            if !centre && set(column, row) {
                return true;
            }
        }
    }

    false
}

/// The luma rows a clustered chroma sample on chroma row `y` marks.
fn rows_to_mark(y: usize, shift_y: u32) -> Vec<usize> {
    let row = y << shift_y;
    if shift_y == 0 {
        return vec![row];
    }

    if y % 2 == 1 {
        vec![row - 1, row, row + 1]
    } else {
        vec![row, row + 1, row + 2]
    }
}

/// Marks the luma samples under every chroma sample clustered in `u` or `v`, as `calcMI` does.
///
/// Chroma samples on the plane's edge are never tested, though they still count as neighbours.
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the marking"
)]
pub fn reference_chroma_into_luma(
    luma: &mut [u8],
    luma_width: usize,
    u: &[u8],
    v: &[u8],
    chroma_width: usize,
    chroma_height: usize,
    shift_x: u32,
    shift_y: u32,
) {
    for y in 1..chroma_height - 1 {
        for x in 1..chroma_width - 1 {
            let combed = clustered(u, chroma_width, x, y) || clustered(v, chroma_width, x, y);
            if !combed {
                continue;
            }

            let column = x << shift_x;
            for row in rows_to_mark(y, shift_y) {
                let start = row * luma_width + column;
                luma[start] = 255;
                if shift_x > 0 {
                    luma[start + 1] = 255;
                }
            }
        }
    }
}

/// The highest count of samples, in any of four offset grids of `block_x` by `block_y` blocks, whose mask is
/// set on their own row and the rows above and below.
///
/// It equals `calcMI`'s per-region count for any `height` of at least `block_y`.
pub fn reference_mic(mask: &[u8], width: usize, height: usize, block_x: usize, block_y: usize) -> u32 {
    let half_x = block_x / 2;
    let half_y = block_y / 2;
    let blocks_across = (width + half_x) / block_x + 1;
    let blocks_down = (height + half_y) / block_y + 1;
    let row_stride = blocks_across * 4;
    let mut counts = vec![0u32; blocks_across * blocks_down * 4];
    let set = |x: usize, y: usize| mask[y * width + x] == 255;

    for y in 1..height - 1 {
        for x in 0..width {
            let triple = set(x, y - 1) && set(x, y) && set(x, y + 1);
            if !triple {
                continue;
            }

            let first_row = (y / block_y) * row_stride;
            let second_row = ((y + half_y) / block_y) * row_stride;
            let first_column = (x / block_x) * 4;
            let second_column = ((x + half_x) / block_x) * 4;
            counts[first_row + first_column] += 1;
            counts[first_row + second_column + 1] += 1;
            counts[second_row + first_column + 2] += 1;
            counts[second_row + second_column + 3] += 1;
        }
    }

    counts.into_iter().max().unwrap_or(0)
}
