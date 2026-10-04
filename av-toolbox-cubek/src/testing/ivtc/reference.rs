use crate::ivtc::{ComparePass, FieldSums, MatchSums};

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

/// The absolute differences of two frames' rows `start_row`, `start_row + 2`, and so on, `height / 2` rows.
fn field_differences(first: IntPlane<'_>, second: IntPlane<'_>, start_row: usize) -> Vec<i32> {
    let width = first.width;
    let rows = first.height / 2;
    let mut differences = Vec::with_capacity(rows * width);
    for row in 0..rows {
        let source_row = start_row + 2 * row;
        for x in 0..width {
            let difference = first.at(x, source_row) - second.at(x, source_row);
            differences.push(difference.abs());
        }
    }

    differences
}

/// `buildDiffMap`'s value for difference row `row` and column `x`, 0, 1, 3 or 5.
fn map_value(differences: &[i32], width: usize, height: usize, row: usize, x: usize, shift: u32) -> u8 {
    let low = 3 << shift;
    let high = 19 << shift;
    let at = |column: usize, line: usize| differences[line * width + column];
    let difference = at(x, row);
    if difference <= low {
        return 0;
    }

    // `buildDiffMap` stops counting once `count` reaches 2, which only decides `count > 1`.
    let mut count = 0;
    let mut column = x - 1;
    while column < x + 2 && count < 2 {
        count += u32::from(at(column, row - 1) > low);
        count += u32::from(at(column, row) > low);
        count += u32::from(at(column, row + 1) > low);
        column += 1;
    }

    if count <= 1 {
        return 0;
    }

    let mut value = 1;
    if difference <= high {
        return value;
    }

    // This loop's early exit decides which of `upper` and `lower` get set, so it is kept exactly.
    let mut upper = false;
    let mut lower = false;
    count = 0;
    column = x - 1;
    while column < x + 2 && count < 6 {
        if at(column, row - 1) > high {
            count += 1;
            upper = true;
        }

        if at(column, row) > high {
            count += 1;
        }

        if at(column, row + 1) > high {
            count += 1;
            lower = true;
        }

        column += 1;
    }

    if count <= 3 {
        return value;
    }

    if upper && lower {
        return value + 2;
    }

    let y = 2 * row;
    let mut upper_two = false;
    let mut lower_two = false;
    let first_column = x.saturating_sub(4);
    let end_column = usize::min(x + 5, width);
    for column in first_column..end_column {
        if y != 2 && at(column, row - 2) > high {
            upper_two = true;
        }

        if at(column, row - 1) > high {
            upper = true;
        }

        if at(column, row + 1) > high {
            lower = true;
        }

        if y != height - 4 && at(column, row + 2) > high {
            lower_two = true;
        }
    }

    if (upper && (lower || upper_two)) || (lower && (upper || lower_two)) {
        value += 2;
    } else if count > 5 {
        value += 4;
    }

    value
}

/// Adds one sample's `difference` into `sums` where its map `flags` and VFM's thresholds allow it.
fn accumulate(sums: &mut MatchSums, difference: i32, flags: u8, shift: u32) {
    let amount = difference as u64;
    if difference > 23 << shift && flags & 1 != 0 {
        sums.combing += amount;
    }

    if difference > 42 << shift {
        if flags & 2 != 0 {
            sums.motion += amount;
        }

        if flags & 4 != 0 {
            sums.isolated += amount;
        }
    }
}

/// One plane of VFM's field compare, `compareFieldsSlow` for `pass`, with the map `buildDiffMap` builds.
///
/// `frames` are the previous, current and next frames. The map holds one byte per sample.
pub fn reference_compare_plane(frames: [IntPlane<'_>; 3], pass: &ComparePass) -> (FieldSums, Vec<u8>) {
    assert!(
        pass.first < 3 && pass.second < 3,
        "the field compare takes matches 0, 1 or 2, not {} and {}",
        pass.first,
        pass.second,
    );

    let first = frames[pass.first as usize];
    let second = frames[pass.second as usize];
    let current = frames[1];
    let width = current.width;
    let height = current.height;
    let field = pass.field as usize;
    let other_row = if field == 1 { 1 } else { 2 };
    let current_row = 3 - field;
    let mut map = vec![0u8; width * height];

    // `buildDiffMap` is handed rows `other_row` or `other_row + 2` and reads from two rows above them.
    let (difference_start, map_start) = if field == 0 {
        (other_row - 2, other_row)
    } else {
        (other_row, other_row + 2)
    };
    let differences = field_differences(first, second, difference_start);
    for y in (2..height - 2).step_by(2) {
        let row = y / 2;
        let map_row = map_start + y - 2;
        for x in 1..width - 1 {
            map[map_row * width + x] = map_value(&differences, width, height, row, x, pass.shift);
        }
    }

    let start_x = pass.start_x as usize;
    let exclude_from = pass.exclude_from as usize;
    let exclude_to = pass.exclude_to as usize;
    let mut sums = FieldSums::default();
    for y in (2..height - 2).step_by(2) {
        let used = exclude_from == exclude_to || y < exclude_from || y > exclude_to;
        if !used {
            continue;
        }

        let pair = (y - 2) / 2;
        let matched_row = other_row + 2 * pair;
        let kept_row = current_row + 2 * pair;
        for x in start_x..width.saturating_sub(start_x) {
            let upper_map = map[matched_row * width + x];
            let lower_map = map[(matched_row + 2) * width + x];
            if upper_map == 0 && lower_map == 0 {
                continue;
            }

            let above = current.at(x, kept_row - 2);
            let kept = current.at(x, kept_row) << 2;
            let below = current.at(x, kept_row + 2);
            let centre = above + kept + below;
            let flags = upper_map | lower_map;

            let woven = 3 * (first.at(x, matched_row) + first.at(x, matched_row + 2));
            let difference = (woven - centre).abs();
            accumulate(&mut sums.first, difference, flags, pass.shift);

            let woven = 3 * (second.at(x, matched_row) + second.at(x, matched_row + 2));
            let difference = (woven - centre).abs();
            accumulate(&mut sums.second, difference, flags, pass.shift);
        }
    }

    (sums, map)
}
