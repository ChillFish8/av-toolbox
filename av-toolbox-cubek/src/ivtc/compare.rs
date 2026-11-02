use std::iter::Sum;
use std::ops::{Add, AddAssign};

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::combed::{clear, tap};
use super::difference::{SEGMENT, segments};
use crate::{DevicePlane, Dispatch, assert_plane};

/// How many sums one field compare writes per row pair and segment.
///
/// They are the first match's combing, motion and isolated sums, then the same for the second match.
pub const FIELD_SUMS: usize = 6;

/// The three sums VFM's field compare gathers for one match.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MatchSums {
    pub combing: u64,
    pub motion: u64,
    pub isolated: u64,
}

impl Add for MatchSums {
    type Output = MatchSums;

    fn add(self, other: MatchSums) -> MatchSums {
        MatchSums {
            combing: self.combing + other.combing,
            motion: self.motion + other.motion,
            isolated: self.isolated + other.isolated,
        }
    }
}

impl AddAssign for MatchSums {
    fn add_assign(&mut self, other: MatchSums) {
        *self = *self + other;
    }
}

impl Sum for MatchSums {
    fn sum<I: Iterator<Item = MatchSums>>(sums: I) -> MatchSums {
        sums.fold(MatchSums::default(), Add::add)
    }
}

/// The sums of both matches of one field compare, VFM's `match1` first.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FieldSums {
    pub first: MatchSums,
    pub second: MatchSums,
}

impl Add for FieldSums {
    type Output = FieldSums;

    fn add(self, other: FieldSums) -> FieldSums {
        FieldSums {
            first: self.first + other.first,
            second: self.second + other.second,
        }
    }
}

impl AddAssign for FieldSums {
    fn add_assign(&mut self, other: FieldSums) {
        *self = *self + other;
    }
}

impl Sum for FieldSums {
    fn sum<I: Iterator<Item = FieldSums>>(sums: I) -> FieldSums {
        sums.fold(FieldSums::default(), Add::add)
    }
}

impl FieldSums {
    /// Adds the partial sums [compare_fields] writes, [FIELD_SUMS] at a time, into one set of sums.
    pub fn from_partials(partials: &[u32]) -> FieldSums {
        assert!(
            partials.len().is_multiple_of(FIELD_SUMS),
            "{} partial sums do not split into groups of {FIELD_SUMS}",
            partials.len(),
        );

        let mut totals = [0u64; FIELD_SUMS];
        let (groups, _) = partials.as_chunks::<FIELD_SUMS>();
        for group in groups {
            for (total, partial) in totals.iter_mut().zip(group) {
                *total += u64::from(*partial);
            }
        }

        let [
            first_combing,
            first_motion,
            first_isolated,
            second_combing,
            second_motion,
            second_isolated,
        ] = totals;
        FieldSums {
            first: MatchSums {
                combing: first_combing,
                motion: first_motion,
                isolated: first_isolated,
            },
            second: MatchSums {
                combing: second_combing,
                motion: second_motion,
                isolated: second_isolated,
            },
        }
    }
}

/// The settings of one plane's field compare.
///
/// `first` and `second` are VFM's `match1` and `match2`, each 0, 1 or 2 for the previous, current or next
/// frame. `field` is VFM's field. Row pairs whose loop row lies between `exclude_from` and `exclude_to` are
/// left out of the sums, unless the two are equal. Each row sums the columns from `start_x` up to
/// `width - start_x`. `shift` is the bit depth less 8.
#[derive(Debug, Clone, Copy)]
pub struct ComparePass {
    pub first: u32,
    pub second: u32,
    pub field: u32,
    pub exclude_from: u32,
    pub exclude_to: u32,
    pub start_x: u32,
    pub shift: u32,
}

/// How many row pairs a field compare sums, the rows `2, 4, ..` below `height - 2`.
fn compare_rows(height: u32) -> u32 {
    height.saturating_sub(3) / 2
}

/// How many `u32` partial sums [compare_fields] writes for one plane.
pub fn compare_sums_len(width: u32, height: u32) -> usize {
    compare_rows(height) as usize * segments(width) as usize * FIELD_SUMS
}

#[cube(launch_unchecked)]
fn field_differences_kernel(
    first: &[f32],
    second: &[f32],
    differences: &mut [u32],
    width: u32,
    start_row: u32,
) {
    if ABSOLUTE_POS >= differences.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = position % width;
    let row = position / width;
    let source_row = start_row + 2 * row;
    let first_value = tap(first, width, x, source_row);
    let second_value = tap(second, width, x, source_row);
    let difference = first_value - second_value;

    differences[ABSOLUTE_POS] = u32::cast_from(i32::abs(difference));
}

#[cube]
fn difference_at(differences: &[u32], width: u32, column: u32, row: u32) -> u32 {
    differences[(row * width + column) as usize]
}

/// `buildDiffMap`'s value for difference row `row` and column `x`, 0, 1, 3 or 5.
///
/// The loops and their early exits follow `buildDiffMap`.
#[cube]
fn map_value(differences: &[u32], width: u32, height: u32, row: u32, x: u32, shift: u32) -> u32 {
    let low = 3u32 << shift;
    let high = 19u32 << shift;
    let difference = difference_at(differences, width, x, row);
    let mut value = 0u32;

    if difference > low {
        let mut count = 0u32;
        let mut column = x - 1;
        while column < x + 2 && count < 2 {
            if difference_at(differences, width, column, row - 1) > low {
                count += 1u32;
            }

            if difference_at(differences, width, column, row) > low {
                count += 1u32;
            }

            if difference_at(differences, width, column, row + 1) > low {
                count += 1u32;
            }

            column += 1u32;
        }

        if count > 1 {
            value += 1;
            if difference > high {
                value += wide_value(differences, width, height, row, x, high);
            }
        }
    }

    value
}

/// What `buildDiffMap` adds to a sample above both thresholds, 0, 2 or 4.
#[cube]
fn wide_value(differences: &[u32], width: u32, height: u32, row: u32, x: u32, high: u32) -> u32 {
    let mut upper = false;
    let mut lower = false;
    let mut count = 0u32;
    let mut column = x - 1;
    while column < x + 2 && count < 6 {
        if difference_at(differences, width, column, row - 1) > high {
            count += 1u32;
            upper = true;
        }

        if difference_at(differences, width, column, row) > high {
            count += 1u32;
        }

        if difference_at(differences, width, column, row + 1) > high {
            count += 1u32;
            lower = true;
        }

        column += 1u32;
    }

    let mut added = 0u32;
    if count > 3 {
        if upper && lower {
            added = 2;
        } else {
            let y = 2 * row;
            let mut upper_two = false;
            let mut lower_two = false;
            let mut first_column = 0u32;
            if x > 4 {
                first_column = x - 4;
            }

            let end_column = u32::min(x + 5, width);

            // The two outer rows are only read away from the edge rows, so neither read leaves the
            // differences.
            for column in first_column..end_column {
                if y != 2 {
                    let two_above = difference_at(differences, width, column, row - 2);
                    if two_above > high {
                        upper_two = true;
                    }
                }

                if difference_at(differences, width, column, row - 1) > high {
                    upper = true;
                }

                if difference_at(differences, width, column, row + 1) > high {
                    lower = true;
                }

                if y != height - 4 {
                    let two_below = difference_at(differences, width, column, row + 2);
                    if two_below > high {
                        lower_two = true;
                    }
                }
            }

            if (upper && (lower || upper_two)) || (lower && (upper || lower_two)) {
                added = 2;
            } else if count > 5 {
                added = 4;
            }
        }
    }

    added
}

#[cube(launch_unchecked)]
fn diff_map_kernel(
    differences: &[u32],
    map: &mut [u32],
    width: u32,
    height: u32,
    map_start: u32,
    shift: u32,
) {
    let position = ABSOLUTE_POS as u32;
    let rows = (height - 3) / 2;
    if position >= rows * width {
        terminate!();
    }

    let x = position % width;
    let row = position / width + 1;
    if x == 0 || x == width - 1 {
        terminate!();
    }

    let map_row = map_start + 2 * (row - 1);
    let value = map_value(differences, width, height, row, x, shift);

    map[(map_row * width + x) as usize] = value;
}

/// The difference of one match's woven rows from the kept field's centre, as `compareFieldsSlow` measures it.
#[cube]
fn match_difference(frame: &[f32], width: u32, x: u32, matched_row: u32, centre: i32) -> u32 {
    let upper = tap(frame, width, x, matched_row);
    let lower = tap(frame, width, x, matched_row + 2);
    let woven = 3 * (upper + lower);

    u32::cast_from(i32::abs(woven - centre))
}

#[cube(launch_unchecked)]
#[expect(
    clippy::too_many_arguments,
    reason = "each argument is a distinct input of the launch"
)]
fn field_sums_kernel(
    first: &[f32],
    second: &[f32],
    current: &[f32],
    map: &[u32],
    sums: &mut [u32],
    width: u32,
    field: u32,
    exclude_from: u32,
    exclude_to: u32,
    start_x: u32,
    shift: u32,
    segments: u32,
) {
    let unit = ABSOLUTE_POS as u32;
    if unit * 6 >= sums.len() as u32 {
        terminate!();
    }

    let pair = unit / segments;
    let segment = unit % segments;
    let y = 2 + 2 * pair;
    let mut other_row = 2u32;
    if field == 1 {
        other_row = 1u32;
    }

    let matched_row = other_row + 2 * pair;
    let kept_row = 3 - field + 2 * pair;
    let combing_threshold = 23u32 << shift;
    let motion_threshold = 42u32 << shift;

    let mut first_combing = 0u32;
    let mut first_motion = 0u32;
    let mut first_isolated = 0u32;
    let mut second_combing = 0u32;
    let mut second_motion = 0u32;
    let mut second_isolated = 0u32;

    let used = exclude_from == exclude_to || y < exclude_from || y > exclude_to;
    if used {
        let start = u32::max(segment * SEGMENT, start_x);
        let end = u32::min(segment * SEGMENT + SEGMENT, width - start_x);
        for x in start..end {
            let upper_map = map[(matched_row * width + x) as usize];
            let lower_map = map[((matched_row + 2) * width + x) as usize];
            if upper_map > 0 || lower_map > 0 {
                let above = tap(current, width, x, kept_row - 2);
                let kept = tap(current, width, x, kept_row) << 2;
                let below = tap(current, width, x, kept_row + 2);
                let centre = above + kept + below;
                let flags = upper_map | lower_map;
                let combed = (flags & 1) != 0;
                let moving = (flags & 2) != 0;
                let isolated = (flags & 4) != 0;

                let difference = match_difference(first, width, x, matched_row, centre);
                if difference > combing_threshold && combed {
                    first_combing += difference;
                }

                if difference > motion_threshold {
                    if moving {
                        first_motion += difference;
                    }

                    if isolated {
                        first_isolated += difference;
                    }
                }

                let difference = match_difference(second, width, x, matched_row, centre);
                if difference > combing_threshold && combed {
                    second_combing += difference;
                }

                if difference > motion_threshold {
                    if moving {
                        second_motion += difference;
                    }

                    if isolated {
                        second_isolated += difference;
                    }
                }
            }
        }
    }

    let base = (unit * 6) as usize;
    sums[base] = first_combing;
    sums[base + 1] = first_motion;
    sums[base + 2] = first_isolated;
    sums[base + 3] = second_combing;
    sums[base + 4] = second_motion;
    sums[base + 5] = second_isolated;
}

/// The planes one field compare reads and writes.
///
/// `frames` are the previous, current and next frames' f32 code values. `differences` holds `width` by
/// `height / 2` `u32` values, and `map` one `u32` per sample. `sums` holds [compare_sums_len] `u32` values.
pub struct ComparePlanes<'a> {
    pub frames: [DevicePlane<'a>; 3],
    pub differences: DevicePlane<'a>,
    pub map: DevicePlane<'a>,
    pub sums: DevicePlane<'a>,
}

/// Runs VFM's `compareFieldsSlow` on one plane, building its difference map and partial sums.
///
/// The map gets VFM's 0, 1, 3 or 5 per sample. For each row pair and [SEGMENT] of columns, `sums` gets the
/// [FIELD_SUMS] sums, which [FieldSums::from_partials] adds up. An excluded row pair's sums are 0. The plane
/// must have an even height of at least 4 rows.
pub fn compare_fields(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: ComparePlanes<'_>,
    pass: ComparePass,
) {
    assert!(
        pass.first < 3 && pass.second < 3,
        "the field compare takes matches 0, 1 or 2, not {} and {}",
        pass.first,
        pass.second,
    );
    assert!(pass.field < 2, "a field is 0 or 1, not {}", pass.field);

    let width = planes.map.width();
    let height = planes.map.height();
    assert!(
        height >= 4 && height.is_multiple_of(2),
        "the field compare needs an even height of at least 4, the plane has {height}",
    );
    assert!(
        2 * pass.start_x <= width,
        "the {width} column plane is narrower than twice start_x {}",
        pass.start_x,
    );
    for (index, frame) in planes.frames.iter().enumerate() {
        let name = format!("frame {index}");
        assert_plane(frame, width, height, &name);
    }
    assert_plane(&planes.differences, width, height / 2, "differences");
    assert_plane(&planes.map, width, height, "map");

    let sums_len = compare_sums_len(width, height);
    assert!(
        planes.sums.samples() == sums_len,
        "the sums hold {} values, expected {sums_len}",
        planes.sums.samples(),
    );

    let first = planes.frames[pass.first as usize];
    let second = planes.frames[pass.second as usize];
    let current = planes.frames[1];

    // `buildDiffMap` differences rows `field`, `field + 2`, and so on, and maps them onto the same rows.
    let difference_units = planes.differences.samples();
    let cubes = dispatch.cube_count(difference_units);
    let unit = dispatch.cube_dim();

    // Safety: every frame holds `width * height` f32 values, and each unit reads one sample of row
    // `field + 2 * row`, below `height`, and writes only its own difference.
    unsafe {
        field_differences_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            first.array(),
            second.array(),
            planes.differences.array(),
            width,
            pass.field,
        );
    }

    clear(client, dispatch, planes.map);

    let map_units = compare_rows(height) as usize * width as usize;
    let cubes = dispatch.cube_count(map_units);
    let unit = dispatch.cube_dim();
    let map_start = 2 + pass.field;

    // Safety: each unit reads difference rows within two of its own, which the edge-row checks keep inside
    // the `height / 2` rows, and writes only its own map sample.
    unsafe {
        diff_map_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            planes.differences.array(),
            planes.map.array(),
            width,
            height,
            map_start,
            pass.shift,
        );
    }

    let segment_count = segments(width);
    let sum_units = sums_len / FIELD_SUMS;
    let cubes = dispatch.cube_count(sum_units);
    let unit = dispatch.cube_dim();

    // Safety: each unit reads frame and map rows between `1 - field + 2 * pair` and `5 - field + 2 * pair`,
    // inside the plane for every summed pair, and writes only its own six sums.
    unsafe {
        field_sums_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            first.array(),
            second.array(),
            current.array(),
            planes.map.array(),
            planes.sums.array(),
            width,
            pass.field,
            pass.exclude_from,
            pass.exclude_to,
            pass.start_x,
            pass.shift,
            segment_count,
        );
    }
}
