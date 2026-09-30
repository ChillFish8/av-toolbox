// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// EEDI2's `limlut` at `index`, the agreement threshold for a direction of `index` whole pixels.
///
/// Index 31 and above hold -1, so a threshold read there is negative and every `<=` test against it fails.
#[cube]
fn limlut(index: f32) -> f32 {
    let mut threshold = f32::new(-1.0f32);

    if index < 2.0 {
        threshold = 6.0;
    } else if index < 4.0 {
        threshold = 7.0;
    } else if index < 6.0 {
        threshold = 8.0;
    } else if index < 9.0 {
        threshold = 9.0;
    } else if index < 11.0 {
        threshold = 10.0;
    } else if index < 13.0 {
        threshold = 11.0;
    } else if index < 31.0 {
        threshold = 12.0;
    }

    threshold
}

/// How direction map values are scaled on a plane.
///
/// `reduce_quarter` is `2^(6 - bits)`, which turns a floor multiplication into a right shift by `bits - 6`.
#[derive(CubeType)]
pub(super) struct MapScale {
    pub(super) peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
}

#[cube]
impl MapScale {
    pub(super) fn new(peak: f32, neutral: f32, unit: f32, reduce_quarter: f32) -> MapScale {
        MapScale {
            peak,
            neutral,
            unit,
            reduce_quarter,
        }
    }
}

/// `LIM(value)`, the agreement threshold for the direction map value `value`.
#[cube]
fn lim(value: f32, scale: &MapScale) -> f32 {
    let distance = f32::abs(value - scale.neutral);
    let index = f32::floor(distance * scale.reduce_quarter);

    limlut(index) * scale.unit
}

/// `RA(total, count)`, the plugin's float average rounded half up, as the integer
/// `floor((2 * total + count) / (2 * count))`.
///
/// `total` is a non-negative integer, so the division is exact in u32 where a float division may not be.
#[cube]
pub(super) fn round_average(total: f32, count: u32) -> f32 {
    let total = total as u32;
    let numerator = 2 * total + count;
    let denominator = 2 * count;

    (numerator / denominator) as f32
}

/// The value at position `rank` of the first `length` values once sorted ascending.
#[cube]
pub(super) fn ranked(values: &Array<f32>, length: u32, rank: u32) -> f32 {
    let mut chosen = 0.0f32;

    for candidate_slot in 0..length {
        let candidate = values[candidate_slot as usize];
        let mut below = 0u32;
        let mut equal = 0u32;

        for other_slot in 0..length {
            let other = values[other_slot as usize];

            if other < candidate {
                below += 1;
            } else if other == candidate {
                equal += 1;
            }
        }

        if below <= rank && rank < below + equal {
            chosen = candidate;
        }
    }

    chosen
}

/// The median of `length` values, averaging the middle two rounded up when `length` is even.
#[cube]
fn median(values: &Array<f32>, length: u32) -> f32 {
    let upper = ranked(values, length, length / 2);
    let is_odd = length % 2 == 1;
    let mut mid = upper;

    if !is_odd {
        let lower = ranked(values, length, length / 2 - 1);
        mid = f32::floor((lower + upper + 1.0) * 0.5);
    }

    mid
}

/// The median of some direction map values, and the count and sum of the values that agree with it.
#[derive(CubeType)]
pub(super) struct Consensus {
    pub(super) mid: f32,
    pub(super) lim: f32,
    pub(super) count: u32,
    pub(super) sum: f32,
}

/// Keeps the first `length` values, none of them `peak`, that lie within `LIM(mid)` of their median.
#[cube]
pub(super) fn consensus(values: &Array<f32>, length: u32, scale: &MapScale) -> Consensus {
    let mid = median(values, length);
    let threshold = lim(mid, scale);
    let mut count = 0u32;
    let mut sum = 0.0f32;

    for slot in 0..length {
        let value = values[slot as usize];

        if f32::abs(value - mid) <= threshold {
            count += 1;
            sum += value;
        }
    }

    Consensus {
        mid,
        lim: threshold,
        count,
        sum,
    }
}

/// Appends `value` to the first `length` slots of `values` unless it is `peak`, giving the new length.
#[cube]
pub(super) fn push_present(values: &mut Array<f32>, length: u32, value: f32, peak: f32) -> u32 {
    let mut new_length = length;

    if value != peak {
        values[length as usize] = value;
        new_length += 1;
    }

    new_length
}

/// Collects the values of `map` other than `peak` in rows `first_row` to `last_row` and the 3 columns
/// around `column`, skipping the centre unless `with_centre` is set. Gives how many it collected.
#[cube]
pub(super) fn gather_block(
    map: &Array<f32>,
    width: u32,
    first_row: u32,
    last_row: u32,
    row: u32,
    column: u32,
    peak: f32,
    with_centre: bool,
    values: &mut Array<f32>,
) -> u32 {
    let mut length = 0u32;

    for map_row in first_row..last_row + 1 {
        for step in 0..3u32 {
            let map_column = column + step - 1;
            let is_centre = map_row == row && step == 1;
            let value = map[(map_row * width + map_column) as usize];

            if with_centre || !is_centre {
                length = push_present(values, length, value, peak);
            }
        }
    }

    length
}

/// The filter of passes 7 and 11, keeping a direction only where at least 4 of its block agree.
#[cube]
pub(super) fn filter_vote(values: &Array<f32>, length: u32, centre: f32, scale: &MapScale) -> f32 {
    let mut result = scale.peak;

    if length >= 4 {
        let vote = consensus(values, length, scale);
        let centre_missing = centre == scale.peak;
        let rejected = vote.count < 4 || (vote.count < 5 && centre_missing);

        if !rejected {
            result = round_average(vote.sum + vote.mid, vote.count + 1);
        }
    }

    result
}

/// The expansion of passes 8 and 12, giving a pixel with no direction one where at least 5 neighbours
/// agree.
#[cube]
pub(super) fn expand_vote(values: &Array<f32>, length: u32, scale: &MapScale) -> f32 {
    let mut result = scale.peak;

    if length >= 5 {
        let vote = consensus(values, length, scale);

        if vote.count >= 5 {
            result = round_average(vote.sum + vote.mid, vote.count + 1);
        }
    }

    result
}
