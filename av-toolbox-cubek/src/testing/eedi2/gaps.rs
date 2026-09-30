use super::{Consensus, Grid, Scaled, Typo};

/// Whether gap pixel `column` of gap row `gap` touches the edge mask above or below it.
fn touches_edge(edges: &Grid, gap: i64, column: i64) -> bool {
    edges.at(gap, column) == 1 || edges.at(gap + 1, column) == 1
}

/// Whether two directions agree within `lim`, or either is missing.
fn agrees(first: i64, second: i64, second_missing: bool, lim: i64, peak: i64) -> bool {
    (first - second).abs() <= lim || first == peak || second_missing
}

/// Pass 10's direction at one gap pixel from the 3 directions above it and the 3 below.
///
/// The third agreement test compares the right direction above with the left one below, as the plugin does,
/// unless `typo` asks for the corrected test.
fn gap_direction(directions: &Grid, gap: i64, column: i64, scaled: &Scaled, typo: Typo) -> Option<i64> {
    let peak = scaled.peak;
    let above = |offset: i64| directions.at(gap, column + offset);
    let below = |offset: i64| directions.at(gap + 1, column + offset);

    let neighbours = [above(-1), above(0), above(1), below(-1), below(0), below(1)];
    let mut values: Vec<i64> = neighbours.into_iter().filter(|value| *value != peak).collect();
    let present = values.len() as i64;
    if present < 3 {
        return None;
    }

    let vote = Consensus::of(&mut values, scaled);
    let [
        above_left,
        above_centre,
        above_right,
        below_left,
        below_centre,
        below_right,
    ] = neighbours;
    let right_partner = match typo {
        Typo::Kept => below_left,
        Typo::Fixed => below_right,
    };
    let left_agrees = agrees(above_left, below_left, below_left == peak, vote.lim, peak);
    let centre_agrees = agrees(above_centre, below_centre, below_centre == peak, vote.lim, peak);
    let right_agrees = agrees(above_right, right_partner, below_right == peak, vote.lim, peak);
    let agreeing = i64::from(left_agrees) + i64::from(centre_agrees) + i64::from(right_agrees);
    if agreeing < 2 {
        return None;
    }

    if vote.count < present - 2 || vote.count < 2 {
        return None;
    }

    Some(vote.average())
}

/// Pass 10, the gap directions `X0` from the cleaned field directions, `peak` where none agree.
pub(super) fn gap_directions(directions: &Grid, edges: &Grid, scaled: &Scaled, typo: Typo) -> Grid {
    let gaps = directions.rows() - 1;
    let width = directions.width;

    let mut gap_map = Grid::filled(gaps, width, scaled.peak);
    for gap in 0..gaps {
        for column in 1..width - 1 {
            if !touches_edge(edges, gap, column) {
                continue;
            }

            if let Some(value) = gap_direction(directions, gap, column, scaled, typo) {
                gap_map.set(gap, column, value);
            }
        }
    }

    gap_map
}

/// The values of `gap_map` other than `peak` in the 3 by 3 block around a gap pixel, using only the gap
/// rows that exist.
fn gap_block_values(gap_map: &Grid, gap: i64, column: i64, peak: i64, with_centre: bool) -> Vec<i64> {
    let gaps = gap_map.rows();

    let mut values = Vec::with_capacity(9);
    for row in gap - 1..=gap + 1 {
        if !(0..gaps).contains(&row) {
            continue;
        }

        for column_offset in -1..=1 {
            let is_centre = row == gap && column_offset == 0;
            let value = gap_map.at(row, column + column_offset);
            if value != peak && (with_centre || !is_centre) {
                values.push(value);
            }
        }
    }

    values
}

/// Passes 11 and 16's `filter2x`, which keeps a gap direction only where at least 4 of its block agree.
pub(super) fn filter_gaps(gap_map: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let mut filtered = gap_map.clone();
    for gap in 0..gap_map.rows() {
        for column in 1..gap_map.width - 1 {
            if !touches_edge(edges, gap, column) {
                continue;
            }

            let mut values = gap_block_values(gap_map, gap, column, scaled.peak, true);
            if values.len() < 4 {
                filtered.set(gap, column, scaled.peak);
                continue;
            }

            let vote = Consensus::of(&mut values, scaled);
            let centre_missing = gap_map.at(gap, column) == scaled.peak;
            let value = if vote.count < 4 || (vote.count < 5 && centre_missing) {
                scaled.peak
            } else {
                vote.average()
            };
            filtered.set(gap, column, value);
        }
    }

    filtered
}

/// Passes 12 and 16's `expand2x`, which gives a gap pixel with no direction one where at least 5
/// neighbours agree.
pub(super) fn expand_gaps(gap_map: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let mut expanded = gap_map.clone();
    for gap in 0..gap_map.rows() {
        for column in 1..gap_map.width - 1 {
            let missing = gap_map.at(gap, column) == scaled.peak;
            if !missing || !touches_edge(edges, gap, column) {
                continue;
            }

            let mut values = gap_block_values(gap_map, gap, column, scaled.peak, false);
            if values.len() < 5 {
                continue;
            }

            let vote = Consensus::of(&mut values, scaled);
            if vote.count >= 5 {
                let average = vote.average();
                expanded.set(gap, column, average);
            }
        }
    }

    expanded
}

/// Rounds `value` to `value >> dropped` with round-to-nearest-even on the dropped bits.
fn round_dropping(value: i64, dropped: i64) -> i64 {
    if dropped <= 0 {
        return value;
    }

    let kept = value >> dropped;
    let guard = (value >> (dropped - 1)) & 1 == 1;
    let sticky = value & ((1 << (dropped - 1)) - 1) != 0;
    let round_up = guard && (sticky || kept & 1 == 1);

    kept + i64::from(round_up)
}

/// `FILLROUND(difference, length, step)`, the plugin's `trunc(step * (difference / length) + 0.5)` with
/// both the quotient and the product rounded to f32, in integers.
///
/// The quotient is found by long division to 24 bits and rounded to nearest even. The exact product is then
/// rounded to 24 bits the same way, and half is added before truncating.
pub(super) fn fill_round(difference: i64, length: i64, step: i64) -> i64 {
    if difference == 0 || step == 0 {
        return 0;
    }

    let magnitude = difference.abs();
    let mut quotient = magnitude / length;
    let mut remainder = magnitude % length;
    let mut quotient_shift = 0;
    while quotient < 1 << 23 {
        remainder *= 2;
        quotient *= 2;
        if remainder >= length {
            remainder -= length;
            quotient += 1;
        }

        quotient_shift += 1;
    }

    remainder *= 2;
    let guard = remainder >= length;
    if guard {
        remainder -= length;
    }

    let sticky = remainder != 0;
    let round_up = guard && (sticky || quotient & 1 == 1);
    let mantissa = quotient + i64::from(round_up);

    let product = step * mantissa;
    let product_bits = 64 - i64::from(product.leading_zeros());
    let dropped = (product_bits - 24).max(0);
    let product_mantissa = round_dropping(product, dropped);

    let exponent = dropped - quotient_shift;
    let nearest = if exponent >= 0 {
        product_mantissa << exponent
    } else {
        let half = 1 << (-exponent - 1);
        (product_mantissa + half) >> -exponent
    };

    if difference > 0 {
        nearest
    } else {
        -(nearest - 1).max(0)
    }
}

/// The value and bounds of one run of missing gap directions.
struct Run {
    back: i64,
    forward: i64,
    first: i64,
    last: i64,
}

/// The run holding gap pixel `column`, or `None` when it does not reach a direction on both sides.
///
/// `first` and `last` are the columns of the directions that bound it.
fn find_run(gap_map: &Grid, edges: &Grid, gap: i64, column: i64, peak: i64) -> Option<Run> {
    let mut first = column - 1;
    let mut back = None;
    while first > 0 {
        let value = gap_map.at(gap, first);
        if value != peak {
            back = Some(value);
            break;
        }

        if !touches_edge(edges, gap, first) {
            break;
        }

        first -= 1;
    }

    let mut last = column + 1;
    let mut forward = None;
    while last < gap_map.width {
        let value = gap_map.at(gap, last);
        if value != peak {
            forward = Some(value);
            break;
        }

        if !touches_edge(edges, gap, last) {
            break;
        }

        last += 1;
    }

    let back = back?;
    let forward = forward?;

    Some(Run {
        back,
        forward,
        first,
        last,
    })
}

/// The spread of the directions of gap row `row` over a run, or `None` when a column has no direction or
/// touches no edge.
fn neighbour_span(gap_map: &Grid, edges: &Grid, row: i64, run: &Run, peak: i64) -> Option<i64> {
    let mut lowest = i64::MAX;
    let mut highest = i64::MIN;
    for column in run.first..=run.last {
        let value = gap_map.at(row, column);
        let on_edge = touches_edge(edges, row, column);
        if value == peak || !on_edge {
            return None;
        }

        lowest = lowest.min(value);
        highest = highest.max(value);
    }

    Some(highest - lowest)
}

/// Pass 13 and 14's value for one missing gap pixel, stepping from the run's back direction toward its
/// forward one.
fn fill_value(gap_map: &Grid, edges: &Grid, gap: i64, column: i64, scaled: &Scaled) -> Option<i64> {
    let run = find_run(gap_map, edges, gap, column, scaled.peak)?;

    let span_above = if gap > 0 {
        neighbour_span(gap_map, edges, gap - 1, &run, scaled.peak)
    } else {
        None
    };
    let span_below = if gap < gap_map.rows() - 1 {
        neighbour_span(gap_map, edges, gap + 1, &run, scaled.peak)
    } else {
        None
    };

    let forward_reach = (run.forward - scaled.neutral).abs();
    let back_reach = (run.back - scaled.neutral).abs();
    let reach = forward_reach.max(back_reach);
    let spread_above = span_above.unwrap_or(0);
    let spread_below = span_below.unwrap_or(0);
    let threshold = (reach >> 2).max(scaled.eight).max(spread_above).max(spread_below);
    let lim = (reach >> (scaled.shift + 2)).min(6);
    let length = run.last - run.first;
    let difference = run.forward - run.back;

    let close = difference.abs() <= threshold;
    let short = length - 1 <= lim;
    let supported = span_above.is_some() || span_below.is_some();
    if !(close && (short || supported)) {
        return None;
    }

    let step = column - run.first - 1;
    let offset = fill_round(difference, length, step);
    Some(run.back + offset)
}

/// Passes 13 and 14, `fill`, which fills a run of missing gap directions between two that agree.
pub(super) fn fill_gaps(gap_map: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let mut filled = gap_map.clone();
    for gap in 0..gap_map.rows() {
        for column in 1..gap_map.width - 1 {
            let missing = gap_map.at(gap, column) == scaled.peak;
            if !missing || !touches_edge(edges, gap, column) {
                continue;
            }

            if let Some(value) = fill_value(gap_map, edges, gap, column, scaled) {
                filled.set(gap, column, value);
            }
        }
    }

    filled
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The plugin's own expression, with the quotient and product in f32 and the half added in f64.
    fn plugin_fill_round(difference: i64, length: i64, step: i64) -> i64 {
        let quotient = difference as f32 / length as f32;
        let product = step as f32 * quotient;
        let shifted = f64::from(product) + 0.5;
        shifted.trunc() as i64
    }

    #[test]
    fn fill_round_matches_the_plugins_float_expression() {
        for difference in -232..=232 {
            for length in 2..=128 {
                for step in 0..length - 1 {
                    let ours = fill_round(difference, length, step);
                    let plugin = plugin_fill_round(difference, length, step);
                    assert_eq!(ours, plugin, "{difference} over {length} at step {step}");
                }
            }
        }
    }

    /// Checks every reachable step of one difference and run length, returning how many it checked.
    fn check_every_step(difference: i64, length: i64) -> u64 {
        for step in 0..length - 1 {
            let ours = fill_round(difference, length, step);
            let plugin = plugin_fill_round(difference, length, step);
            assert_eq!(ours, plugin, "{difference} over {length} at step {step}");
        }

        (length - 1) as u64
    }

    /// Every difference up to `largest` that is a multiple of half of `length`, so `step * difference` lands
    /// on or next to a tie, and every `stride`-th difference besides, with both signs.
    fn differences(length: i64, largest: i64, stride: i64) -> Vec<i64> {
        let half = if length % 2 == 0 { length / 2 } else { length };
        let multiples = (0..=largest).step_by(half as usize);
        let strided = (1..=largest).step_by(stride as usize);

        let mut magnitudes: Vec<i64> = multiples.chain(strided).collect();
        magnitudes.sort_unstable();
        magnitudes.dedup();

        magnitudes
            .iter()
            .flat_map(|magnitude| [*magnitude, -*magnitude])
            .collect()
    }

    #[test]
    fn fill_round_matches_the_plugin_up_to_16_bits() {
        // 232 * 256 is the largest difference a fill can see, at 16 bits.
        let largest = 59392;

        let mut checked = 0;
        for length in 2..=64 {
            for difference in differences(length, largest, 61) {
                checked += check_every_step(difference, length);
            }
        }

        for length in [1000, 3837] {
            for difference in differences(length, largest, 997) {
                checked += check_every_step(difference, length);
            }
        }

        assert!(checked > 10_000_000, "only {checked} inputs were checked");
    }

    #[test]
    fn fill_round_keeps_the_plugins_tie_rounding() {
        // Reference section 4.3 gives -14.5 exactly, which the plugin's f32 quotient turns into -13.
        assert_eq!(fill_round(-232, 432, 27), -13);
    }

    #[test]
    fn fill_round_truncates_negative_steps_toward_zero() {
        // As reference section 1.14 gives, -1.2 becomes 0 and -1.6 becomes -1.
        assert_eq!(fill_round(-6, 5, 1), 0);
        assert_eq!(fill_round(-8, 5, 1), -1);
        assert_eq!(fill_round(6, 5, 1), 1);
    }
}
