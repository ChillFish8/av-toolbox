use super::{Grid, LeftDirection, Scaled};

/// The field rows and cleaned directions above and below one gap row.
struct GapRows<'a> {
    field: &'a Grid,
    directions: &'a Grid,
    gap: i64,
}

impl GapRows<'_> {
    fn up(&self, column: i64) -> i64 {
        self.field.at(self.gap, column)
    }

    fn down(&self, column: i64) -> i64 {
        self.field.at(self.gap + 1, column)
    }

    fn up_direction(&self, column: i64) -> i64 {
        self.directions.at(self.gap, column)
    }

    fn down_direction(&self, column: i64) -> i64 {
        self.directions.at(self.gap + 1, column)
    }

    fn width(&self) -> i64 {
        self.field.width
    }

    fn vertical(&self, column: i64) -> i64 {
        (self.up(column) + self.down(column) + 1) >> 1
    }
}

/// Whether the 3 samples above and below a gap pixel vary too little to follow a direction.
///
/// A negative variance, possible above 8 bits, is not low, as with the edge mask.
fn has_low_variance(rows: &GapRows, column: i64, scaled: &Scaled) -> bool {
    let mut sum = 0;
    let mut sum_of_squares = 0;
    for tap in -1..=1 {
        for sample in [rows.up(column + tap), rows.down(column + tap)] {
            let reduced = sample >> scaled.shift;
            sum += sample;
            sum_of_squares += reduced * reduced;
        }
    }

    let reduced_sum = sum >> scaled.shift;
    let variance = 6 * sum_of_squares - reduced_sum * reduced_sum;
    (0..576).contains(&variance)
}

/// Whether `row` of the field dips at `column` below both sides by more than `three`.
fn is_valley(field: &Grid, row: i64, column: i64, three: i64) -> bool {
    let centre = field.at(row, column);
    let two_before = field.at(row, column - 2);
    let one_before = field.at(row, column - 1);
    let one_after = field.at(row, column + 1);
    let two_after = field.at(row, column + 2);
    let before = two_before.max(one_before);
    let after = two_after.max(one_after);

    centre < before - three && centre < after - three
}

/// Whether `row` of the field peaks at `column` above both sides by more than `three`.
fn is_ridge(field: &Grid, row: i64, column: i64, three: i64) -> bool {
    let centre = field.at(row, column);
    let two_before = field.at(row, column - 2);
    let one_before = field.at(row, column - 1);
    let one_after = field.at(row, column + 1);
    let two_after = field.at(row, column + 2);
    let before = two_before.min(one_before);
    let after = two_after.min(one_after);

    centre > before + three && centre > after + three
}

/// Whether the rows above and below a gap pixel both hold a valley there, or both a ridge.
fn is_extremum(rows: &GapRows, column: i64, three: i64) -> bool {
    let field = rows.field;
    let up = rows.gap;
    let down = rows.gap + 1;
    let valleys = is_valley(field, up, column, three) && is_valley(field, down, column, three);
    let ridges = is_ridge(field, up, column, three) && is_ridge(field, down, column, three);

    valleys || ridges
}

/// The cost of pairing the samples above a gap pixel with those below shifted by `shift`, both ways.
fn crossing_cost(rows: &GapRows, column: i64, shift: i64) -> i64 {
    (-1..=1)
        .map(|tap| {
            let up_to_down = (rows.up(column + tap) - rows.down(column - shift + tap)).abs();
            let down_to_up = (rows.down(column + tap) - rows.up(column + shift + tap)).abs();
            up_to_down + down_to_up
        })
        .sum()
}

/// A direction agreeing with the gap pixel's own within its `lim`.
fn is_near(value: i64, direction: i64, lim: i64, peak: i64) -> bool {
    value != peak && (value - direction).abs() <= lim
}

fn within(first: i64, second: i64, lim: i64) -> bool {
    (first - second).abs() <= lim
}

/// Whether the cleaned directions along `shift` back up the gap pixel's `direction`.
fn is_supported(rows: &GapRows, column: i64, shift: i64, direction: i64, lim: i64, peak: i64) -> bool {
    let near_above = (-1..=1).any(|tap| {
        let value = rows.up_direction(column + tap + shift);
        is_near(value, direction, lim, peak)
    });
    let near_below = (-1..=1).any(|tap| {
        let value = rows.down_direction(column + tap - shift);
        is_near(value, direction, lim, peak)
    });

    near_above && near_below
}

/// Whether a direction above agrees with either direction of the half-pixel pair below.
fn pairs_below(above: i64, below_down: i64, below_up: i64, lim: i64, peak: i64) -> bool {
    let close = within(above, below_down, lim) || within(above, below_up, lim);

    close && above != peak
}

/// Whether the cleaned directions at the half-pixel pair of `shift` agree with each other and with the
/// gap pixel's `direction`.
fn is_matched(rows: &GapRows, column: i64, shift: i64, direction: i64, lim: i64, peak: i64) -> bool {
    let half_down = shift / 2;
    let half_up = (shift + 1) / 2;
    let above_down = rows.up_direction(column + half_down);
    let above_up = rows.up_direction(column + half_up);
    let below_down = rows.down_direction(column - half_down);
    let below_up = rows.down_direction(column - half_up);

    let pair_a = pairs_below(above_down, below_down, below_up, lim, peak);
    let pair_b = pairs_below(above_up, below_down, below_up, lim, peak);
    let above_matches = within(direction, above_down, lim) || within(direction, above_up, lim);
    let below_matches = within(direction, below_down, lim) || within(direction, below_up, lim);

    (pair_a || pair_b) && above_matches && below_matches
}

/// The average of the half-pixel pair of `shift` above and below a gap pixel.
fn pair_average(rows: &GapRows, column: i64, shift: i64) -> i64 {
    let half_down = shift / 2;
    let half_up = (shift + 1) / 2;
    let above = rows.up(column + half_down) + rows.up(column + half_up);
    let below = rows.down(column - half_down) + rows.down(column - half_up);

    (above + below + 2) >> 2
}

/// The directional search of reference section 1.15.2, within 2 pixels of `direction`, as the rebuilt
/// value and the shift it took.
fn directional_pick(
    rows: &GapRows,
    column: i64,
    direction: i64,
    lim: i64,
    scaled: &Scaled,
) -> Option<(i64, i64)> {
    let peak = scaled.peak;
    let rounded = (direction - scaled.neutral + (1 << (scaled.shift + 1))) >> (scaled.shift + 2);
    let reach = (column - 1).min(rows.width() - 2 - column);
    let first = (rounded - 2).clamp(-reach, reach);
    let last = (rounded + 2).clamp(-reach, reach);

    let mut best = scaled.nt8;
    let mut pick = None;
    for shift in first..=last {
        let cost = crossing_cost(rows, column, shift);
        if cost >= best {
            continue;
        }

        if !is_supported(rows, column, shift, direction, lim, peak) {
            continue;
        }

        let half_down = shift / 2;
        let half_difference: i64 = (-1..=1)
            .map(|tap| (rows.up(column + half_down + tap) - rows.down(column - half_down + tap)).abs())
            .sum();
        if half_difference >= scaled.nt4 {
            continue;
        }

        if !is_matched(rows, column, shift, direction, lim, peak) {
            continue;
        }

        let value = pair_average(rows, column, shift);
        best = cost;
        pick = Some((value, shift));
    }

    pick
}

/// The fallback search of reference section 1.15.3 within `dt`, as the rebuilt value and the shift it took.
///
/// It only takes a value between the samples directly above and below.
fn fallback_pick(rows: &GapRows, column: i64, scaled: &Scaled) -> Option<(i64, i64)> {
    let reach = scaled.dt.min(column - 1).min(rows.width() - 2 - column);
    let up = rows.up(column);
    let down = rows.down(column);
    let lowest = up.min(down);
    let highest = up.max(down);

    let mut best = scaled.nt7;
    let mut pick = None;
    for shift in -reach..=reach {
        let half_down = shift / 2;
        let half_up = (shift + 1) / 2;
        let above = rows.up(column + half_down) + rows.up(column + half_up);
        let below = rows.down(column - half_down) + rows.down(column - half_up);
        let crossing = crossing_cost(rows, column, shift);
        let cost = (above - below).abs() + crossing;
        if cost >= best {
            continue;
        }

        let candidate = (above + below + 2) >> 2;
        if (lowest..=highest).contains(&candidate) {
            best = cost;
            pick = Some((candidate, shift));
        }
    }

    pick
}

/// The rebuilt value and new direction at a gap pixel whose `direction` is not isolated.
fn rebuild(rows: &GapRows, column: i64, direction: i64, lim: i64, scaled: &Scaled) -> (i64, i64) {
    let vertical = rows.vertical(column);
    if lim < scaled.nine && has_low_variance(rows, column, scaled) {
        return (vertical, scaled.peak);
    }

    let inside = column > 1 && column < rows.width() - 2;
    if inside && is_extremum(rows, column, scaled.three) {
        return (vertical, scaled.neutral);
    }

    if let Some((value, shift)) = directional_pick(rows, column, direction, lim, scaled) {
        let new_direction = scaled.map_value(shift);
        return (value, new_direction);
    }

    match fallback_pick(rows, column, scaled) {
        Some((value, shift)) => (value, scaled.map_value(shift)),
        None => (vertical, scaled.neutral),
    }
}

/// Pass 15, the lattice, as the directions it used `L` and the rebuilt gap rows `R`.
///
/// Along each gap row it runs left to right, and a direction differing from both neighbours by more than its
/// `LIM` is dropped. The left neighbour is the direction already written for the column before.
pub(super) fn lattice(
    field: &Grid,
    directions: &Grid,
    gap_map: &Grid,
    scaled: &Scaled,
    left: LeftDirection,
) -> (Grid, Grid) {
    let gaps = gap_map.rows();
    let width = gap_map.width;

    let mut used = gap_map.clone();
    let mut rebuilt = Grid::filled(gaps, width, 0);
    for gap in 0..gaps {
        let rows = GapRows {
            field,
            directions,
            gap,
        };
        for column in 0..width {
            let direction = gap_map.at(gap, column);
            let vertical = rows.vertical(column);
            if direction == scaled.peak {
                rebuilt.set(gap, column, vertical);
                continue;
            }

            assert!(
                column > 0 && column < width - 1,
                "the gap directions hold a direction at edge column {column}"
            );

            let lim = scaled.lim(direction);
            let left_direction = match left {
                LeftDirection::Updated => used.at(gap, column - 1),
                LeftDirection::Original => gap_map.at(gap, column - 1),
            };
            let right_direction = gap_map.at(gap, column + 1);
            let isolated =
                (direction - left_direction).abs() > lim && (direction - right_direction).abs() > lim;

            let (value, new_direction) = if isolated {
                (vertical, scaled.neutral)
            } else {
                rebuild(&rows, column, direction, lim, scaled)
            };
            rebuilt.set(gap, column, value);
            used.set(gap, column, new_direction);
        }
    }

    (used, rebuilt)
}

/// Pass 16, which puts back the vertical average wherever the re-filtered direction disagrees with the one
/// the lattice used, unless that was `peak` or `neutral`.
pub(super) fn post_process(
    field: &Grid,
    used: &Grid,
    refiltered: &Grid,
    rebuilt: &Grid,
    scaled: &Scaled,
) -> Grid {
    let mut processed = rebuilt.clone();
    for gap in 0..used.rows() {
        for column in 0..used.width {
            let fresh = refiltered.at(gap, column);
            let kept = used.at(gap, column);
            let lim = scaled.lim(fresh);
            let disagrees = (fresh - kept).abs() > lim;
            let turned = kept != scaled.peak && kept != scaled.neutral;
            if disagrees && turned {
                let up = field.at(gap, column);
                let down = field.at(gap + 1, column);
                processed.set(gap, column, (up + down + 1) >> 1);
            }
        }
    }

    processed
}
