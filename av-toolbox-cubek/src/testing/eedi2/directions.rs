use super::{Consensus, Grid, Scaled, quarter_lim};

/// The sum of absolute differences of 3 samples, row `first` around `column` against row `second` shifted
/// by `shift` columns.
fn window_difference(field: &Grid, column: i64, first: i64, second: i64, shift: i64) -> i64 {
    (-1..=1)
        .map(|tap| {
            let own = field.at(first, column + tap);
            let other = field.at(second, column + tap + shift);
            (own - other).abs()
        })
        .sum()
}

/// Whether any of the 3 mask pixels of `row` centred on `column` is set.
fn any_edge(edges: &Grid, row: i64, column: i64) -> bool {
    (-1..=1).any(|tap| edges.at(row, column + tap) == 1)
}

/// The lowest cost one of pass 6's five windows has found and the direction that found it.
struct Best {
    cost: i64,
    direction: Option<i64>,
}

impl Best {
    fn starting_at(cost: i64) -> Self {
        Self {
            cost,
            direction: None,
        }
    }

    /// Takes `direction` when `cost` is strictly lower, so the first direction wins a tie.
    fn offer(&mut self, cost: i64, direction: i64) {
        if cost < self.cost {
            self.cost = cost;
            self.direction = Some(direction);
        }
    }
}

/// The direction the found winners agree on, or 0 when fewer than two agree.
fn agreed_direction(mut found: Vec<i64>) -> i64 {
    let length = found.len();
    if length <= 1 {
        return 0;
    }

    found.sort_unstable();
    let mid = if length % 2 == 1 {
        found[length / 2]
    } else {
        (found[length / 2 - 1] + found[length / 2] + 1) / 2
    };
    let lim = quarter_lim(mid);

    let agreeing = found.iter().filter(|direction| (**direction - mid).abs() <= lim);
    let count = agreeing.clone().count() as i64;
    let sum: i64 = agreeing.sum();
    if count > 1 { sum / count } else { 0 }
}

/// Pass 6's direction at one pixel, searching every shift within `maxd` that stays inside the row.
fn raw_direction(field: &Grid, edges: &Grid, row: i64, column: i64, scaled: &Scaled) -> i64 {
    let last_row = field.rows() - 1;
    let reach = scaled.maxd.min(column - 1).min(field.width - 2 - column);
    let centre = field.at(row, column);
    let below = field.at(row + 1, column);
    let above = field.at(row - 1, column);
    let vertical = (centre - below).abs() + (centre - above).abs();

    let start_a = scaled.nt19.min(9 * vertical);
    let start_b = scaled.nt13.min(6 * vertical);
    let mut best_a = Best::starting_at(start_a);
    let mut best_b = Best::starting_at(start_b);
    let mut best_c = Best::starting_at(best_a.cost);
    let mut best_d = Best::starting_at(best_b.cost);
    let mut best_e = Best::starting_at(best_b.cost);

    for shift in -reach..=reach {
        let edge_above = row == 1 || any_edge(edges, row - 1, column + shift);
        let edge_below = row == last_row - 1 || any_edge(edges, row + 1, column - shift);
        if !(edge_above && edge_below) {
            continue;
        }

        let own_to_below = window_difference(field, column, row, row + 1, -shift);
        let own_to_above = window_difference(field, column, row, row - 1, shift);
        let above_to_own = window_difference(field, column, row - 1, row, -shift);
        let below_to_own = window_difference(field, column, row + 1, row, shift);
        let cost = own_to_below + own_to_above + above_to_own + below_to_own;
        let mut cost_d = own_to_above + below_to_own;
        let mut cost_e = own_to_below + above_to_own;
        best_b.offer(cost, shift);

        if row > 1 {
            let two_above_to_above = window_difference(field, column, row - 2, row - 1, -shift);
            let above_to_two_above = window_difference(field, column, row - 1, row - 2, shift);
            let cost_a = cost + two_above_to_above + above_to_two_above;
            cost_d += above_to_two_above;
            cost_e += two_above_to_above;
            best_a.offer(cost_a, shift);
        }

        if row < last_row - 1 {
            let two_below_to_below = window_difference(field, column, row + 2, row + 1, shift);
            let below_to_two_below = window_difference(field, column, row + 1, row + 2, -shift);
            let cost_c = cost + two_below_to_below + below_to_two_below;
            cost_d += two_below_to_below;
            cost_e += below_to_two_below;
            best_c.offer(cost_c, shift);
        }

        best_d.offer(cost_d, shift);
        best_e.offer(cost_e, shift);
    }

    let winners = [best_a, best_b, best_c, best_d, best_e];
    let found: Vec<i64> = winners.iter().filter_map(|best| best.direction).collect();
    agreed_direction(found)
}

/// Pass 6, the raw direction `D0` of every edge pixel with an edge beside it, and `peak` elsewhere.
pub(super) fn raw_directions(field: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let rows = field.rows();
    let width = field.width;

    let mut directions = Grid::filled(rows, width, scaled.peak);
    for row in 1..rows - 1 {
        for column in 1..width - 1 {
            let on_edge = edges.at(row, column) == 1;
            let edge_beside = edges.at(row, column - 1) == 1 || edges.at(row, column + 1) == 1;
            if !(on_edge && edge_beside) {
                continue;
            }

            let pixels = raw_direction(field, edges, row, column, scaled);
            let value = scaled.map_value(pixels);
            directions.set(row, column, value);
        }
    }

    directions
}

/// The values of `map` other than `peak` in the 3 by 3 block around a pixel.
fn block_values(map: &Grid, row: i64, column: i64, peak: i64, with_centre: bool) -> Vec<i64> {
    let mut values = Vec::with_capacity(9);
    for row_offset in -1..=1 {
        for column_offset in -1..=1 {
            let is_centre = row_offset == 0 && column_offset == 0;
            let value = map.at(row + row_offset, column + column_offset);
            if value != peak && (with_centre || !is_centre) {
                values.push(value);
            }
        }
    }

    values
}

/// Pass 7, `D1`, which keeps an edge pixel's direction only where at least 4 of its block agree.
pub(super) fn filter_directions(directions: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let mut filtered = directions.clone();
    for row in 1..directions.rows() - 1 {
        for column in 1..directions.width - 1 {
            if edges.at(row, column) == 0 {
                continue;
            }

            let mut values = block_values(directions, row, column, scaled.peak, true);
            if values.len() < 4 {
                filtered.set(row, column, scaled.peak);
                continue;
            }

            let vote = Consensus::of(&mut values, scaled);
            let centre_missing = directions.at(row, column) == scaled.peak;
            let value = if vote.count < 4 || (vote.count < 5 && centre_missing) {
                scaled.peak
            } else {
                vote.average()
            };
            filtered.set(row, column, value);
        }
    }

    filtered
}

/// Pass 8, `D2`, which gives an edge pixel with no direction one where at least 5 neighbours agree.
pub(super) fn expand_directions(directions: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let mut expanded = directions.clone();
    for row in 1..directions.rows() - 1 {
        for column in 1..directions.width - 1 {
            let missing = directions.at(row, column) == scaled.peak;
            if !missing || edges.at(row, column) == 0 {
                continue;
            }

            let mut values = block_values(directions, row, column, scaled.peak, false);
            if values.len() < 5 {
                continue;
            }

            let vote = Consensus::of(&mut values, scaled);
            if vote.count >= 5 {
                let average = vote.average();
                expanded.set(row, column, average);
            }
        }
    }

    expanded
}

/// A direction under test in pass 9, with how far a neighbour may differ and still continue its line.
struct Line {
    value: i64,
    lim: i64,
    peak: i64,
}

/// Whether column `column` of row `other` breaks the line a direction at row `row` claims.
fn is_broken(directions: &Grid, other: i64, row: i64, column: i64, line: &Line) -> bool {
    let other_value = directions.at(other, column);
    let own_value = directions.at(row, column);
    let other_far = (other_value - line.value).abs() > line.lim && other_value != line.peak;
    let both_missing = own_value == line.peak && other_value == line.peak;
    let own_far = (own_value - line.value).abs() > line.lim && own_value != line.peak;

    other_far || both_missing || own_far
}

/// Whether any column of row `other` between `first` and `last` breaks the line.
fn any_broken(directions: &Grid, other: i64, row: i64, first: i64, last: i64, line: &Line) -> bool {
    (first..=last).any(|column| is_broken(directions, other, row, column, line))
}

/// Pass 9, `D3`, which drops a direction whose line is broken both above and below.
pub(super) fn clean_directions(directions: &Grid, edges: &Grid, scaled: &Scaled) -> Grid {
    let width = directions.width;

    let mut cleaned = directions.clone();
    for row in 1..directions.rows() - 1 {
        for column in 1..width - 1 {
            let value = directions.at(row, column);
            if value == scaled.peak || edges.at(row, column) == 0 {
                continue;
            }

            let quarter = (value - scaled.neutral) / 4;
            let line = Line {
                value,
                lim: (quarter.abs() * 2).max(scaled.twelve),
                peak: scaled.peak,
            };
            let pixels = quarter >> scaled.shift;

            let (top_first, top_last, bottom_first, bottom_last) = if pixels < 0 {
                (pixels.max(-column), 0, 0, (width - column - 1).min(-pixels))
            } else {
                (0, (width - column - 1).min(pixels), (-pixels).max(-column), 0)
            };

            let top_first = column + top_first;
            let top_last = column + top_last;
            let bottom_first = column + bottom_first;
            let bottom_last = column + bottom_last;
            let top = any_broken(directions, row - 1, row, top_first, top_last, &line);
            let bottom = any_broken(directions, row + 1, row, bottom_first, bottom_last, &line);
            if top && bottom {
                cleaned.set(row, column, scaled.peak);
            }
        }
    }

    cleaned
}
