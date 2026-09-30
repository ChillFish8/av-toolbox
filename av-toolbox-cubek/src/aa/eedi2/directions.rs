// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::votes::{
    MapScale,
    consensus,
    expand_vote,
    filter_vote,
    gather_block,
    push_present,
    ranked,
    round_average,
};
use super::{Eedi2Settings, FieldBuffer, FieldPass, GapBuffer, field_sample, is_interior};
use crate::Dispatch;

/// C's integer division, truncating toward zero, done on the magnitude so every device agrees.
#[cube]
fn truncated_division(numerator: i32, denominator: u32) -> i32 {
    let mut magnitude = numerator;
    if numerator < 0 {
        magnitude = -numerator;
    }

    let quotient = (magnitude as u32 / denominator) as i32;
    let mut result = quotient;
    if numerator < 0 {
        result = -quotient;
    }

    result
}

// The slots of the field rows 2 above to 2 below a pixel, in its loaded windows.
const TWO_ABOVE: u32 = 0;
const ABOVE: u32 = 1;
const OWN: u32 = 2;
const BELOW: u32 = 3;
const TWO_BELOW: u32 = 4;

/// Loads the 3 samples around `column` of each field row from 2 above `row` to 2 below it, 3 per slot.
///
/// A row outside the field loads as 0 and is never read.
#[cube]
fn load_windows(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    rows: u32,
    row: u32,
    column: u32,
    windows: &mut Array<f32>,
) {
    for slot in 0..5u32 {
        let exists = row + slot >= 2 && row + slot < rows + 2;

        for tap in 0..3u32 {
            let mut sample = 0.0f32;
            if exists {
                sample = field_sample(input, width, offset, row + slot - 2, column + tap - 1);
            }

            windows[(slot * 3 + tap) as usize] = sample;
        }
    }
}

/// The sum of absolute differences of 3 samples, the loaded window in `slot` against field row `second`
/// shifted by `shift` columns.
#[cube]
fn window_difference(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    column: u32,
    windows: &Array<f32>,
    slot: u32,
    second: u32,
    shift: i32,
) -> f32 {
    let mut total = 0.0f32;

    for tap in 0..3u32 {
        let own_column = column + tap - 1;
        let other_column = (own_column as i32 + shift) as u32;
        let own = windows[(slot * 3 + tap) as usize];
        let other = field_sample(input, width, offset, second, other_column);
        let difference = f32::abs(own - other);
        total += difference;
    }

    total
}

/// Whether any of the 3 mask pixels of `row` centred on `column` is set.
#[cube]
fn any_edge(edges: &Array<f32>, width: u32, row: u32, column: i32) -> bool {
    let centre = row * width + column as u32;
    let before = edges[(centre - 1) as usize] == 1.0;
    let here = edges[centre as usize] == 1.0;
    let after = edges[(centre + 1) as usize] == 1.0;

    before || here || after
}

/// The lowest cost one of pass 6's five windows has found and the direction that found it.
#[derive(CubeType)]
struct Best {
    cost: f32,
    direction: f32,
    found: bool,
}

#[cube]
impl Best {
    fn starting_at(cost: f32) -> Best {
        Best {
            cost,
            direction: 0.0,
            found: false,
        }
    }

    /// Takes `direction` when `cost` is strictly lower, so the first direction wins a tie.
    fn offer(&mut self, cost: f32, direction: i32) {
        if cost < self.cost {
            self.cost = cost;
            self.direction = direction as f32;
            self.found = true;
        }
    }
}

/// Appends the direction `best` found, if any, to the first `length` slots of `found`.
#[cube]
fn push_found(found: &mut Array<f32>, length: u32, best: &Best) -> u32 {
    let mut new_length = length;

    if best.found {
        found[length as usize] = best.direction;
        new_length += 1;
    }

    new_length
}

/// The direction in whole pixels the found winners agree on, or 0 when fewer than two agree.
#[cube]
fn agreed_direction(found: &Array<f32>, length: u32) -> i32 {
    let mut result = 0i32;

    if length > 1 {
        let upper = ranked(found, length, length / 2);
        let is_odd = length % 2 == 1;
        let mut mid = upper;

        if !is_odd {
            let lower = ranked(found, length, length / 2 - 1);
            mid = f32::trunc((lower + upper + 1.0) * 0.5);
        }

        let mut threshold = 2.0f32;
        if f32::abs(mid) >= 13.0 {
            threshold = 3.0;
        }

        let mut count = 0u32;
        let mut sum = 0i32;
        for slot in 0..length {
            let direction = found[slot as usize];

            if f32::abs(direction - mid) <= threshold {
                count += 1;
                sum += direction as i32;
            }
        }

        if count > 1 {
            result = truncated_division(sum, count);
        }
    }

    result
}

/// The costs pass 6 weighs, scaled to the plane.
#[derive(CubeType)]
struct SearchLimits {
    maxd: u32,
    nt13: f32,
    nt19: f32,
}

/// Pass 6's direction at one pixel, searching every shift within `maxd` that stays inside the row.
#[cube]
fn raw_direction(
    input: &Array<f32>,
    edges: &Array<f32>,
    width: u32,
    rows: u32,
    offset: u32,
    row: u32,
    column: u32,
    limits: &SearchLimits,
) -> i32 {
    let reach_left = u32::min(limits.maxd, column - 1);
    let reach = u32::min(reach_left, width - 2 - column) as i32;
    let centre = field_sample(input, width, offset, row, column);
    let below = field_sample(input, width, offset, row + 1, column);
    let above = field_sample(input, width, offset, row - 1, column);
    let below_step = f32::abs(centre - below);
    let above_step = f32::abs(centre - above);
    let vertical = below_step + above_step;

    let start_a = f32::min(limits.nt19, 9.0 * vertical);
    let start_b = f32::min(limits.nt13, 6.0 * vertical);
    let mut best_a = Best::starting_at(start_a);
    let mut best_b = Best::starting_at(start_b);
    let mut best_c = Best::starting_at(start_a);
    let mut best_d = Best::starting_at(start_b);
    let mut best_e = Best::starting_at(start_b);

    let mut windows = Array::<f32>::new(15usize);
    load_windows(input, width, offset, rows, row, column, &mut windows);

    let span = (2 * reach + 1) as u32;
    for step in 0..span {
        let shift = step as i32 - reach;
        let signed_column = column as i32;
        let edge_above = row == 1 || any_edge(edges, width, row - 1, signed_column + shift);
        let edge_below = row + 2 == rows || any_edge(edges, width, row + 1, signed_column - shift);

        if edge_above && edge_below {
            let own_to_below =
                window_difference(input, width, offset, column, &windows, OWN, row + 1, -shift);
            let own_to_above = window_difference(input, width, offset, column, &windows, OWN, row - 1, shift);
            let above_to_own = window_difference(input, width, offset, column, &windows, ABOVE, row, -shift);
            let below_to_own = window_difference(input, width, offset, column, &windows, BELOW, row, shift);
            let cost = own_to_below + own_to_above + above_to_own + below_to_own;
            let mut cost_d = own_to_above + below_to_own;
            let mut cost_e = own_to_below + above_to_own;
            best_b.offer(cost, shift);

            if row > 1 {
                let two_above = row - 2;
                let two_above_to_above =
                    window_difference(input, width, offset, column, &windows, TWO_ABOVE, row - 1, -shift);
                let above_to_two_above =
                    window_difference(input, width, offset, column, &windows, ABOVE, two_above, shift);
                let cost_a = cost + two_above_to_above + above_to_two_above;
                cost_d += above_to_two_above;
                cost_e += two_above_to_above;
                best_a.offer(cost_a, shift);
            }

            if row + 2 < rows {
                let two_below = row + 2;
                let two_below_to_below =
                    window_difference(input, width, offset, column, &windows, TWO_BELOW, row + 1, shift);
                let below_to_two_below =
                    window_difference(input, width, offset, column, &windows, BELOW, two_below, -shift);
                let cost_c = cost + two_below_to_below + below_to_two_below;
                cost_d += two_below_to_below;
                cost_e += below_to_two_below;
                best_c.offer(cost_c, shift);
            }

            best_d.offer(cost_d, shift);
            best_e.offer(cost_e, shift);
        }
    }

    let mut found = Array::<f32>::new(5usize);
    let mut length = push_found(&mut found, 0u32, &best_a);
    length = push_found(&mut found, length, &best_b);
    length = push_found(&mut found, length, &best_c);
    length = push_found(&mut found, length, &best_d);
    length = push_found(&mut found, length, &best_e);

    agreed_direction(&found, length)
}

/// Pass 6, the raw direction `D0` of every edge pixel with an edge beside it, and `peak` elsewhere.
///
/// A direction of `d` pixels is written as `neutral + d * four_unit`.
#[cube(launch_unchecked)]
fn raw_directions_kernel(
    input: &Array<f32>,
    edges: &Array<f32>,
    directions: &mut Array<f32>,
    width: u32,
    rows: u32,
    offset: u32,
    maxd: u32,
    nt13: f32,
    nt19: f32,
    peak: f32,
    neutral: f32,
    four_unit: f32,
) {
    if ABSOLUTE_POS >= directions.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let interior = is_interior(position, width, rows);
    let mut value = peak;

    if interior {
        let on_edge = edges[ABSOLUTE_POS] == 1.0;
        let before = edges[(position - 1) as usize] == 1.0;
        let after = edges[(position + 1) as usize] == 1.0;

        if on_edge && (before || after) {
            let limits = SearchLimits { maxd, nt13, nt19 };
            let pixels = raw_direction(input, edges, width, rows, offset, row, column, &limits);
            value = neutral + pixels as f32 * four_unit;
        }
    }

    directions[ABSOLUTE_POS] = value;
}

/// Pass 7, `D1`, which keeps an edge pixel's direction only where at least 4 of its block agree.
#[cube(launch_unchecked)]
fn filter_directions_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    rows: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let interior = is_interior(position, width, rows);
    let centre = source[ABSOLUTE_POS];
    let mut value = centre;

    if interior && edges[ABSOLUTE_POS] == 1.0 {
        let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
        let mut values = Array::<f32>::new(9usize);
        let length = gather_block(
            source,
            width,
            row - 1,
            row + 1,
            row,
            column,
            peak,
            true,
            &mut values,
        );
        value = filter_vote(&values, length, centre, &scale);
    }

    destination[ABSOLUTE_POS] = value;
}

/// Pass 8, `D2`, which gives an edge pixel with no direction one where at least 5 neighbours agree.
#[cube(launch_unchecked)]
fn expand_directions_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    rows: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let interior = is_interior(position, width, rows);
    let centre = source[ABSOLUTE_POS];
    let mut value = centre;

    if interior && centre == peak && edges[ABSOLUTE_POS] == 1.0 {
        let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
        let mut values = Array::<f32>::new(9usize);
        let length = gather_block(
            source,
            width,
            row - 1,
            row + 1,
            row,
            column,
            peak,
            false,
            &mut values,
        );
        value = expand_vote(&values, length, &scale);
    }

    destination[ABSOLUTE_POS] = value;
}

/// A direction under test in pass 9, with how far a neighbour may differ and still continue its line.
#[derive(CubeType)]
struct Line {
    value: f32,
    lim: f32,
    peak: f32,
}

/// Whether column `column` of row `other` breaks the line a direction at row `row` claims.
#[cube]
fn is_broken(directions: &Array<f32>, width: u32, other: u32, row: u32, column: u32, line: &Line) -> bool {
    let other_value = directions[(other * width + column) as usize];
    let own_value = directions[(row * width + column) as usize];
    let other_far = f32::abs(other_value - line.value) > line.lim && other_value != line.peak;
    let both_missing = own_value == line.peak && other_value == line.peak;
    let own_far = f32::abs(own_value - line.value) > line.lim && own_value != line.peak;

    other_far || both_missing || own_far
}

/// Whether any column of row `other` between `first` and `last` breaks the line.
#[cube]
fn any_broken(
    directions: &Array<f32>,
    width: u32,
    other: u32,
    row: u32,
    first: i32,
    last: i32,
    line: &Line,
) -> bool {
    let mut broken = false;
    let span = (last - first + 1) as u32;

    for step in 0..span {
        let column = first as u32 + step;

        if is_broken(directions, width, other, row, column, line) {
            broken = true;
        }
    }

    broken
}

/// Pass 9, `D3`, which drops a direction whose line is broken both above and below.
///
/// `reduce` is `2^(8 - bits)`, which turns a floor multiplication into a right shift by `bits - 8`.
#[cube(launch_unchecked)]
fn clean_directions_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    rows: u32,
    peak: f32,
    neutral: f32,
    twelve: f32,
    reduce: f32,
) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let interior = is_interior(position, width, rows);
    let centre = source[ABSOLUTE_POS];
    let mut value = centre;

    if interior && centre != peak && edges[ABSOLUTE_POS] == 1.0 {
        let quarter = f32::trunc((centre - neutral) * 0.25);
        let quarter_size = f32::abs(quarter);
        let line = Line {
            value: centre,
            lim: f32::max(quarter_size * 2.0, twelve),
            peak,
        };
        let pixels = f32::floor(quarter * reduce) as i32;
        let signed_column = column as i32;
        let room_right = width as i32 - signed_column - 1;

        let mut top_first = 0i32;
        let mut top_last = 0i32;
        let mut bottom_first = 0i32;
        let mut bottom_last = 0i32;
        if pixels < 0 {
            top_first = i32::max(pixels, -signed_column);
            bottom_last = i32::min(room_right, -pixels);
        } else {
            top_last = i32::min(room_right, pixels);
            bottom_first = i32::max(-pixels, -signed_column);
        }

        let top_first = signed_column + top_first;
        let top_last = signed_column + top_last;
        let bottom_first = signed_column + bottom_first;
        let bottom_last = signed_column + bottom_last;
        let top = any_broken(source, width, row - 1, row, top_first, top_last, &line);
        let bottom = any_broken(source, width, row + 1, row, bottom_first, bottom_last, &line);

        if top && bottom {
            value = peak;
        }
    }

    destination[ABSOLUTE_POS] = value;
}

/// Whether gap pixel `column` of gap row `gap` touches the edge mask above or below it.
#[cube]
fn touches_edge(edges: &Array<f32>, width: u32, gap: u32, column: u32) -> bool {
    let above = gap * width + column;
    let below = above + width;

    edges[above as usize] == 1.0 || edges[below as usize] == 1.0
}

/// Whether two directions agree within `lim`, or either is missing.
#[cube]
fn agrees(first: f32, second: f32, second_missing: bool, lim: f32, peak: f32) -> u32 {
    let mut agreeing = 0u32;

    let difference = f32::abs(first - second);

    if difference <= lim || first == peak || second_missing {
        agreeing = 1;
    }

    agreeing
}

/// Pass 10's direction at one gap pixel from the 3 directions above it and the 3 below.
///
/// The third agreement test compares the right direction above with the left one below, as the plugin does.
#[cube]
fn gap_direction(directions: &Array<f32>, width: u32, gap: u32, column: u32, scale: &MapScale) -> f32 {
    let peak = scale.peak;
    let above = gap * width + column;
    let below = above + width;
    let above_left = directions[(above - 1) as usize];
    let above_centre = directions[above as usize];
    let above_right = directions[(above + 1) as usize];
    let below_left = directions[(below - 1) as usize];
    let below_centre = directions[below as usize];
    let below_right = directions[(below + 1) as usize];

    let mut values = Array::<f32>::new(6usize);
    let mut length = push_present(&mut values, 0u32, above_left, peak);
    length = push_present(&mut values, length, above_centre, peak);
    length = push_present(&mut values, length, above_right, peak);
    length = push_present(&mut values, length, below_left, peak);
    length = push_present(&mut values, length, below_centre, peak);
    length = push_present(&mut values, length, below_right, peak);

    let mut result = peak;
    if length >= 3 {
        let vote = consensus(&values, length, scale);
        let left_agrees = agrees(above_left, below_left, below_left == peak, vote.lim, peak);
        let centre_agrees = agrees(above_centre, below_centre, below_centre == peak, vote.lim, peak);
        let right_agrees = agrees(above_right, below_left, below_right == peak, vote.lim, peak);
        let agreeing = left_agrees + centre_agrees + right_agrees;
        let enough_agree = vote.count + 2 >= length && vote.count >= 2;

        if agreeing >= 2 && enough_agree {
            result = round_average(vote.sum + vote.mid, vote.count + 1);
        }
    }

    result
}

/// Pass 10, the gap directions `X0` from the cleaned field directions, `peak` where none agree.
#[cube(launch_unchecked)]
fn gap_directions_kernel(
    directions: &Array<f32>,
    edges: &Array<f32>,
    gaps: &mut Array<f32>,
    width: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= gaps.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let gap = position / width;
    let column = position % width;
    let inside = column >= 1 && column + 1 < width;
    let mut value = peak;

    if inside && touches_edge(edges, width, gap, column) {
        let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
        value = gap_direction(directions, width, gap, column, &scale);
    }

    gaps[ABSOLUTE_POS] = value;
}

/// The first and last gap rows around `gap` that exist among `gap_rows`.
#[derive(CubeType)]
struct GapRows {
    first: u32,
    last: u32,
}

#[cube]
fn gap_rows_around(gap: u32, gap_rows: u32) -> GapRows {
    let mut first = gap;
    if gap >= 1 {
        first = gap - 1;
    }

    let mut last = gap;
    if gap + 1 < gap_rows {
        last = gap + 1;
    }

    GapRows { first, last }
}

/// Pass 11, `X1`, the gap filter `filter2x`, using only the gap rows that exist.
#[cube(launch_unchecked)]
fn filter_gaps_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    gap_rows: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let gap = position / width;
    let column = position % width;
    let inside = column >= 1 && column + 1 < width;
    let centre = source[ABSOLUTE_POS];
    let mut value = centre;

    if inside && touches_edge(edges, width, gap, column) {
        let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
        let rows = gap_rows_around(gap, gap_rows);
        let mut values = Array::<f32>::new(9usize);
        let length = gather_block(
            source,
            width,
            rows.first,
            rows.last,
            gap,
            column,
            peak,
            true,
            &mut values,
        );
        value = filter_vote(&values, length, centre, &scale);
    }

    destination[ABSOLUTE_POS] = value;
}

/// Pass 12, `X2`, the gap expansion `expand2x`, using only the gap rows that exist.
#[cube(launch_unchecked)]
fn expand_gaps_kernel(
    source: &Array<f32>,
    edges: &Array<f32>,
    destination: &mut Array<f32>,
    width: u32,
    gap_rows: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let gap = position / width;
    let column = position % width;
    let inside = column >= 1 && column + 1 < width;
    let centre = source[ABSOLUTE_POS];
    let mut value = centre;

    if inside && centre == peak && touches_edge(edges, width, gap, column) {
        let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
        let rows = gap_rows_around(gap, gap_rows);
        let mut values = Array::<f32>::new(9usize);
        let length = gather_block(
            source,
            width,
            rows.first,
            rows.last,
            gap,
            column,
            peak,
            false,
            &mut values,
        );
        value = expand_vote(&values, length, &scale);
    }

    destination[ABSOLUTE_POS] = value;
}

/// Launches pass 6, reading the final mask in `edges` and writing `D0` into `working`.
pub(super) fn raw_directions<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let four_unit = 4.0 * settings.unit;

    // Safety: the input plane holds `width * 2 * rows` values, and `edges` and `working` hold
    // `width * rows`, as `FieldPass::new` checked. Each unit writes only its own pixel. It reads field rows
    // within two of its own, only for interior pixels, and its search reach is clamped so every column it
    // reads stays inside the row.
    unsafe {
        let input = pass.input.array::<R>();
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let directions = pass.field_array::<R>(FieldBuffer::Working);

        raw_directions_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            edges,
            directions,
            pass.width,
            pass.rows,
            pass.offset,
            settings.maxd,
            settings.nt13,
            settings.nt19,
            settings.peak,
            settings.neutral,
            four_unit,
        );
    }
}

/// Launches pass 7, filtering `D0` in `working` into `D1` in `directions`.
pub(super) fn filter_directions<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `working`, `edges` and `directions` each hold `width * rows` values, as `FieldPass::new`
    // checked. Each unit writes only its own pixel and reads its 3 by 3 block only for interior pixels.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Working);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.field_array::<R>(FieldBuffer::Directions);

        filter_directions_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            pass.rows,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}

/// Launches pass 8, expanding `D1` in `directions` into `D2` in `working`.
pub(super) fn expand_directions<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `directions`, `edges` and `working` each hold `width * rows` values, as `FieldPass::new`
    // checked. Each unit writes only its own pixel and reads its 3 by 3 block only for interior pixels.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Directions);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.field_array::<R>(FieldBuffer::Working);

        expand_directions_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            pass.rows,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}

/// Launches pass 9, cleaning `D2` in `working` into `D3` in `directions`.
pub(super) fn clean_directions<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `working`, `edges` and `directions` each hold `width * rows` values, as `FieldPass::new`
    // checked. Each unit writes only its own pixel. It reads the rows above and below only for interior
    // pixels, at columns clamped into the row.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Working);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.field_array::<R>(FieldBuffer::Directions);

        clean_directions_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            pass.rows,
            settings.peak,
            settings.neutral,
            settings.twelve,
            settings.reduce,
        );
    }
}

/// Launches pass 10, writing `X0` into `working` from `D3` in `directions`.
pub(super) fn gap_directions<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.gap_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `directions` and `edges` hold `width * rows` values and `working` at least
    // `width * (rows - 1)`, as `FieldPass::new` checked. Each unit writes only its own gap pixel and reads
    // field rows `gap` and `gap + 1`, at columns within one of its own only when those lie inside the row.
    unsafe {
        let directions = pass.field_array::<R>(FieldBuffer::Directions);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let gaps = pass.gap_array::<R>(GapBuffer::Working);

        gap_directions_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            directions,
            edges,
            gaps,
            pass.width,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}

/// Launches pass 11, filtering `X0` in `working` into `X1` in `spare`.
pub(super) fn filter_gaps<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.gap_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `working` and `spare` hold at least `width * (rows - 1)` values and `edges` `width * rows`,
    // as `FieldPass::new` checked. Each unit writes only its own gap pixel and reads only the gap rows
    // around it that exist, at columns within one of its own only when those lie inside the row.
    unsafe {
        let source = pass.gap_array::<R>(GapBuffer::Working);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.gap_array::<R>(GapBuffer::Spare);

        filter_gaps_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            pass.rows - 1,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}

/// Launches pass 12, expanding `X1` in `spare` into `X2` in `working`.
pub(super) fn expand_gaps<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.gap_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `spare` and `working` hold at least `width * (rows - 1)` values and `edges` `width * rows`,
    // as `FieldPass::new` checked. Each unit writes only its own gap pixel and reads only the gap rows
    // around it that exist, at columns within one of its own only when those lie inside the row.
    unsafe {
        let source = pass.gap_array::<R>(GapBuffer::Spare);
        let edges = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.gap_array::<R>(GapBuffer::Working);

        expand_gaps_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            edges,
            destination,
            pass.width,
            pass.rows - 1,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}
