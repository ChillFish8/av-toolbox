// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::votes::{MapScale, lim};
use super::{Eedi2Settings, FieldBuffer, FieldPass, GapBuffer, field_sample};
use crate::aa::row_launch;
use crate::{DevicePlane, Dispatch};

/// The values the lattice compares with, scaled to the plane.
///
/// `reduce` is `2^(8 - bits)`, which turns a floor multiplication into a right shift by `bits - 8`.
#[derive(CubeType)]
struct LatticeScale {
    map: MapScale,
    reduce: f32,
    three: f32,
    nine: f32,
    nt4: f32,
    nt7: f32,
    nt8: f32,
    dt: u32,
}

/// The kept field rows either side of one gap row.
#[derive(CubeType)]
struct GapRows {
    width: u32,
    offset: u32,
    gap: u32,
}

#[cube]
impl GapRows {
    fn up(&self, input: &[f32], column: u32) -> f32 {
        field_sample(input, self.width, self.offset, self.gap, column)
    }

    fn down(&self, input: &[f32], column: u32) -> f32 {
        field_sample(input, self.width, self.offset, self.gap + 1, column)
    }

    fn up_direction(&self, directions: &[f32], column: u32) -> f32 {
        directions[(self.gap * self.width + column) as usize]
    }

    fn down_direction(&self, directions: &[f32], column: u32) -> f32 {
        directions[((self.gap + 1) * self.width + column) as usize]
    }

    /// The average of the samples directly above and below `column`, rounded up.
    fn vertical(&self, input: &[f32], column: u32) -> f32 {
        let up = self.up(input, column);
        let down = self.down(input, column);

        f32::floor((up + down + 1.0) * 0.5)
    }
}

/// C's integer division by 2, truncating toward zero.
#[cube]
fn half_toward_zero(value: i32) -> i32 {
    let mut half = value / 2;
    if value < 0 {
        half = -((-value) / 2);
    }

    half
}

/// `column` moved by `shift`, which stays inside the row.
#[cube]
fn shifted(column: u32, shift: i32) -> u32 {
    (column as i32 + shift) as u32
}

/// The columns of the half-pixel pair of `shift` at `column`, above and below.
///
/// The pair steps by `shift / 2` and `(shift + 1) / 2`, both truncated toward zero, so a shift of 1 pairs
/// the steps 0 and 1 while a shift of -1 pairs step 0 with itself.
#[derive(CubeType)]
struct HalfPair {
    above_down: u32,
    above_up: u32,
    below_down: u32,
    below_up: u32,
}

#[cube]
impl HalfPair {
    fn of(column: u32, shift: i32) -> HalfPair {
        let half_down = half_toward_zero(shift);
        let half_up = half_toward_zero(shift + 1);

        HalfPair {
            above_down: shifted(column, half_down),
            above_up: shifted(column, half_up),
            below_down: shifted(column, -half_down),
            below_up: shifted(column, -half_up),
        }
    }
}

/// Whether the 3 samples above and below `column` vary too little to follow a direction.
///
/// A negative variance, possible above 8 bits, is not low.
#[cube]
fn has_low_variance(input: &[f32], rows: &GapRows, column: u32, reduce: f32) -> bool {
    let mut sum = 0.0f32;
    let mut sum_of_squares = 0.0f32;

    for tap in 0..3u32 {
        let tap_column = column + tap - 1;
        let up = rows.up(input, tap_column);
        let down = rows.down(input, tap_column);
        let reduced_up = f32::floor(up * reduce);
        let reduced_down = f32::floor(down * reduce);
        sum += up + down;
        sum_of_squares += reduced_up * reduced_up + reduced_down * reduced_down;
    }

    let reduced_sum = f32::floor(sum * reduce);
    let variance = 6.0 * sum_of_squares - reduced_sum * reduced_sum;
    let is_negative = variance < 0.0;

    !is_negative && variance < 576.0
}

/// The 5 samples of field row `row` around `column`, from 2 before to 2 after.
#[derive(CubeType)]
struct Neighbourhood {
    two_before: f32,
    one_before: f32,
    centre: f32,
    one_after: f32,
    two_after: f32,
}

#[cube]
fn neighbourhood(input: &[f32], rows: &GapRows, row: u32, column: u32) -> Neighbourhood {
    let width = rows.width;
    let offset = rows.offset;

    Neighbourhood {
        two_before: field_sample(input, width, offset, row, column - 2),
        one_before: field_sample(input, width, offset, row, column - 1),
        centre: field_sample(input, width, offset, row, column),
        one_after: field_sample(input, width, offset, row, column + 1),
        two_after: field_sample(input, width, offset, row, column + 2),
    }
}

/// Whether the samples dip at the centre below both sides by more than `three`.
#[cube]
fn is_valley(samples: &Neighbourhood, three: f32) -> bool {
    let before = f32::max(samples.two_before, samples.one_before);
    let after = f32::max(samples.two_after, samples.one_after);

    samples.centre < before - three && samples.centre < after - three
}

/// Whether the samples peak at the centre above both sides by more than `three`.
#[cube]
fn is_ridge(samples: &Neighbourhood, three: f32) -> bool {
    let before = f32::min(samples.two_before, samples.one_before);
    let after = f32::min(samples.two_after, samples.one_after);

    samples.centre > before + three && samples.centre > after + three
}

/// Whether the rows above and below `column` both hold a valley there, or both a ridge.
///
/// `column` must lie at least 2 inside the row.
#[cube]
fn is_extremum(input: &[f32], rows: &GapRows, column: u32, three: f32) -> bool {
    let up = neighbourhood(input, rows, rows.gap, column);
    let down = neighbourhood(input, rows, rows.gap + 1, column);
    let valleys = is_valley(&up, three) && is_valley(&down, three);
    let ridges = is_ridge(&up, three) && is_ridge(&down, three);

    valleys || ridges
}

/// The cost of pairing the samples above `column` with those below shifted by `shift`, both ways.
#[cube]
fn crossing_cost(input: &[f32], rows: &GapRows, column: u32, shift: i32) -> f32 {
    let mut total = 0.0f32;

    for tap in 0..3u32 {
        let own = column + tap - 1;
        let down_column = shifted(own, -shift);
        let up_column = shifted(own, shift);
        let up_own = rows.up(input, own);
        let down_shifted = rows.down(input, down_column);
        let down_own = rows.down(input, own);
        let up_shifted = rows.up(input, up_column);
        let up_to_down = f32::abs(up_own - down_shifted);
        let down_to_up = f32::abs(down_own - up_shifted);
        total += up_to_down + down_to_up;
    }

    total
}

/// The direction under test at a gap pixel, and how far another may differ and still agree with it.
#[derive(CubeType)]
struct Claim {
    direction: f32,
    lim: f32,
    peak: f32,
}

#[cube]
fn within(first: f32, second: f32, lim: f32) -> bool {
    f32::abs(first - second) <= lim
}

/// A direction agreeing with the claim within its `lim`.
#[cube]
fn is_near(value: f32, claim: &Claim) -> bool {
    let close = within(value, claim.direction, claim.lim);

    value != claim.peak && close
}

/// Whether the cleaned directions along `shift` back up the claim, both above and below.
#[cube]
fn is_supported(directions: &[f32], rows: &GapRows, column: u32, shift: i32, claim: &Claim) -> bool {
    let mut near_above = false;
    let mut near_below = false;

    for tap in 0..3u32 {
        let own = column + tap - 1;
        let above_column = shifted(own, shift);
        let below_column = shifted(own, -shift);
        let above = rows.up_direction(directions, above_column);
        let below = rows.down_direction(directions, below_column);
        let above_near = is_near(above, claim);
        let below_near = is_near(below, claim);
        near_above = near_above || above_near;
        near_below = near_below || below_near;
    }

    near_above && near_below
}

/// Whether a direction above agrees with either direction of the half-pixel pair below.
#[cube]
fn pairs_below(above: f32, below_down: f32, below_up: f32, claim: &Claim) -> bool {
    let close_down = within(above, below_down, claim.lim);
    let close_up = within(above, below_up, claim.lim);

    (close_down || close_up) && above != claim.peak
}

/// Whether the cleaned directions at the half-pixel pair of `shift` agree with each other and with the
/// claim.
#[cube]
fn is_matched(directions: &[f32], rows: &GapRows, column: u32, shift: i32, claim: &Claim) -> bool {
    let pair = HalfPair::of(column, shift);
    let above_down = rows.up_direction(directions, pair.above_down);
    let above_up = rows.up_direction(directions, pair.above_up);
    let below_down = rows.down_direction(directions, pair.below_down);
    let below_up = rows.down_direction(directions, pair.below_up);

    let pair_a = pairs_below(above_down, below_down, below_up, claim);
    let pair_b = pairs_below(above_up, below_down, below_up, claim);
    let above_matches =
        within(claim.direction, above_down, claim.lim) || within(claim.direction, above_up, claim.lim);
    let below_matches =
        within(claim.direction, below_down, claim.lim) || within(claim.direction, below_up, claim.lim);

    (pair_a || pair_b) && above_matches && below_matches
}

/// The cost of pairing the 3 samples above and below the half-pixel point of `shift`.
#[cube]
fn half_cost(input: &[f32], rows: &GapRows, column: u32, shift: i32) -> f32 {
    let half_down = half_toward_zero(shift);
    let mut total = 0.0f32;

    for tap in 0..3u32 {
        let own = column + tap - 1;
        let up_column = shifted(own, half_down);
        let down_column = shifted(own, -half_down);
        let up = rows.up(input, up_column);
        let down = rows.down(input, down_column);
        total += f32::abs(up - down);
    }

    total
}

/// The two samples above and the two below the half-pixel point of `shift`, each pair summed.
#[derive(CubeType)]
struct PairSums {
    above: f32,
    below: f32,
}

#[cube]
fn pair_sums(input: &[f32], rows: &GapRows, column: u32, shift: i32) -> PairSums {
    let pair = HalfPair::of(column, shift);
    let above_down = rows.up(input, pair.above_down);
    let above_up = rows.up(input, pair.above_up);
    let below_down = rows.down(input, pair.below_down);
    let below_up = rows.down(input, pair.below_up);

    PairSums {
        above: above_down + above_up,
        below: below_down + below_up,
    }
}

/// The average of the 4 samples of a half-pixel pair, rounded to nearest.
#[cube]
fn pair_average(sums: &PairSums) -> f32 {
    f32::floor((sums.above + sums.below + 2.0) * 0.25)
}

/// A rebuilt value and the shift it was taken along, when a search found one.
#[derive(CubeType)]
struct Pick {
    value: f32,
    shift: i32,
    found: bool,
}

/// The directional search within 2 pixels of the claim, clamped so every read stays inside the row.
#[cube]
fn directional_pick(
    input: &[f32],
    directions: &[f32],
    rows: &GapRows,
    column: u32,
    claim: &Claim,
    scale: &LatticeScale,
) -> Pick {
    let centred = (claim.direction - scale.map.neutral + 2.0 * scale.map.unit) * scale.map.reduce_quarter;
    let rounded = f32::floor(centred) as i32;
    let room_left = column - 1;
    let room_right = rows.width - 2 - column;
    let reach = u32::min(room_left, room_right) as i32;
    let lowest = i32::max(rounded - 2, -reach);
    let first = i32::min(lowest, reach);
    let highest = i32::max(rounded + 2, -reach);
    let last = i32::min(highest, reach);

    let mut best = scale.nt8;
    let mut pick = Pick {
        value: 0.0,
        shift: 0,
        found: false,
    };

    let span = (last - first + 1) as u32;
    for step in 0..span {
        let shift = first + step as i32;
        let cost = crossing_cost(input, rows, column, shift);

        if cost < best && is_supported(directions, rows, column, shift, claim) {
            let half_difference = half_cost(input, rows, column, shift);

            if half_difference < scale.nt4 && is_matched(directions, rows, column, shift, claim) {
                let sums = pair_sums(input, rows, column, shift);
                pick.value = pair_average(&sums);
                pick.shift = shift;
                pick.found = true;
                best = cost;
            }
        }
    }

    pick
}

/// The fallback search within `dt`, taking only a value between the samples directly above and below.
#[cube]
fn fallback_pick(input: &[f32], rows: &GapRows, column: u32, scale: &LatticeScale) -> Pick {
    let room_left = column - 1;
    let room_right = rows.width - 2 - column;
    let room = u32::min(room_left, room_right);
    let reach = u32::min(scale.dt, room) as i32;
    let up = rows.up(input, column);
    let down = rows.down(input, column);
    let lowest = f32::min(up, down);
    let highest = f32::max(up, down);

    let mut best = scale.nt7;
    let mut pick = Pick {
        value: 0.0,
        shift: 0,
        found: false,
    };

    let span = (2 * reach + 1) as u32;
    for step in 0..span {
        let shift = step as i32 - reach;
        let sums = pair_sums(input, rows, column, shift);
        let crossing = crossing_cost(input, rows, column, shift);
        let cost = f32::abs(sums.above - sums.below) + crossing;

        if cost < best {
            let candidate = pair_average(&sums);

            if candidate >= lowest && candidate <= highest {
                pick.value = candidate;
                pick.shift = shift;
                pick.found = true;
                best = cost;
            }
        }
    }

    pick
}

/// A gap pixel's rebuilt value and the direction the lattice records for it.
#[derive(CubeType)]
struct Rebuilt {
    value: f32,
    direction: f32,
}

/// The rebuilt value and new direction at a gap pixel whose direction is not isolated.
///
/// `column` must lie inside the row.
#[cube]
fn rebuild(
    input: &[f32],
    directions: &[f32],
    rows: &GapRows,
    column: u32,
    claim: &Claim,
    scale: &LatticeScale,
) -> Rebuilt {
    let four_unit = 4.0 * scale.map.unit;
    let mut value = rows.vertical(input, column);
    let mut direction = scale.map.neutral;
    let mut decided = false;

    if claim.lim < scale.nine && has_low_variance(input, rows, column, scale.reduce) {
        direction = scale.map.peak;
        decided = true;
    }

    let far_from_sides = column > 1 && column + 2 < rows.width;
    if !decided && far_from_sides {
        // The extremum test reads 2 columns either side, so it runs only where those exist.
        if is_extremum(input, rows, column, scale.three) {
            decided = true;
        }
    }

    if !decided {
        let pick = directional_pick(input, directions, rows, column, claim, scale);

        if pick.found {
            value = pick.value;
            direction = scale.map.neutral + pick.shift as f32 * four_unit;
            decided = true;
        }
    }

    if !decided {
        let pick = fallback_pick(input, rows, column, scale);

        if pick.found {
            value = pick.value;
            direction = scale.map.neutral + pick.shift as f32 * four_unit;
        }
    }

    Rebuilt { value, direction }
}

/// Pass 15, the lattice's result at every gap pixel whose direction is not isolated, as its rebuilt
/// `value` and new `direction`.
///
/// A pixel with no direction gets the vertical average and `peak`.
#[cube(launch_unchecked)]
fn lattice_candidates_kernel(
    input: &[f32],
    directions: &[f32],
    gaps: &[f32],
    values: &mut [f32],
    candidate_directions: &mut [f32],
    width: u32,
    offset: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce: f32,
    reduce_quarter: f32,
    three: f32,
    nine: f32,
    nt4: f32,
    nt7: f32,
    nt8: f32,
    dt: u32,
) {
    if ABSOLUTE_POS >= values.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let gap = position / width;
    let column = position % width;
    let rows = GapRows { width, offset, gap };
    let direction = gaps[ABSOLUTE_POS];
    let inside = column >= 1 && column + 1 < width;
    let mut value = rows.vertical(input, column);
    let mut new_direction = peak;

    if inside && direction != peak {
        let scale = LatticeScale {
            map: MapScale::new(peak, neutral, unit, reduce_quarter),
            reduce,
            three,
            nine,
            nt4,
            nt7,
            nt8,
            dt,
        };
        let claim = Claim {
            direction,
            lim: lim(direction, &scale.map),
            peak,
        };
        let rebuilt = rebuild(input, directions, &rows, column, &claim, &scale);
        value = rebuilt.value;
        new_direction = rebuilt.direction;
    }

    values[ABSOLUTE_POS] = value;
    candidate_directions[ABSOLUTE_POS] = new_direction;
}

/// Pass 16, one unit per gap row, which walks the row left to right writing the directions the lattice
/// used `L` and the rebuilt rows `R`.
///
/// A direction differing by more than its `LIM` from both neighbours is isolated and gives the vertical
/// average and `neutral`. The left neighbour is the direction already written for the column before.
/// Every other pixel takes its candidate.
#[cube(launch_unchecked)]
fn lattice_resolve_kernel(
    input: &[f32],
    gaps: &[f32],
    values: &[f32],
    candidate_directions: &[f32],
    used: &mut [f32],
    rebuilt: &mut [f32],
    width: u32,
    gap_rows: u32,
    offset: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    let gap = ABSOLUTE_POS as u32;
    if gap >= gap_rows {
        terminate!();
    }

    let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
    let rows = GapRows { width, offset, gap };
    let row_start = gap * width;
    let mut previous = peak;

    for column in 0..width {
        let index = (row_start + column) as usize;
        let direction = gaps[index];
        let inside = column >= 1 && column + 1 < width;
        let mut value = values[index];
        let mut new_direction = candidate_directions[index];

        if inside && direction != peak {
            let threshold = lim(direction, &scale);
            let right = gaps[index + 1];
            let left_far = f32::abs(direction - previous) > threshold;
            let right_far = f32::abs(direction - right) > threshold;

            if left_far && right_far {
                value = rows.vertical(input, column);
                new_direction = neutral;
            }
        }

        rebuilt[index] = value;
        used[index] = new_direction;
        previous = new_direction;
    }
}

/// Writes one output sample, copying kept rows and taking the others from the rebuilt rows.
///
/// With `pp` 1, a rebuilt sample whose lattice direction disagrees with the re-filtered one goes back to
/// the vertical average, unless the lattice used `peak` or `neutral` there.
#[cube(launch_unchecked)]
fn assemble_kernel(
    input: &[f32],
    used: &[f32],
    refiltered: &[f32],
    rebuilt: &[f32],
    output: &mut [f32],
    width: u32,
    height: u32,
    offset: u32,
    pp: u32,
    peak: f32,
    neutral: f32,
    unit: f32,
    reduce_quarter: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let top_edge = offset == 1 && row == 0;
    let bottom_edge = offset == 0 && row == height - 1;

    if row % 2 == offset {
        output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
    } else if top_edge {
        output[ABSOLUTE_POS] = input[(width + column) as usize];
    } else if bottom_edge {
        output[ABSOLUTE_POS] = input[((height - 2) * width + column) as usize];
    } else {
        let gap = (row - 1 - offset) / 2;
        let index = (gap * width + column) as usize;
        let mut value = rebuilt[index];

        if pp == 1 {
            let scale = MapScale::new(peak, neutral, unit, reduce_quarter);
            let fresh = refiltered[index];
            let kept = used[index];
            let threshold = lim(fresh, &scale);
            let disagrees = f32::abs(fresh - kept) > threshold;
            let turned = kept != peak && kept != neutral;

            if disagrees && turned {
                let up = input[(position - width) as usize];
                let down = input[(position + width) as usize];
                value = f32::floor((up + down + 1.0) * 0.5);
            }
        }

        output[ABSOLUTE_POS] = value;
    }
}

/// Launches pass 15, reading `D3` in `directions` and `X4` in `working`, and writing the candidate values
/// into `candidates` and their directions into `spare`.
pub(super) fn lattice_candidates(
    client: &Client,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.gap_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input plane holds `width * 2 * rows` values, `directions` `width * rows` and every other
    // scratch buffer at least `width * (rows - 1)`, as `FieldPass::new` checked. Each unit writes only its
    // own gap pixel. It reads the field rows either side of its gap row, in the input and in `directions`,
    // and searches only pixels inside the row, with every shift clamped so each column it reads stays
    // inside the row.
    unsafe {
        let input = pass.input.array();
        let directions = pass.field_array(FieldBuffer::Directions);
        let gaps = pass.gap_array(GapBuffer::Working);
        let values = pass.gap_array(GapBuffer::Candidates);
        let candidate_directions = pass.gap_array(GapBuffer::Spare);

        lattice_candidates_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            directions,
            gaps,
            values,
            candidate_directions,
            pass.width,
            pass.offset,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce,
            settings.reduce_quarter,
            settings.three,
            settings.nine,
            settings.nt4,
            settings.nt7,
            settings.nt8,
            settings.dt,
        );
    }
}

/// Launches pass 16, resolving the candidates along each gap row into `L` in `directions` and `R` in
/// `rebuilt`, with one unit per gap row.
pub(super) fn lattice_resolve(client: &Client, pass: &FieldPass<'_>, settings: &Eedi2Settings) {
    let gap_rows = pass.rows - 1;
    let (cubes, unit) = row_launch(gap_rows);

    // Safety: the input plane holds `width * 2 * rows` values, and every scratch buffer at least
    // `width * (rows - 1)`, as `FieldPass::new` checked. Each unit owns one gap row, the only row it writes,
    // and reads a neighbour's direction only for columns inside the row.
    unsafe {
        let input = pass.input.array();
        let gaps = pass.gap_array(GapBuffer::Working);
        let values = pass.gap_array(GapBuffer::Candidates);
        let candidate_directions = pass.gap_array(GapBuffer::Spare);
        let used = pass.gap_array(GapBuffer::Directions);
        let rebuilt = pass.gap_array(GapBuffer::Rebuilt);

        lattice_resolve_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            gaps,
            values,
            candidate_directions,
            used,
            rebuilt,
            pass.width,
            gap_rows,
            pass.offset,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}

/// Launches the assembly into `output`, reading `L` in `directions`, `Y2` in `working` and `R` in
/// `rebuilt`.
///
/// `output` must be as large as the input plane.
pub(super) fn assemble(
    client: &Client,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    output: DevicePlane<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.input.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let height = 2 * pass.rows;

    // Safety: the input plane holds `width * 2 * rows` values and every scratch buffer at least
    // `width * (rows - 1)`, as `FieldPass::new` checked, and the caller checked that `output` matches the
    // input. Each unit writes only its own output sample. A rebuilt row lies strictly between two kept rows,
    // so its gap index is below `rows - 1` and the rows it averages exist.
    unsafe {
        let input = pass.input.array();
        let used = pass.gap_array(GapBuffer::Directions);
        let refiltered = pass.gap_array(GapBuffer::Working);
        let rebuilt = pass.gap_array(GapBuffer::Rebuilt);
        let output = output.array();

        assemble_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input,
            used,
            refiltered,
            rebuilt,
            output,
            pass.width,
            height,
            pass.offset,
            settings.pp,
            settings.peak,
            settings.neutral,
            settings.unit,
            settings.reduce_quarter,
        );
    }
}
