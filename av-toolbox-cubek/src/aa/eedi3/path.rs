// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::mask::ActiveWindow;
use super::{
    ACTIVE_PER_WORD,
    BACKPOINTERS_PER_WORD,
    EEDI3_MAX_MDIS,
    FieldPass,
    TO_ZERO_CODE,
    plane_row,
    row_start,
    sample_at,
};
use crate::aa::row_cubes;

/// The starts of the four kept rows a rebuilt row reads, each reflected into the plane.
#[derive(CubeType)]
struct KeptRows {
    far_above: u32,
    above: u32,
    below: u32,
    far_below: u32,
}

#[cube]
impl KeptRows {
    fn new(row: u32, width: u32, height: u32) -> KeptRows {
        let signed_row = row as i32;

        KeptRows {
            far_above: row_start(signed_row - 3, width, height),
            above: row_start(signed_row - 1, width, height),
            below: row_start(signed_row + 1, width, height),
            far_below: row_start(signed_row + 3, width, height),
        }
    }
}

/// The three absolute differences of one tap, between the upper rows at column `upper` and the lower rows
/// at column `lower`.
#[cube]
fn column_difference(input: &[f32], rows: &KeptRows, width: u32, upper: i32, lower: i32) -> i32 {
    let far_above = sample_at(input, rows.far_above, upper, width);
    let above_upper = sample_at(input, rows.above, upper, width);
    let below_upper = sample_at(input, rows.below, upper, width);
    let above_lower = sample_at(input, rows.above, lower, width);
    let below_lower = sample_at(input, rows.below, lower, width);
    let far_below = sample_at(input, rows.far_below, lower, width);

    let mut total = i32::abs(far_above - above_lower);
    total += i32::abs(above_upper - below_lower);
    total += i32::abs(below_upper - far_below);

    total
}

/// The absolute differences between the upper rows shifted by `upper_shift` and the lower rows shifted by
/// `lower_shift`, summed over the `2 nrad + 1` columns around `column`.
#[cube]
fn band_sum(
    input: &[f32],
    rows: &KeptRows,
    width: u32,
    column: i32,
    upper_shift: i32,
    lower_shift: i32,
    #[comptime] nrad: u32,
) -> i32 {
    let mut total = 0i32;
    for step in 0..2 * nrad + 1 {
        let tap = step as i32 - nrad as i32;
        let upper = column + upper_shift + tap;
        let lower = column + lower_shift + tap;

        total += column_difference(input, rows, width, upper, lower);
    }

    total
}

/// The three band sums of a connection along `direction` at `column` added together.
///
/// The two bands shifted along the connection count only where their shifted column stays inside the row.
/// A band that leaves takes the other one's sum, or the connection's own when both leave.
#[cube]
fn bands_along(connection: i32, left: i32, right: i32, column: i32, direction: i32, width: u32) -> i32 {
    let left_column = column - 2 * direction;
    let right_column = column + 2 * direction;
    let left_valid = left_column >= 0 && left_column < width as i32;
    let right_valid = right_column >= 0 && right_column < width as i32;

    let mut left_band = connection;
    let mut right_band = connection;
    if left_valid {
        left_band = left;
        right_band = left;
    }

    if right_valid {
        right_band = right;
        if !left_valid {
            left_band = right;
        }
    }

    connection + left_band + right_band
}

/// The three band sums of a connection added together, each summed from scratch.
#[cube]
fn band_total(
    input: &[f32],
    rows: &KeptRows,
    width: u32,
    column: i32,
    direction: i32,
    #[comptime] nrad: u32,
) -> i32 {
    let connection = band_sum(input, rows, width, column, direction, -direction, nrad);
    let left = band_sum(input, rows, width, column, 0, -2i32 * direction, nrad);
    let right = band_sum(input, rows, width, column, 2 * direction, 0, nrad);

    bands_along(connection, left, right, column, direction, width)
}

/// How far the rows above and below `column` sit from the plain average along `direction`.
#[cube]
fn vertical_difference(input: &[f32], rows: &KeptRows, width: u32, column: i32, direction: i32) -> i32 {
    let above = sample_at(input, rows.above, column + direction, width);
    let below = sample_at(input, rows.below, column - direction, width);
    let average = (above + below + 1) >> 1;
    let straight_above = sample_at(input, rows.above, column, width);
    let straight_below = sample_at(input, rows.below, column, width);

    i32::abs(straight_above - average) + i32::abs(straight_below - average)
}

/// The weights of a connection's cost.
#[derive(CubeType)]
struct CostWeights {
    alpha3: f32,
    beta: f32,
    vertical: f32,
}

/// The cost of a connection along `direction` from its band total and vertical difference, one rounded
/// f32 operation at a time in the plugin's order.
#[cube]
fn cost_of(bands: i32, difference: i32, direction: i32, weights: &CostWeights) -> f32 {
    let band_term = weights.alpha3 * bands as f32;
    let lean_term = weights.beta * i32::abs(direction) as f32;
    let vertical_term = weights.vertical * difference as f32;
    let partial = band_term + lean_term;

    partial + vertical_term
}

/// The cost of connecting `column` of a rebuilt row along `direction`.
#[cube]
fn connection_cost(
    input: &[f32],
    rows: &KeptRows,
    width: u32,
    column: i32,
    direction: i32,
    weights: &CostWeights,
    #[comptime] nrad: u32,
) -> f32 {
    let bands = band_total(input, rows, width, column, direction, nrad);
    let difference = vertical_difference(input, rows, width, column, direction);

    cost_of(bands, difference, direction, weights)
}

/// How many bands a connection's cost sums.
const BANDS: u32 = 3;

/// The slot of the left band, which is also its offset from the column in directions.
const LEFT_BAND: usize = 0;

/// The slot of the connection's own band, which is also its offset from the column in directions.
const CONNECTION_BAND: usize = 1;

/// The slot of the right band, which is also its offset from the column in directions.
const RIGHT_BAND: usize = 2;

/// The three bands of a connection's cost, each with a delay line of its taps and their running sum.
///
/// The band in slot `b` reads the upper rows `b` directions past the column, and the lower rows two
/// directions before that. Each delay line holds its band's `2 nrad + 1` taps, left to right.
#[derive(CubeType)]
struct BandSums {
    delays: Array<i32>,
    sums: Array<i32>,
}

#[cube]
impl BandSums {
    fn new(#[comptime] nrad: u32) -> BandSums {
        let window = comptime!(2 * nrad + 1);
        let delays = Array::<i32>::new(comptime!((BANDS * window) as usize));
        let sums = Array::<i32>::new(comptime!(BANDS as usize));

        BandSums { delays, sums }
    }

    /// Fills each band's delay line with its taps around `column`, and sets its sum to their total.
    fn fill(
        &mut self,
        input: &[f32],
        rows: &KeptRows,
        width: u32,
        column: i32,
        direction: i32,
        #[comptime] nrad: u32,
    ) {
        let window = comptime!(2 * nrad + 1);

        #[unroll]
        for band in 0..comptime!(BANDS) {
            let centre = column + band as i32 * direction;
            let mut total = 0i32;

            #[unroll]
            for step in 0..window {
                let upper = centre + step as i32 - nrad as i32;
                let lower = upper - 2 * direction;
                let difference = column_difference(input, rows, width, upper, lower);

                self.delays[(band * window + step) as usize] = difference;
                total += difference;
            }

            self.sums[band as usize] = total;
        }
    }

    /// Moves each band from the column before `column` to `column`, adding the tap that enters its window
    /// and taking away the one that leaves.
    fn slide(
        &mut self,
        input: &[f32],
        rows: &KeptRows,
        width: u32,
        column: i32,
        direction: i32,
        #[comptime] nrad: u32,
    ) {
        let window = comptime!(2 * nrad + 1);

        #[unroll]
        for band in 0..comptime!(BANDS) {
            let start = band * window;
            let upper = column + band as i32 * direction + nrad as i32;
            let lower = upper - 2 * direction;
            let entering = column_difference(input, rows, width, upper, lower);
            let leaving = self.delays[start as usize];

            #[unroll]
            for step in 0..window - 1 {
                let slot = start + step;
                self.delays[slot as usize] = self.delays[(slot + 1) as usize];
            }

            self.delays[(start + window - 1) as usize] = entering;
            self.sums[band as usize] += entering - leaving;
        }
    }

    /// The three band sums added together at `column` along `direction`.
    fn total(&self, column: i32, direction: i32, width: u32) -> i32 {
        let connection = self.sums[CONNECTION_BAND];
        let left = self.sums[LEFT_BAND];
        let right = self.sums[RIGHT_BAND];

        bands_along(connection, left, right, column, direction, width)
    }
}

/// How far a path may lean at `column`, the largest direction whose reads stay inside the row.
#[cube]
fn reach(column: u32, width: u32, mdis: u32) -> i32 {
    let room = u32::min(column, width - 1 - column);

    u32::min(room, mdis) as i32
}

/// The plane geometry a rebuilt row's costs read.
#[derive(CubeType)]
struct RowShape {
    width: u32,
    height: u32,
    first_rebuilt: u32,
}

/// The cost at `column` along `direction` of the rebuilt row that lends it, stored at `slot` of the
/// lenders, or 0 when no row lends it.
#[cube]
fn lent_cost(
    input: &[f32],
    lenders: &[i32],
    slot: u32,
    shape: &RowShape,
    column: i32,
    direction: i32,
    weights: &CostWeights,
    #[comptime] nrad: u32,
) -> f32 {
    let lender = lenders[slot as usize];

    let mut cost = 0.0f32;
    if lender >= 0 {
        let row = plane_row(lender as u32, shape.first_rebuilt);
        let kept = KeptRows::new(row, shape.width, shape.height);
        cost = connection_cost(input, &kept, shape.width, column, direction, weights, nrad);
    }

    cost
}

/// The backpointer code an inactive `column` gives the column before it along `direction`, a copy of
/// `previous_code`.
///
/// Within `mdis` columns of either side, the two widest directions the column before can take instead
/// lean one step back towards 0.
#[cube]
fn copied_code(previous_code: u32, column: u32, width: u32, direction: i32, mdis: u32) -> u32 {
    let edge = u32::min(column - 1, width - column) as i32;

    let mut code = previous_code;
    if edge < mdis as i32 {
        if direction == -edge {
            code = 2u32;
        }

        if direction == edge {
            code = 0u32;
        }
    }

    code
}

/// The path search, one cube per rebuilt row and one unit per direction, which walks the row left to right
/// and records each direction's cheapest predecessor.
///
/// The path costs of the previous column and the current one live in shared memory, in two halves that
/// swap every column, so one barrier per column orders every read before the next write. Each unit packs
/// its backpointers into a word, 2 bits a column, and stores it every 16 columns. A code of 0 to 2 holds
/// the predecessor's offset plus 1, and a code of 3 leads to direction 0.
///
/// Each unit keeps its band sums over the window and slides them a column at a time, refilling them after
/// a column it skips.
///
/// Under a mask, unit 0 records which columns are active, one bit a column, in the lenders scratch after the
/// lenders. An inactive column copies the path state of the previous one. At column 0 and 1 it takes the
/// costs of the rebuilt row above that lends them. A direction a column cannot take keeps a path cost of 0
/// and leads to direction 0.
#[cube(launch_unchecked)]
fn path_kernel(
    input: &[f32],
    mclip: &[f32],
    lenders: &mut [i32],
    backpointers: &mut [u32],
    width: u32,
    groups: u32,
    active_words: u32,
    height: u32,
    first_rebuilt: u32,
    rows: u32,
    alpha3: f32,
    beta: f32,
    vertical: f32,
    gamma: f32,
    mdis: u32,
    #[comptime] nrad: u32,
    #[comptime] masked: bool,
) {
    let field_row = CUBE_POS as u32;
    if field_row >= rows {
        terminate!();
    }

    let lane = UNIT_POS_X;
    let direction = lane as i32 - mdis as i32;
    let span = 2 * mdis + 1;
    let word_start = (field_row * span + lane) * groups;
    let row = plane_row(field_row, first_rebuilt);
    let kept = KeptRows::new(row, width, height);
    let shape = RowShape {
        width,
        height,
        first_rebuilt,
    };
    let weights = CostWeights {
        alpha3,
        beta,
        vertical,
    };
    let half = comptime!((2 * EEDI3_MAX_MDIS + 1) as usize);
    let mut path_costs = Shared::<[f32]>::new_slice(comptime!(2 * half));

    let mask_start = row * width;
    let leaving_distance = mdis as i32 + 1;
    let mut window = ActiveWindow::new(mclip, mask_start, 0, mdis, width, masked);

    let mut band_sums = BandSums::new(nrad);
    let mut warm = false;

    let active_start = 3 * rows + field_row * active_words;
    let mut active_word = 0i32;

    let mut word = 0u32;
    let mut previous_code = u32::new(TO_ZERO_CODE as i64);
    for column in 0..width {
        let widest = reach(column, width, mdis);
        let current = (column % 2) as usize * half;
        let previous = half - current;
        let inside = i32::abs(direction) <= widest;
        let signed_column = column as i32;
        let entering = signed_column + mdis as i32;
        let leaving = signed_column - leaving_distance;
        let active = window.slide(mclip, mask_start, entering, leaving, width, masked);

        if comptime!(masked) {
            let bit = column % ACTIVE_PER_WORD;
            if active {
                active_word |= 1i32 << bit as i32;
            }

            let word_full = bit == ACTIVE_PER_WORD - 1;
            if lane == 0 && (word_full || column + 1 == width) {
                let index = active_start + column / ACTIVE_PER_WORD;
                lenders[index as usize] = active_word;
            }

            if word_full {
                active_word = 0i32;
            }
        }

        // A direction outside the column's reach keeps a path cost of 0. The zeros are read. An inactive
        // run from the left edge copies them from column 1 to its first active column.
        let mut path_cost = 0.0f32;
        let mut code = u32::new(TO_ZERO_CODE as i64);
        if active && inside {
            if warm {
                band_sums.slide(input, &kept, width, signed_column, direction, nrad);
            } else {
                band_sums.fill(input, &kept, width, signed_column, direction, nrad);
            }

            warm = true;

            let bands = band_sums.total(signed_column, direction, width);
            let difference = vertical_difference(input, &kept, width, signed_column, direction);
            let cost = cost_of(bands, difference, direction, &weights);

            if column == 0 {
                path_cost = cost;
            } else {
                let previous_widest = reach(column - 1, width, mdis);
                let mut best = 0.0f32;
                let mut chosen = direction;
                let mut found = false;

                if direction > -previous_widest {
                    best = path_costs[previous + lane as usize - 1] + gamma;
                    chosen = direction - 1;
                    found = true;
                }

                if i32::abs(direction) <= previous_widest {
                    let candidate = path_costs[previous + lane as usize];
                    if !found || candidate < best {
                        best = candidate;
                        chosen = direction;
                        found = true;
                    }
                }

                if direction < previous_widest {
                    let candidate = path_costs[previous + lane as usize + 1] + gamma;
                    if !found || candidate < best {
                        best = candidate;
                        chosen = direction + 1;
                    }
                }

                path_cost = best + cost;
                code = (chosen - direction + 1) as u32;
            }
        } else {
            warm = false;
        }

        if comptime!(masked) {
            let lent_slot = rows + 2 * field_row;

            if !active && column == 0 {
                if direction == 0 {
                    path_cost = lent_cost(input, &*lenders, lent_slot, &shape, 0, 0, &weights, nrad);
                }
            } else if !active && column == 1 {
                if inside {
                    let slot = lent_slot + 1;
                    path_cost = lent_cost(input, &*lenders, slot, &shape, 1, direction, &weights, nrad);
                }
            } else if !active {
                path_cost = path_costs[previous + lane as usize];
                code = copied_code(previous_code, column, width, direction, mdis);
            }
        }

        path_costs[current + lane as usize] = path_cost;

        if column >= 1 {
            let back_column = column - 1;
            let shift = back_column % BACKPOINTERS_PER_WORD * 2;
            word |= code << shift;
            previous_code = code;

            let word_full = back_column % BACKPOINTERS_PER_WORD == BACKPOINTERS_PER_WORD - 1;
            if word_full || back_column + 2 == width {
                let index = word_start + back_column / BACKPOINTERS_PER_WORD;
                backpointers[index as usize] = word;
                word = 0u32;
            }
        }

        sync_cube();
    }
}

/// Launches the path search over every rebuilt row, writing the backpointers and, under a mask, the
/// active bits.
pub(super) fn search_paths(client: &Client, pass: &FieldPass<'_>) {
    let span = pass.span();
    let cubes = row_cubes(pass.rows);
    let shape = CubeDim::new_1d(span);
    let weights = pass.settings.weights;
    let mask = pass.mask_plane();
    let masked = pass.mclip.is_some();

    debug_assert!(span <= 2 * EEDI3_MAX_MDIS + 1, "the settings checked mdis");

    // Safety: the input and the mask hold `width * height` f32 values, `lenders` three `i32` values per
    // rebuilt row then the active words of every rebuilt row, and `backpointers` a `u32` per word of every
    // rebuilt row, as `FieldPass::new` checked. Each cube owns one rebuilt row and each unit one direction,
    // writing only that direction's words of that row. Unit 0 alone writes the row's active words, which
    // lie after every lender. Every read of the input reflects and clamps into the plane, and the mask is
    // read only inside the row. A lender is a rebuilt row above, or -1 and never read.
    unsafe {
        let input = pass.input.array();
        let mclip = mask.array();
        let lenders = pass.lenders_array();
        let backpointers = pass.backpointers_array();
        let groups = pass.groups();
        let active_words = pass.active_words();

        path_kernel::launch_unchecked(
            client,
            cubes,
            shape,
            input,
            mclip,
            lenders,
            backpointers,
            pass.width,
            groups,
            active_words,
            pass.height,
            pass.first_rebuilt,
            pass.rows,
            weights.alpha3,
            weights.beta,
            weights.vertical,
            weights.gamma,
            pass.settings.mdis,
            pass.settings.nrad,
            masked,
        );
    }
}
