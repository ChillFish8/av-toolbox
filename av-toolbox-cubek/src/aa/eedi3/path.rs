// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BACKPOINTERS_PER_WORD, EEDI3_MAX_MDIS, FieldPass, plane_row, row_start, sample_at};
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

/// The absolute differences between the upper rows shifted by `upper_shift` and the lower rows shifted by
/// `lower_shift`, summed over the `2 nrad + 1` columns around `column`.
#[cube]
fn band_sum(
    input: &Array<f32>,
    rows: &KeptRows,
    width: u32,
    column: i32,
    upper_shift: i32,
    lower_shift: i32,
    nrad: u32,
) -> i32 {
    let mut total = 0i32;
    for step in 0..2 * nrad + 1 {
        let tap = step as i32 - nrad as i32;
        let upper = column + upper_shift + tap;
        let lower = column + lower_shift + tap;
        let far_above = sample_at(input, rows.far_above, upper, width);
        let above_upper = sample_at(input, rows.above, upper, width);
        let below_upper = sample_at(input, rows.below, upper, width);
        let above_lower = sample_at(input, rows.above, lower, width);
        let below_lower = sample_at(input, rows.below, lower, width);
        let far_below = sample_at(input, rows.far_below, lower, width);

        total += i32::abs(far_above - above_lower);
        total += i32::abs(above_upper - below_lower);
        total += i32::abs(below_upper - far_below);
    }

    total
}

/// The three band sums of a connection added together.
///
/// The two bands shifted along the connection count only where their shifted column stays inside the row.
/// A band that leaves takes the other one's sum, or the connection's own when both leave.
#[cube]
fn band_total(
    input: &Array<f32>,
    rows: &KeptRows,
    width: u32,
    column: i32,
    direction: i32,
    nrad: u32,
) -> i32 {
    let connection = band_sum(input, rows, width, column, direction, -direction, nrad);
    let left_column = column - 2 * direction;
    let right_column = column + 2 * direction;
    let left_valid = left_column >= 0 && left_column < width as i32;
    let right_valid = right_column >= 0 && right_column < width as i32;

    let mut left = connection;
    let mut right = connection;
    if left_valid {
        left = band_sum(input, rows, width, column, 0, -2 * direction, nrad);
        right = left;
    }

    if right_valid {
        right = band_sum(input, rows, width, column, 2 * direction, 0, nrad);
        if !left_valid {
            left = right;
        }
    }

    connection + left + right
}

/// How far the rows above and below `column` sit from the plain average along `direction`.
#[cube]
fn vertical_difference(input: &Array<f32>, rows: &KeptRows, width: u32, column: i32, direction: i32) -> i32 {
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

/// The cost of connecting `column` of a rebuilt row along `direction`, one rounded f32 operation at a
/// time in the plugin's order.
#[cube]
fn connection_cost(
    input: &Array<f32>,
    rows: &KeptRows,
    width: u32,
    column: i32,
    direction: i32,
    nrad: u32,
    weights: &CostWeights,
) -> f32 {
    let bands = band_total(input, rows, width, column, direction, nrad);
    let difference = vertical_difference(input, rows, width, column, direction);
    let band_term = weights.alpha3 * bands as f32;
    let lean_term = weights.beta * i32::abs(direction) as f32;
    let vertical_term = weights.vertical * difference as f32;
    let partial = band_term + lean_term;

    partial + vertical_term
}

/// How far a path may lean at `column`, the largest direction whose reads stay inside the row.
#[cube]
fn reach(column: u32, width: u32, mdis: u32) -> i32 {
    let room = u32::min(column, width - 1 - column);

    u32::min(room, mdis) as i32
}

/// The path search, one cube per rebuilt row and one unit per direction, which walks the row left to right
/// and records each direction's cheapest predecessor.
///
/// The path costs of the previous column and the current one live in shared memory, in two halves that
/// swap every column, so one barrier per column orders every read before the next write. Each unit packs
/// its backpointers into a word, 2 bits a column holding the predecessor's offset plus 1, and stores it
/// every 16 columns.
#[cube(launch_unchecked)]
fn path_kernel(
    input: &Array<f32>,
    backpointers: &mut Array<u32>,
    width: u32,
    groups: u32,
    height: u32,
    first_rebuilt: u32,
    rows: u32,
    alpha3: f32,
    beta: f32,
    vertical: f32,
    gamma: f32,
    nrad: u32,
    mdis: u32,
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
    let weights = CostWeights {
        alpha3,
        beta,
        vertical,
    };
    let half = comptime!((2 * EEDI3_MAX_MDIS + 1) as usize);
    let mut path_costs = SharedMemory::<f32>::new(comptime!(2 * half));

    let mut word = 0u32;
    for column in 0..width {
        let widest = reach(column, width, mdis);
        let current = (column % 2) as usize * half;
        let previous = half - current;

        if i32::abs(direction) <= widest {
            let cost = connection_cost(input, &kept, width, column as i32, direction, nrad, &weights);

            if column == 0 {
                path_costs[current + lane as usize] = cost;
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

                path_costs[current + lane as usize] = best + cost;
                let code = (chosen - direction + 1) as u32;
                let shift = (column - 1) % BACKPOINTERS_PER_WORD * 2;
                word |= code << shift;
            }
        }

        if column >= 1 {
            let back_column = column - 1;
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

/// Launches the path search over every rebuilt row, writing the backpointers.
pub(super) fn search_paths<R: Runtime>(client: &ComputeClient<R>, pass: &FieldPass<'_>) {
    let span = pass.span();
    let cubes = row_cubes(pass.rows);
    let shape = CubeDim::new_1d(span);
    let weights = pass.settings.weights;

    debug_assert!(span <= 2 * EEDI3_MAX_MDIS + 1, "the settings checked mdis");

    // Safety: the input holds `width * height` f32 values and `backpointers` a `u32` per word of every
    // rebuilt row, as `FieldPass::new` checked. Each cube owns one rebuilt row and each unit one direction,
    // writing only that direction's words of that row. Every read of the input reflects and clamps into
    // the plane.
    unsafe {
        let input = pass.input.array::<R>();
        let backpointers = pass.backpointers_array::<R>();
        let groups = pass.groups();

        path_kernel::launch_unchecked::<R>(
            client,
            cubes,
            shape,
            input,
            backpointers,
            pass.width,
            groups,
            pass.height,
            pass.first_rebuilt,
            pass.rows,
            weights.alpha3,
            weights.beta,
            weights.vertical,
            weights.gamma,
            pass.settings.nrad,
            pass.settings.mdis,
        );
    }
}
