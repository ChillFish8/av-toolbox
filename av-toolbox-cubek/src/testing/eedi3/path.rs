use super::touched::close_directions;
use super::{Contraction, Eedi3Parameters, Lending, Settings, TieOrder};

/// The f32 weights of the costs and the reliability check, each one rounded operation in the plugin's
/// order.
pub(super) struct Weights {
    pub(super) alpha3: f32,
    pub(super) beta: f32,
    pub(super) gamma: f32,
    pub(super) vertical: f32,
    pub(super) reciprocal0: f32,
    pub(super) reciprocal1: f32,
    pub(super) reciprocal2: f32,
    pub(super) vthresh2: f32,
}

impl Weights {
    pub(super) fn new(parameters: &Eedi3Parameters) -> Self {
        let unit = (1u32 << (parameters.bits - 8)) as f32;
        let without_alpha = 1.0 - parameters.alpha;
        let threshold0 = parameters.vthresh[0] * unit;
        let threshold1 = parameters.vthresh[1] * unit;

        Self {
            alpha3: parameters.alpha / 3.0,
            beta: parameters.beta * unit,
            gamma: parameters.gamma * unit,
            vertical: without_alpha - parameters.beta,
            reciprocal0: 1.0 / threshold0,
            reciprocal1: 1.0 / threshold1,
            reciprocal2: 1.0 / parameters.vthresh[2],
            vthresh2: parameters.vthresh[2],
        }
    }
}

/// `index` reflected into `0..length` without repeating the edge.
fn reflect_once(index: i32, length: i32) -> i32 {
    let reflected = if index < 0 {
        -index
    } else if index > length - 1 {
        2 * (length - 1) - index
    } else {
        index
    };
    debug_assert!(
        (0..length).contains(&reflected),
        "{index} leaves 0..{length} after one reflection"
    );

    reflected
}

/// A plane of integer samples.
pub(super) struct Grid {
    samples: Vec<i32>,
    pub(super) width: i32,
    pub(super) height: i32,
}

impl Grid {
    pub(super) fn new(plane: &[f32], width: u32, height: u32) -> Self {
        Self {
            samples: plane.iter().map(|value| *value as i32).collect(),
            width: width as i32,
            height: height as i32,
        }
    }

    pub(super) fn samples(&self) -> &[i32] {
        &self.samples
    }

    /// The sample at `column` of `row`, both inside the plane.
    pub(super) fn at(&self, row: i32, column: i32) -> i32 {
        let index = row * self.width + column;

        self.samples[index as usize]
    }

    /// The sample at `column` of `row`, reflecting either once past the plane's edge.
    pub(super) fn reflected(&self, row: i32, column: i32) -> i32 {
        let inside_row = reflect_once(row, self.height);
        let inside_column = reflect_once(column, self.width);

        self.at(inside_row, inside_column)
    }
}

/// Which samples of an 8-bit mask plane are set.
pub(super) struct Mask {
    set: Vec<bool>,
    width: i32,
}

impl Mask {
    pub(super) fn new(mclip: &[f32], width: u32) -> Self {
        Self {
            set: mclip.iter().map(|value| *value != 0.0).collect(),
            width: width as i32,
        }
    }

    /// Whether a set sample lies within `mdis` columns of `column` on `row`.
    fn is_active(&self, row: i32, column: i32, mdis: i32) -> bool {
        let first = (column - mdis).max(0);
        let last = (column + mdis).min(self.width - 1);
        let start = (row * self.width + first) as usize;
        let end = (row * self.width + last) as usize;

        self.set[start..=end].iter().any(|set| *set)
    }
}

/// How far a path may lean at `column`, the largest direction whose reads stay inside the row.
pub(super) fn reach(column: i32, width: i32, mdis: i32) -> i32 {
    column.min(width - 1 - column).min(mdis)
}

/// Values for every column and every direction between `-mdis` and `mdis`.
pub(super) struct Table<T> {
    values: Vec<T>,
    mdis: i32,
}

impl<T: Copy> Table<T> {
    fn filled(width: i32, mdis: i32, value: T) -> Self {
        let length = width * (2 * mdis + 1);

        Self {
            values: vec![value; length as usize],
            mdis,
        }
    }

    fn index(&self, column: i32, direction: i32) -> usize {
        let span = 2 * self.mdis + 1;

        (column * span + direction + self.mdis) as usize
    }

    pub(super) fn get(&self, column: i32, direction: i32) -> T {
        let index = self.index(column, direction);

        self.values[index]
    }

    fn set(&mut self, column: i32, direction: i32, value: T) {
        let index = self.index(column, direction);
        self.values[index] = value;
    }
}

/// The absolute differences between the upper rows shifted by `upper_shift` and the lower rows shifted by
/// `lower_shift`, summed over the `2 nrad + 1` columns around `column`.
fn band_sum(grid: &Grid, row: i32, column: i32, upper_shift: i32, lower_shift: i32, nrad: i32) -> i32 {
    let mut total = 0;
    for tap in -nrad..=nrad {
        let upper = column + upper_shift + tap;
        let lower = column + lower_shift + tap;
        total += (grid.reflected(row - 3, upper) - grid.reflected(row - 1, lower)).abs();
        total += (grid.reflected(row - 1, upper) - grid.reflected(row + 1, lower)).abs();
        total += (grid.reflected(row + 1, upper) - grid.reflected(row + 3, lower)).abs();
    }

    total
}

/// The three band sums of a connection added together.
///
/// The two bands shifted along the connection count only where their shifted column stays inside the row.
/// A band that leaves takes the other one's sum, or the connection's own when both leave.
fn band_total(grid: &Grid, row: i32, column: i32, direction: i32, nrad: i32) -> i32 {
    let connection = band_sum(grid, row, column, direction, -direction, nrad);
    let left_valid = (0..grid.width).contains(&(column - 2 * direction));
    let right_valid = (0..grid.width).contains(&(column + 2 * direction));

    let left = if left_valid {
        band_sum(grid, row, column, 0, -2 * direction, nrad)
    } else {
        0
    };
    let right = if right_valid {
        band_sum(grid, row, column, 2 * direction, 0, nrad)
    } else {
        0
    };

    let (left, right) = match (left_valid, right_valid) {
        (true, true) => (left, right),
        (true, false) => (left, left),
        (false, true) => (right, right),
        (false, false) => (connection, connection),
    };

    connection + left + right
}

/// How far the rows above and below `column` sit from the plain average along `direction`.
fn vertical_difference(grid: &Grid, row: i32, column: i32, direction: i32) -> i32 {
    let above = grid.reflected(row - 1, column + direction);
    let below = grid.reflected(row + 1, column - direction);
    let average = (above + below + 1) >> 1;
    let straight_above = grid.reflected(row - 1, column);
    let straight_below = grid.reflected(row + 1, column);

    (straight_above - average).abs() + (straight_below - average).abs()
}

/// The cost of connecting `column` of the rebuilt `row` along `direction`.
pub(super) fn connection_cost(
    grid: &Grid,
    row: i32,
    column: i32,
    direction: i32,
    settings: &Settings,
) -> f32 {
    let bands = band_total(grid, row, column, direction, settings.nrad);
    let vertical = vertical_difference(grid, row, column, direction);
    let bands = bands as f32;
    let lean = direction.abs() as f32;
    let vertical = vertical as f32;
    let weights = &settings.weights;

    match settings.variant.contraction {
        Contraction::Plain => {
            let band_term = weights.alpha3 * bands;
            let lean_term = weights.beta * lean;
            let vertical_term = weights.vertical * vertical;
            let partial = band_term + lean_term;
            partial + vertical_term
        },
        Contraction::FuseLater => {
            let band_term = weights.alpha3 * bands;
            let partial = weights.beta.mul_add(lean, band_term);
            weights.vertical.mul_add(vertical, partial)
        },
        Contraction::FuseEarlier => {
            let lean_term = weights.beta * lean;
            let partial = weights.alpha3.mul_add(bands, lean_term);
            weights.vertical.mul_add(vertical, partial)
        },
    }
}

/// Whether each column of `row` is searched. Without a mask every column is.
fn active_columns(mask: Option<&Mask>, row: i32, width: i32, mdis: i32) -> Vec<bool> {
    match mask {
        Some(mask) => (0..width)
            .map(|column| mask.is_active(row, column, mdis))
            .collect(),
        None => vec![true; width as usize],
    }
}

/// The connection costs of every active column of `row`, 0 elsewhere.
fn row_costs(grid: &Grid, row: i32, active: &[bool], settings: &Settings) -> Table<f32> {
    let mut costs = Table::filled(grid.width, settings.mdis, 0.0);
    for column in 0..grid.width {
        if !active[column as usize] {
            continue;
        }

        let widest = reach(column, grid.width, settings.mdis);
        for direction in -widest..=widest {
            let cost = connection_cost(grid, row, column, direction, settings);
            costs.set(column, direction, cost);
        }
    }

    costs
}

/// Gives an inactive column 0 or 1 the costs of the nearest rebuilt row above where it is active.
///
/// Such a column keeps costs of 0 when no row above has it active.
fn lend_costs(
    grid: &Grid,
    mask: &Mask,
    row: i32,
    active: &[bool],
    costs: &mut Table<f32>,
    settings: &Settings,
) {
    for column in 0..=1 {
        if active[column as usize] {
            continue;
        }

        let lender = (1..=row / 2)
            .map(|step| row - 2 * step)
            .find(|above| mask.is_active(*above, column, settings.mdis));
        let Some(lender) = lender else {
            continue;
        };

        let widest = reach(column, grid.width, settings.mdis);
        for direction in -widest..=widest {
            let cost = connection_cost(grid, lender, column, direction, settings);
            costs.set(column, direction, cost);
        }
    }
}

/// The running path costs, the backpointers into the previous column, and the candidates each step
/// compared.
///
/// `candidates` at `column` and `direction` holds the step into `column + 1`, one slot per previous
/// direction from `direction - 1` to `direction + 1`, and infinity where there is no such candidate.
pub(super) struct PathTables {
    costs: Table<f32>,
    pub(super) back: Table<i32>,
    pub(super) candidates: Table<[f32; 3]>,
}

/// Whether `candidate` beats the best candidate so far.
fn beats(candidate: f32, best: Option<f32>, ties: TieOrder) -> bool {
    let Some(best) = best else {
        return true;
    };

    match ties {
        TieOrder::SmallestDirection => candidate < best,
        TieOrder::LargestDirection => candidate <= best,
    }
}

/// One path step into the active `column`, choosing each direction's cheapest predecessor.
fn searched_step(tables: &mut PathTables, costs: &Table<f32>, column: i32, width: i32, settings: &Settings) {
    let widest = reach(column, width, settings.mdis);
    let previous_widest = reach(column - 1, width, settings.mdis);

    for direction in -widest..=widest {
        let first = (direction - 1).max(-previous_widest);
        let last = (direction + 1).min(previous_widest);

        let mut best = None;
        let mut chosen = 0;
        let mut compared = [f32::INFINITY; 3];
        for previous in first..=last {
            let previous_cost = tables.costs.get(column - 1, previous);
            let candidate = if previous == direction {
                previous_cost
            } else {
                previous_cost + settings.weights.gamma
            };
            compared[(previous - direction + 1) as usize] = candidate;

            if beats(candidate, best, settings.variant.ties) {
                best = Some(candidate);
                chosen = previous;
            }
        }

        let best = best.expect("every direction has a predecessor");
        let cost = costs.get(column, direction);
        let total = best + cost;
        tables.costs.set(column, direction, total);
        tables.back.set(column - 1, direction, chosen);
        tables.candidates.set(column - 1, direction, compared);
    }
}

/// One path step into the inactive `column`, which carries the path state across without searching.
fn skipped_step(tables: &mut PathTables, costs: &Table<f32>, column: i32, width: i32, mdis: i32) {
    if column == 1 {
        let widest = 1.min(width - 2).min(mdis);
        for direction in -widest..=widest {
            let cost = costs.get(1, direction);
            tables.costs.set(1, direction, cost);
        }

        for direction in -mdis..=mdis {
            tables.back.set(0, direction, 0);
        }

        return;
    }

    for direction in -mdis..=mdis {
        let cost = tables.costs.get(column - 1, direction);
        let back = tables.back.get(column - 2, direction);
        tables.costs.set(column, direction, cost);
        tables.back.set(column - 1, direction, back);
    }

    let edge = (column - 1).min(width - column);
    if edge < mdis {
        tables.back.set(column - 1, -edge, 1 - edge);
        tables.back.set(column - 1, edge, edge - 1);
    }
}

/// The cheapest path along the row, where each step may change direction by at most 1 at a cost of
/// `gamma`.
fn cheapest_path(costs: &Table<f32>, active: &[bool], width: i32, settings: &Settings) -> PathTables {
    let mdis = settings.mdis;
    let mut tables = PathTables {
        costs: Table::filled(width, mdis, 0.0),
        back: Table::filled(width, mdis, 0),
        candidates: Table::filled(width, mdis, [f32::INFINITY; 3]),
    };

    let start = costs.get(0, 0);
    tables.costs.set(0, 0, start);
    for column in 1..width {
        if active[column as usize] {
            searched_step(&mut tables, costs, column, width, settings);
        } else {
            skipped_step(&mut tables, costs, column, width, mdis);
        }
    }

    tables
}

/// Follows the backpointers from direction 0 at the last column.
fn backtrack(back: &Table<i32>, width: i32) -> Vec<i32> {
    let mut path = vec![0; width as usize];
    for column in (0..width - 1).rev() {
        let next = path[column as usize + 1];
        path[column as usize] = back.get(column, next);
    }

    path
}

/// The directions one rebuilt row is interpolated along, and which of them a close decision can change.
///
/// An inactive column takes direction 0.
pub(super) struct RowPath {
    pub(super) directions: Vec<i32>,
    pub(super) touched: Vec<bool>,
}

/// Searches the rebuilt `row` for its cheapest path.
pub(super) fn search_row(grid: &Grid, mask: Option<&Mask>, row: i32, settings: &Settings) -> RowPath {
    let active = active_columns(mask, row, grid.width, settings.mdis);

    let mut costs = row_costs(grid, row, &active, settings);
    if let Some(mask) = mask
        && settings.variant.lending == Lending::On
    {
        lend_costs(grid, mask, row, &active, &mut costs, settings);
    }

    let tables = cheapest_path(&costs, &active, grid.width, settings);
    let path = backtrack(&tables.back, grid.width);
    let touched = close_directions(&tables, &active, grid.width, settings.mdis);
    let directions = path
        .iter()
        .zip(&active)
        .map(|(direction, active)| if *active { *direction } else { 0 })
        .collect();

    RowPath { directions, touched }
}
