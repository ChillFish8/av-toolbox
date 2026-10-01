use super::path::Grid;
use super::{CheckOrder, Settings, eedi3_checked_rows};

/// The planes of one field pass as it is built, each at the input's size.
pub(super) struct PassPlanes {
    pub(super) output: Vec<i32>,
    pub(super) directions: Vec<i32>,
    pub(super) touched: Vec<bool>,
    pub(super) touched_directions: Vec<bool>,
}

impl PassPlanes {
    /// Planes holding the input, with no direction and nothing touched.
    pub(super) fn kept(grid: &Grid) -> Self {
        let samples = grid.samples().len();

        Self {
            output: grid.samples().to_vec(),
            directions: vec![0; samples],
            touched: vec![false; samples],
            touched_directions: vec![false; samples],
        }
    }
}

/// `floor(numerator / 16)`, clamped to the sample range.
fn clamped_sixteenth(numerator: i32, peak: i32) -> i32 {
    let value = numerator.div_euclid(16);

    value.clamp(0, peak)
}

/// The sample between `row - 1` and `row + 1` at `column`, interpolated along `direction`.
///
/// It is a cubic through the four kept rows, or the average of the nearest two where the outer rows'
/// taps would leave the row.
fn interpolate(grid: &Grid, row: i32, column: i32, direction: i32, peak: i32) -> i32 {
    let above = grid.reflected(row - 1, column + direction);
    let below = grid.reflected(row + 1, column - direction);
    let outer_reach = 3 * direction.abs();
    let cubic_fits = outer_reach <= column && column <= grid.width - 1 - outer_reach;
    if !cubic_fits {
        return (above + below + 1) >> 1;
    }

    let far_above = grid.reflected(row - 3, column + 3 * direction);
    let far_below = grid.reflected(row + 3, column - 3 * direction);
    let numerator = 9 * (above + below) - (far_above + far_below) + 8;
    clamped_sixteenth(numerator, peak)
}

/// Every sample of the rebuilt `row`, interpolated along `directions`.
pub(super) fn interpolate_row(grid: &Grid, row: i32, directions: &[i32], peak: i32) -> Vec<i32> {
    (0..grid.width)
        .map(|column| {
            let direction = directions[column as usize];
            interpolate(grid, row, column, direction, peak)
        })
        .collect()
}

/// The value the check falls back to at `column` of `row`, from `sclip` or the vertical cubic.
fn fallback(grid: &Grid, sclip: Option<&Grid>, row: i32, column: i32, peak: i32) -> i32 {
    if let Some(sclip) = sclip {
        return sclip.at(row, column);
    }

    let near = grid.reflected(row - 1, column) + grid.reflected(row + 1, column);
    let far = grid.reflected(row - 3, column) + grid.reflected(row + 3, column);
    let numerator = 9 * near - far + 8;
    clamped_sixteenth(numerator, peak)
}

/// How far the interpolation strays, combined the way `vcheck` picks.
fn combine(first: i32, second: i32, vcheck: u32) -> i32 {
    match vcheck {
        1 => first.min(second),
        2 => (first + second + 1) >> 1,
        _ => first.max(second),
    }
}

/// `keep * sample + weight * fallback`, each operation rounded on its own.
fn blend(keep: f32, sample: i32, weight: f32, fallback: i32) -> f32 {
    let sample = sample as f32;
    let fallback = fallback as f32;
    let kept_part = keep * sample;
    let fallback_part = weight * fallback;

    kept_part + fallback_part
}

/// The planes one checked sample reads, by row and column.
///
/// `above` is the plane the row two above is read from. `touched` holds the marks of every row, with the
/// rows already checked holding their checked marks.
struct CheckReads<'a> {
    width: i32,
    output: &'a [i32],
    above: &'a [i32],
    directions: &'a [i32],
    touched: &'a [bool],
    touched_directions: &'a [bool],
}

impl CheckReads<'_> {
    fn index(&self, row: i32, column: i32) -> usize {
        (row * self.width + column) as usize
    }

    fn output(&self, row: i32, column: i32) -> i32 {
        self.output[self.index(row, column)]
    }

    fn above(&self, row: i32, column: i32) -> i32 {
        self.above[self.index(row, column)]
    }

    fn direction(&self, row: i32, column: i32) -> i32 {
        self.directions[self.index(row, column)]
    }

    fn touched(&self, row: i32, column: i32) -> bool {
        self.touched[self.index(row, column)]
    }

    fn direction_touched(&self, row: i32, column: i32) -> bool {
        self.touched_directions[self.index(row, column)]
    }
}

/// A checked sample and whether a close decision can change it.
struct Checked {
    value: i32,
    touched: bool,
}

/// Checks `column` of the rebuilt `row` against its neighbours along its direction, blending towards
/// the fallback where the interpolation looks unreliable.
fn check_sample(
    grid: &Grid,
    reads: &CheckReads<'_>,
    fallback: i32,
    row: i32,
    column: i32,
    settings: &Settings,
) -> Checked {
    let direction = reads.direction(row, column);
    let neighbour_touched =
        reads.direction_touched(row - 2, column) || reads.direction_touched(row + 2, column);
    let path_touched = reads.direction_touched(row, column) || (direction != 0 && neighbour_touched);
    let unchanged = Checked {
        value: fallback,
        touched: path_touched,
    };
    if direction == 0 {
        return unchanged;
    }

    let direction_above = reads.direction(row - 2, column);
    let direction_below = reads.direction(row + 2, column);
    let opposed = (direction * direction_above).max(direction * direction_below) < 0;
    let neither = direction_above == 0 && direction_below == 0;
    if opposed || neither {
        return unchanged;
    }

    let ahead = column + direction;
    let behind = column - direction;
    let kept_above = grid.at(row - 1, column);
    let kept_below = grid.at(row + 1, column);
    let kept_above_ahead = grid.at(row - 1, ahead);
    let kept_below_behind = grid.at(row + 1, behind);
    let checked_above = reads.above(row - 2, ahead);
    let here = reads.output(row, column);
    let here_ahead = reads.output(row, ahead);
    let here_behind = reads.output(row, behind);
    let unchecked_below = reads.output(row + 2, behind);

    let top = (checked_above + here_behind + 1) >> 1;
    let bottom = (here_ahead + unchecked_below + 1) >> 1;
    let top_variation = (checked_above - kept_above_ahead).abs() + (here_ahead - kept_above_ahead).abs();
    let bottom_variation =
        (unchecked_below - kept_below_behind).abs() + (here_behind - kept_below_behind).abs();
    let centre_variation = (here - kept_above).abs() + (here - kept_below).abs();
    let top_offset = (top - kept_above).abs();
    let bottom_offset = (bottom - kept_below).abs();
    let top_change = (top_variation - centre_variation).abs();
    let bottom_change = (bottom_variation - centre_variation).abs();
    let offset = combine(top_offset, bottom_offset, settings.vcheck);
    let change = combine(top_change, bottom_change, settings.vcheck);

    let weights = &settings.weights;
    let offset_weight = offset as f32 * weights.reciprocal0;
    let change_weight = change as f32 * weights.reciprocal1;
    let lean_left = weights.vthresh2 - direction.abs() as f32;
    let lean_weight = lean_left * weights.reciprocal2;
    let lean_weight = lean_weight.max(0.0);
    let strongest = offset_weight.max(change_weight).max(lean_weight);
    let weight = strongest.min(1.0);
    let keep = 1.0 - weight;
    let blended = blend(keep, here, weight, fallback);

    let reads_touched = reads.touched(row - 2, ahead)
        || reads.touched(row, behind)
        || reads.touched(row, ahead)
        || reads.touched(row + 2, behind)
        || reads.touched(row, column);

    Checked {
        value: blended as i32,
        touched: path_touched || reads_touched,
    }
}

/// The reliability check, down the rebuilt rows that have a rebuilt row two above and two below.
///
/// Each row reads the row above after its own check, and itself and the row below before theirs.
pub(super) fn reliability_check(
    grid: &Grid,
    sclip: Option<&Grid>,
    planes: &mut PassPlanes,
    first_rebuilt: i32,
    settings: &Settings,
) {
    let unchecked = planes.output.clone();
    let width = grid.width as usize;

    let rows = eedi3_checked_rows(first_rebuilt as u32, grid.height as u32);
    for checked_row in rows {
        let row = checked_row as i32;
        let above = match settings.variant.check_order {
            CheckOrder::CheckedAbove => &planes.output,
            CheckOrder::UncheckedAbove => &unchecked,
        };
        let reads = CheckReads {
            width: grid.width,
            output: &planes.output,
            above,
            directions: &planes.directions,
            touched: &planes.touched,
            touched_directions: &planes.touched_directions,
        };
        let checked: Vec<Checked> = (0..grid.width)
            .map(|column| {
                let fallback = fallback(grid, sclip, row, column, settings.peak);
                check_sample(grid, &reads, fallback, row, column, settings)
            })
            .collect();

        let start = row as usize * width;
        for (offset, sample) in checked.iter().enumerate() {
            planes.output[start + offset] = sample.value;
            planes.touched[start + offset] = sample.touched;
        }
    }
}
