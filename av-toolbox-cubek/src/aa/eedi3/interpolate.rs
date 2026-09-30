// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BACKPOINTERS_PER_WORD, FieldPass, plane_row, row_start, sample_at};
use crate::Dispatch;
use crate::aa::row_launch;

/// The backtrack, one unit per rebuilt row, which follows the backpointers from direction 0 at the last
/// column back to the first.
///
/// The direction it follows stays between `-mdis` and `mdis` even on a backpointer the path search never
/// wrote, so every word it reads lies inside the row.
#[cube(launch_unchecked)]
fn backtrack_kernel(
    backpointers: &Array<u32>,
    directions: &mut Array<i32>,
    width: u32,
    groups: u32,
    rows: u32,
    mdis: u32,
) {
    let field_row = ABSOLUTE_POS as u32;
    if field_row >= rows {
        terminate!();
    }

    let span = 2 * mdis + 1;
    let row_words = field_row * span * groups;
    let first_sample = field_row * width;
    let widest = mdis as i32;

    let mut direction = 0i32;
    directions[(first_sample + width - 1) as usize] = 0i32;
    for step in 1..width {
        let column = width - 1 - step;
        let lane = (direction + widest) as u32;
        let index = row_words + lane * groups + column / BACKPOINTERS_PER_WORD;
        let word = backpointers[index as usize];
        let shift = column % BACKPOINTERS_PER_WORD * 2;
        let code = (word >> shift) & 3u32;
        let followed = direction + code as i32 - 1;

        direction = i32::clamp(followed, -widest, widest);
        directions[(first_sample + column) as usize] = direction;
    }
}

/// `floor(numerator / 16)`, clamped to the sample range.
#[cube]
pub(super) fn clamped_sixteenth(numerator: i32, peak: u32) -> i32 {
    let at_least_zero = i32::max(numerator, 0);

    i32::min(at_least_zero / 16, peak as i32)
}

/// The interpolation, one unit per rebuilt sample, along the sample's direction.
///
/// It is a cubic through the four kept rows, or the average of the nearest two where the outer rows' taps
/// would leave the row.
#[cube(launch_unchecked)]
fn interpolate_kernel(
    input: &Array<f32>,
    directions: &Array<i32>,
    rebuilt: &mut Array<f32>,
    width: u32,
    height: u32,
    first_rebuilt: u32,
    peak: u32,
) {
    if ABSOLUTE_POS >= rebuilt.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let field_row = position / width;
    let column = position % width;
    let row = plane_row(field_row, first_rebuilt) as i32;
    let direction = directions[ABSOLUTE_POS];
    let signed_column = column as i32;

    let above_start = row_start(row - 1, width, height);
    let below_start = row_start(row + 1, width, height);
    let above = sample_at(input, above_start, signed_column + direction, width);
    let below = sample_at(input, below_start, signed_column - direction, width);
    let outer_reach = 3 * i32::abs(direction);
    let cubic_fits = outer_reach <= signed_column && signed_column <= width as i32 - 1 - outer_reach;

    let mut value = (above + below + 1) >> 1;
    if cubic_fits {
        let far_above_start = row_start(row - 3, width, height);
        let far_below_start = row_start(row + 3, width, height);
        let far_above = sample_at(input, far_above_start, signed_column + 3 * direction, width);
        let far_below = sample_at(input, far_below_start, signed_column - 3 * direction, width);
        let numerator = 9 * (above + below) - (far_above + far_below) + 8;
        value = clamped_sixteenth(numerator, peak);
    }

    rebuilt[ABSOLUTE_POS] = value as f32;
}

/// Launches the backtrack over every rebuilt row, writing the directions.
pub(super) fn backtrack<R: Runtime>(client: &ComputeClient<R>, pass: &FieldPass<'_>) {
    let (cubes, unit) = row_launch(pass.rows);

    // Safety: `backpointers` holds a `u32` per word of every rebuilt row and `directions` an `i32` per
    // rebuilt sample, as `FieldPass::new` checked. Each unit owns one rebuilt row, the only row it reads
    // and writes, and keeps the direction it follows between `-mdis` and `mdis`.
    unsafe {
        let backpointers = pass.backpointers_array::<R>();
        let directions = pass.directions_array::<R>();
        let groups = pass.groups();

        backtrack_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            backpointers,
            directions,
            pass.width,
            groups,
            pass.rows,
            pass.settings.mdis,
        );
    }
}

/// Launches the interpolation of every rebuilt sample into the rebuilt rows scratch.
pub(super) fn interpolate<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
) {
    let samples = pass.rebuilt_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input holds `width * height` f32 values, and `directions` and `rebuilt` a value per
    // rebuilt sample, as `FieldPass::new` checked. Each unit writes only its own rebuilt sample, and every
    // read of the input reflects and clamps into the plane.
    unsafe {
        let input = pass.input.array::<R>();
        let directions = pass.directions_array::<R>();
        let rebuilt = pass.rebuilt_array::<R>();

        interpolate_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            directions,
            rebuilt,
            pass.width,
            pass.height,
            pass.first_rebuilt,
            pass.settings.peak,
        );
    }
}
