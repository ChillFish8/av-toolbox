// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{FieldPass, plane_row};
use crate::aa::row_launch;

/// 1 when `column` lies inside the row and its mask sample is set, 0 otherwise.
#[cube]
fn mask_set(mclip: &Array<f32>, row_start: u32, column: i32, width: u32) -> u32 {
    let mut set = 0u32;
    if column >= 0 && column < width as i32 {
        let value = mclip[(row_start + column as u32) as usize];
        if value != 0.0 {
            set = 1u32;
        }
    }

    set
}

/// How many set mask samples lie within `mdis` columns of the column a walk along a mask row has reached.
///
/// A column is active when the count is above 0. Without a mask every column is active.
#[derive(CubeType)]
pub(super) struct ActiveWindow {
    set_nearby: u32,
}

#[cube]
impl ActiveWindow {
    /// A window holding the `count` columns from `first`.
    pub(super) fn new(
        mclip: &Array<f32>,
        row_start: u32,
        first: i32,
        count: u32,
        width: u32,
        #[comptime] masked: bool,
    ) -> ActiveWindow {
        let mut set_nearby = 0u32;
        if comptime!(masked) {
            for step in 0..count {
                let column = first + step as i32;
                set_nearby += mask_set(mclip, row_start, column, width);
            }
        }

        ActiveWindow { set_nearby }
    }

    /// Moves the window one column, taking in `entering` and dropping `leaving`, and returns whether the
    /// column it now surrounds is active.
    pub(super) fn slide(
        &mut self,
        mclip: &Array<f32>,
        row_start: u32,
        entering: i32,
        leaving: i32,
        width: u32,
        #[comptime] masked: bool,
    ) -> bool {
        let mut active = true;
        if comptime!(masked) {
            let entered = mask_set(mclip, row_start, entering, width);
            let left = mask_set(mclip, row_start, leaving, width);
            self.set_nearby = self.set_nearby + entered - left;
            active = self.set_nearby > 0;
        }

        active
    }
}

/// The first column up to `last` whose mask sample is set, or `last + 1` when there is none.
#[cube]
fn first_set_column(mclip: &Array<f32>, row_start: u32, last: u32, width: u32) -> u32 {
    let mut column = 0u32;
    let mut found = false;
    while !found && column <= last {
        found = mask_set(mclip, row_start, column as i32, width) == 1;
        if !found {
            column += 1;
        }
    }

    column
}

/// The bit of a row's edge flags set when its column 0 is active.
const FIRST_ACTIVE: i32 = 1;

/// The bit of a row's edge flags set when its column 1 is active.
const SECOND_ACTIVE: i32 = 2;

/// The edge flags, one unit per rebuilt row, which record whether the row's column 0 and column 1 are
/// active.
///
/// Column 0 is active when a mask sample is set in columns 0 to `mdis`, and column 1 when one is set in
/// columns 0 to `mdis + 1`.
#[cube(launch_unchecked)]
fn edge_flags_kernel(
    mclip: &Array<f32>,
    lenders: &mut Array<i32>,
    width: u32,
    first_rebuilt: u32,
    rows: u32,
    mdis: u32,
) {
    let field_row = ABSOLUTE_POS as u32;
    if field_row >= rows {
        terminate!();
    }

    let row_start = plane_row(field_row, first_rebuilt) * width;
    let nearest = first_set_column(mclip, row_start, mdis + 1, width);

    let mut flags = 0i32;
    if nearest <= mdis {
        flags |= FIRST_ACTIVE;
    }

    if nearest <= mdis + 1 {
        flags |= SECOND_ACTIVE;
    }

    lenders[field_row as usize] = flags;
}

/// The lenders search, one unit per rebuilt row, which finds the nearest rebuilt row above whose
/// column 0 is active and the nearest whose column 1 is active, or -1 where there is none.
///
/// It reads the edge flags of the rows above. A row whose column 0 is active has column 1 active too, so
/// the search stops at the first row lending column 0.
#[cube(launch_unchecked)]
fn lenders_kernel(lenders: &mut Array<i32>, rows: u32) {
    let field_row = ABSOLUTE_POS as u32;
    if field_row >= rows {
        terminate!();
    }

    let mut first_lender = i32::new(-1);
    let mut second_lender = i32::new(-1);
    let mut above = field_row;
    while first_lender < 0 && above > 0 {
        above -= 1;
        let flags = lenders[above as usize];

        if second_lender < 0 && (flags & SECOND_ACTIVE) != 0 {
            second_lender = above as i32;
        }

        if (flags & FIRST_ACTIVE) != 0 {
            first_lender = above as i32;
        }
    }

    let lent_start = rows + 2 * field_row;
    lenders[lent_start as usize] = first_lender;
    lenders[(lent_start + 1) as usize] = second_lender;
}

/// Launches the edge flags over every rebuilt row, writing the first part of the lenders scratch.
pub(super) fn mark_active_edges<R: Runtime>(client: &ComputeClient<R>, pass: &FieldPass<'_>) {
    let (cubes, unit) = row_launch(pass.rows);
    let mask = pass.mask_plane();

    // Safety: the mask holds `width * height` f32 values and `lenders` three `i32` values per rebuilt
    // row, as `FieldPass::new` checked. Each unit writes only its own row's flags, and reads the mask only
    // inside its own row.
    unsafe {
        let mclip = mask.array::<R>();
        let lenders = pass.lenders_array::<R>();

        edge_flags_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            mclip,
            lenders,
            pass.width,
            pass.first_rebuilt,
            pass.rows,
            pass.settings.mdis,
        );
    }
}

/// Launches the lenders search over every rebuilt row, reading the edge flags and writing the lenders.
pub(super) fn find_lenders<R: Runtime>(client: &ComputeClient<R>, pass: &FieldPass<'_>) {
    let (cubes, unit) = row_launch(pass.rows);

    // Safety: `lenders` holds three `i32` values per rebuilt row, as `FieldPass::new` checked. Each unit
    // reads the flags the previous launch wrote for the rows above its own, and writes only its own row's
    // two lenders, after every flag.
    unsafe {
        let lenders = pass.lenders_array::<R>();

        lenders_kernel::launch_unchecked::<R>(client, cubes, unit, lenders, pass.rows);
    }
}
