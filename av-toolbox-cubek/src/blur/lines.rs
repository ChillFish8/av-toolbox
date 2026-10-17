// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

/// The index in a plane `width` samples wide of the sample at `position` along `line`.
#[cube]
pub(super) fn sample_index(line: u32, position: u32, width: u32, #[comptime] along_rows: bool) -> u32 {
    let mut index = position * width + line;

    if along_rows {
        index = line * width + position;
    }

    index
}

/// The length of a line and the count of lines in a `width` by `height` plane.
#[cube]
pub(super) fn line_shape(width: u32, height: u32, #[comptime] along_rows: bool) -> (u32, u32) {
    let mut shape = (height, width);

    if along_rows {
        shape = (width, height);
    }

    shape
}

/// The cubes the row pass and the column pass over a `width` by `height` plane take.
///
/// `pass_cubes` gives the cubes of one pass from its line length and line count. The result is
/// `None` when either pass has no cubes.
pub(super) fn both_pass_cubes(
    width: u32,
    height: u32,
    pass_cubes: fn(u32, u32) -> Option<CubeCount>,
) -> Option<(CubeCount, CubeCount)> {
    let row_cubes = pass_cubes(width, height)?;
    let column_cubes = pass_cubes(height, width)?;

    Some((row_cubes, column_cubes))
}
