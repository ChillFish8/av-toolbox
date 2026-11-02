// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// The side of the square tile a cube transposes through shared memory.
pub(super) const TILE: u32 = 32;
/// The rows of units in a cube, each unit moving `TILE / TILE_ROWS` samples.
pub(super) const TILE_ROWS: u32 = 8;
/// The length of a staged tile row.
///
/// It is one sample longer than the tile, so a column of the tile falls in different banks.
pub(super) const TILE_STRIDE: u32 = TILE + 1;
/// The most cubes every device allows along one axis.
const MAX_CUBES_PER_AXIS: u32 = 65535;

/// The cubes giving each `TILE` square of a `width` by `height` plane a cube of its own.
///
/// It is `None` when a plane needs more cubes along an axis than every device allows.
pub(super) fn tile_cubes(width: u32, height: u32) -> Option<CubeCount> {
    let tiles_across = width.div_ceil(TILE);
    let tiles_down = height.div_ceil(TILE);
    if tiles_across > MAX_CUBES_PER_AXIS || tiles_down > MAX_CUBES_PER_AXIS {
        return None;
    }

    Some(CubeCount::Static(tiles_across, tiles_down, 1))
}

/// The shape of a cube moving one `TILE` square.
pub(super) fn tile_shape() -> CubeDim {
    CubeDim::new_2d(TILE, TILE_ROWS)
}

/// Writes a staged `tile` of a `width` by `height` plane into `output` with its axes swapped.
///
/// The tile starts at `first_column` and `first_row` of the plane. The output is `height` wide, so the
/// writes run along the output's rows. Only samples inside the plane are written.
#[cube]
pub(super) fn write_turned_tile(
    tile: &Shared<[f32]>,
    output: &mut [f32],
    first_column: u32,
    first_row: u32,
    width: u32,
    height: u32,
) {
    // A plane row is an output column.
    let output_column = first_row + UNIT_POS_X;

    #[unroll]
    for step in 0..comptime!(TILE / TILE_ROWS) {
        let local_column = UNIT_POS_Y + step * TILE_ROWS;
        let output_row = first_column + local_column;
        if output_column < height && output_row < width {
            let staged = tile[(UNIT_POS_X * TILE_STRIDE + local_column) as usize];
            output[(output_row * height + output_column) as usize] = staged;
        }
    }
}

#[cube(launch_unchecked)]
fn transpose_kernel(input: &[f32], output: &mut [f32], input_width: u32, input_height: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % input_height;
    let row = position / input_height;
    output[ABSOLUTE_POS] = input[(column * input_width + row) as usize];
}

/// The transpose of one `TILE` square, staged through shared memory so both the reads and the writes
/// run along rows.
#[cube(launch_unchecked)]
fn tiled_transpose_kernel(input: &[f32], output: &mut [f32], input_width: u32, input_height: u32) {
    let mut tile = Shared::<[f32]>::new_slice(comptime!((TILE * TILE_STRIDE) as usize));
    let first_column = CUBE_POS_X * TILE;
    let first_row = CUBE_POS_Y * TILE;
    let column = first_column + UNIT_POS_X;

    #[unroll]
    for step in 0..comptime!(TILE / TILE_ROWS) {
        let local_row = UNIT_POS_Y + step * TILE_ROWS;
        let row = first_row + local_row;
        if column < input_width && row < input_height {
            let sample = input[(row * input_width + column) as usize];
            tile[(local_row * TILE_STRIDE + UNIT_POS_X) as usize] = sample;
        }
    }

    sync_cube();

    write_turned_tile(&tile, output, first_column, first_row, input_width, input_height);
}

/// Swaps the axes of `input`, so `output[x][y] = input[y][x]`.
///
/// The output is `input.height()` wide and `input.width()` tall.
pub fn transpose(client: &Client, dispatch: &impl Dispatch, input: DevicePlane<'_>, output: DevicePlane<'_>) {
    let input_width = input.width();
    let input_height = input.height();

    assert!(
        output.width() == input_height,
        "the output width {} must equal the input height {input_height}",
        output.width(),
    );
    assert!(
        output.height() == input_width,
        "the output height {} must equal the input width {input_width}",
        output.height(),
    );

    let Some(cubes) = tile_cubes(input_width, input_height) else {
        transpose_per_sample(client, dispatch, input, output);
        return;
    };

    let input_array = input.array();
    let output_array = output.array();
    let unit = tile_shape();

    // Safety: the output holds its own samples, and each tile reads and writes only the positions its
    // guards place inside the asserted input and output sizes.
    unsafe {
        tiled_transpose_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            input_width,
            input_height,
        );
    }
}

/// Launches one unit per output sample, for a plane too large for one tile per cube.
fn transpose_per_sample(
    client: &Client,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let input_width = input.width();
    let input_height = input.height();
    let samples = output.samples();
    let input_array = input.array();
    let output_array = output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the output holds its own samples, and every read lands inside the input because the
    // asserted sizes make each output position map to an input position.
    unsafe {
        transpose_kernel::launch_unchecked(
            client,
            cubes,
            unit,
            input_array,
            output_array,
            input_width,
            input_height,
        );
    }
}
