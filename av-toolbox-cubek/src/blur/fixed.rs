// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{FilterPlanes, Storage, capped_tile_cubes};
use crate::border::fold;
use crate::{Dispatch, Edge, assert_plane_sized};

/// The units along a cube's rows.
pub(crate) const UNITS_ACROSS: u32 = 32;
/// The units down a cube's columns.
pub(crate) const UNITS_DOWN: u32 = 8;
/// The outputs each unit writes down its column.
pub(crate) const OUTPUTS_DOWN: u32 = 8;
/// The width of the output tile a cube writes.
pub(crate) const TILE_WIDTH: u32 = UNITS_ACROSS;
/// The height of the output tile a cube writes.
pub(crate) const TILE_HEIGHT: u32 = UNITS_DOWN * OUTPUTS_DOWN;

/// A 3x3 blur.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Smooth {
    /// `1 2 1 / 2 4 2 / 1 2 1` over 16.
    Binomial,
    /// Every weight 1, over 9.
    Box,
}

/// The weighted sum of the three samples at `left`, `centre` and `right` in `row`.
///
/// It weighs them `1 2 1` for the binomial and `1 1 1` for the box.
#[cube]
fn row_sum<F: Float>(
    input: &Array<F>,
    width: u32,
    left: u32,
    centre: u32,
    right: u32,
    row: u32,
    #[comptime] binomial: bool,
) -> f32 {
    let row_start = row * width;
    let left_value = f32::cast_from(input[(row_start + left) as usize]);
    let centre_value = f32::cast_from(input[(row_start + centre) as usize]);
    let right_value = f32::cast_from(input[(row_start + right) as usize]);

    if binomial {
        left_value + 2.0 * centre_value + right_value
    } else {
        left_value + centre_value + right_value
    }
}

/// Writes the outputs of one unit, a short run down one column of one tile.
///
/// Each row's sum is read once and slides down through `above`, `centre` and `below`.
#[cube]
fn smooth_run<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    column: u32,
    first_row: u32,
    #[comptime] binomial: bool,
    #[comptime] clamp: bool,
) {
    let left = fold(column as i32 - 1, width, clamp);
    let right = fold(column as i32 + 1, width, clamp);

    let above_row = fold(first_row as i32 - 1, height, clamp);
    let centre_row = fold(first_row as i32, height, clamp);
    let mut above = row_sum(input, width, left, column, right, above_row, binomial);
    let mut centre = row_sum(input, width, left, column, right, centre_row, binomial);

    #[unroll]
    for step in 0..OUTPUTS_DOWN {
        let row = first_row + step;
        let below_row = fold(row as i32 + 1, height, clamp);
        let below = row_sum(input, width, left, column, right, below_row, binomial);

        if row < height {
            let blurred = if binomial {
                (above + 2.0 * centre + below) / 16.0
            } else {
                (above + centre + below) / 9.0
            };
            output[(row * width + column) as usize] = F::cast_from(blurred);
        }

        above = centre;
        centre = below;
    }
}

/// Writes the outputs of one unit of the vertical `1 2 1`, a short run down one column of one tile.
#[cube]
fn vertical_run<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    column: u32,
    first_row: u32,
    #[comptime] clamp: bool,
) {
    let above_row = fold(first_row as i32 - 1, height, clamp);
    let centre_row = fold(first_row as i32, height, clamp);
    let mut above = f32::cast_from(input[(above_row * width + column) as usize]);
    let mut centre = f32::cast_from(input[(centre_row * width + column) as usize]);

    #[unroll]
    for step in 0..OUTPUTS_DOWN {
        let row = first_row + step;
        let below_row = fold(row as i32 + 1, height, clamp);
        let below = f32::cast_from(input[(below_row * width + column) as usize]);

        if row < height {
            let blurred = (above + 2.0 * centre + below) / 4.0;
            output[(row * width + column) as usize] = F::cast_from(blurred);
        }

        above = centre;
        centre = below;
    }
}

/// Runs a 3x3 blur, each cube walking the tiles its grid position lands on.
///
/// A cube usually has one tile. It walks more only when the plane needs more cubes than a device
/// allows along an axis.
#[cube(launch_unchecked)]
fn smooth_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    #[comptime] binomial: bool,
    #[comptime] clamp: bool,
) {
    let mut tile_down = CUBE_POS_Y;
    while tile_down * TILE_HEIGHT < height {
        let first_row = tile_down * TILE_HEIGHT + UNIT_POS_Y * OUTPUTS_DOWN;
        let mut tile_across = CUBE_POS_X;

        while tile_across * TILE_WIDTH < width {
            let column = tile_across * TILE_WIDTH + UNIT_POS_X;
            if column < width && first_row < height {
                smooth_run(input, output, width, height, column, first_row, binomial, clamp);
            }

            tile_across += CUBE_COUNT_X;
        }

        tile_down += CUBE_COUNT_Y;
    }
}

/// Runs the vertical `1 2 1`, each cube walking the tiles its grid position lands on.
#[cube(launch_unchecked)]
fn vertical_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    #[comptime] clamp: bool,
) {
    let mut tile_down = CUBE_POS_Y;
    while tile_down * TILE_HEIGHT < height {
        let first_row = tile_down * TILE_HEIGHT + UNIT_POS_Y * OUTPUTS_DOWN;
        let mut tile_across = CUBE_POS_X;

        while tile_across * TILE_WIDTH < width {
            let column = tile_across * TILE_WIDTH + UNIT_POS_X;
            if column < width && first_row < height {
                vertical_run(input, output, width, height, column, first_row, clamp);
            }

            tile_across += CUBE_COUNT_X;
        }

        tile_down += CUBE_COUNT_Y;
    }
}

/// The cubes covering a `width` by `height` plane, one tile each, capped at what every device allows.
pub(crate) fn fixed_cubes(width: u32, height: u32) -> CubeCount {
    capped_tile_cubes(width, height, TILE_WIDTH, TILE_HEIGHT)
}

fn assert_filter_planes(planes: &FilterPlanes<'_>, storage: Storage) {
    let width = planes.output.width();
    let height = planes.output.height();
    let bytes = storage.bytes();

    assert_plane_sized(&planes.input, width, height, bytes, "filter input");
    assert_plane_sized(&planes.output, width, height, bytes, "filter output");
}

/// Blurs `planes.input` into `planes.output` with a 3x3 kernel, reading past the edge as `edge` says.
///
/// Both planes hold samples of `storage`. The sums stay f32 either way.
pub fn smooth<R: Runtime>(
    client: &ComputeClient<R>,
    _dispatch: &impl Dispatch,
    planes: FilterPlanes<'_>,
    smooth: Smooth,
    edge: Edge,
    storage: Storage,
) {
    assert_filter_planes(&planes, storage);

    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = fixed_cubes(width, height);
    let shape = CubeDim::new_2d(UNITS_ACROSS, UNITS_DOWN);
    let binomial = smooth == Smooth::Binomial;
    let clamp = edge == Edge::Clamp;
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: both planes hold `width * height` values of `storage`, every read is folded into the
    // plane and every write is guarded to it.
    unsafe {
        match storage {
            Storage::F32 => smooth_kernel::launch_unchecked::<f32, R>(
                client, cubes, shape, input, output, width, height, binomial, clamp,
            ),
            Storage::F16 => smooth_kernel::launch_unchecked::<half::f16, R>(
                client, cubes, shape, input, output, width, height, binomial, clamp,
            ),
        }
    }
}

/// The vertical `1 2 1` blur over 4, reading past the top and bottom as `edge` says.
///
/// Both planes hold samples of `storage`. The sums stay f32 either way.
pub fn vertical_121<R: Runtime>(
    client: &ComputeClient<R>,
    _dispatch: &impl Dispatch,
    planes: FilterPlanes<'_>,
    edge: Edge,
    storage: Storage,
) {
    assert_filter_planes(&planes, storage);

    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = fixed_cubes(width, height);
    let shape = CubeDim::new_2d(UNITS_ACROSS, UNITS_DOWN);
    let clamp = edge == Edge::Clamp;
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: both planes hold `width * height` values of `storage`, every read is folded into the
    // plane and every write is guarded to it.
    unsafe {
        match storage {
            Storage::F32 => vertical_kernel::launch_unchecked::<f32, R>(
                client, cubes, shape, input, output, width, height, clamp,
            ),
            Storage::F16 => vertical_kernel::launch_unchecked::<half::f16, R>(
                client, cubes, shape, input, output, width, height, clamp,
            ),
        }
    }
}
