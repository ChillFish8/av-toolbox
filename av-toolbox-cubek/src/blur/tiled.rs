// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{FilterPlanes, MAX_CUBES_PER_AXIS, MAX_SHARED_BYTES, Storage, TILED_MAX_RADIUS};
use crate::Edge;
use crate::border::fold;

/// The units along a cube's rows.
const UNITS_ACROSS: u32 = 16;
/// The units down a cube's columns.
const UNITS_DOWN: u32 = 16;
/// The outputs each unit writes along a row.
const OUTPUTS_ACROSS: u32 = 2;
/// The outputs each unit writes down a column.
const OUTPUTS_DOWN: u32 = 2;
/// The width of the output tile a cube writes.
const TILE_WIDTH: u32 = UNITS_ACROSS * OUTPUTS_ACROSS;
/// The height of the output tile a cube writes.
const TILE_HEIGHT: u32 = UNITS_DOWN * OUTPUTS_DOWN;

/// The width of the bordered tile a cube loads at `radius`.
///
/// It is rounded up to a multiple of the cube's width, so every unit loads the same count of samples.
const fn loaded_width(radius: u32) -> u32 {
    let bordered = TILE_WIDTH + 2 * radius;

    bordered.div_ceil(UNITS_ACROSS) * UNITS_ACROSS
}

/// The height of the bordered tile a cube loads at `radius`.
///
/// It is rounded up to a multiple of the cube's height, so every unit loads the same count of samples.
const fn loaded_height(radius: u32) -> u32 {
    let bordered = TILE_HEIGHT + 2 * radius;

    bordered.div_ceil(UNITS_DOWN) * UNITS_DOWN
}

/// The shared memory one cube takes at `radius`, the bordered tile plus its blur across.
const fn shared_bytes(radius: u32) -> u32 {
    let loaded_width = loaded_width(radius);
    let loaded_height = loaded_height(radius);
    let samples = loaded_width * loaded_height + TILE_WIDTH * loaded_height;

    samples * size_of::<f32>() as u32
}

const _: () = assert!(shared_bytes(TILED_MAX_RADIUS) <= MAX_SHARED_BYTES);

/// The cubes giving each output tile of a `width` by `height` plane a cube of its own.
///
/// It is `None` when a plane needs more cubes along an axis than every device allows.
pub(super) fn tile_cubes(width: u32, height: u32) -> Option<CubeCount> {
    let tiles_across = width.div_ceil(TILE_WIDTH);
    let tiles_down = height.div_ceil(TILE_HEIGHT);
    if tiles_across > MAX_CUBES_PER_AXIS || tiles_down > MAX_CUBES_PER_AXIS {
        return None;
    }

    Some(CubeCount::Static(tiles_across, tiles_down, 1))
}

/// Copies the tile and a border of `radius` samples on every side into `loaded`.
///
/// Every source coordinate is folded into the plane, clamped when `clamp` is set and reflected
/// otherwise, so the blurs read the tile without edge checks.
#[cube]
fn load_tile<F: Float>(
    input: &Array<F>,
    loaded: &mut SharedMemory<f32>,
    width: u32,
    height: u32,
    #[comptime] radius: u32,
    #[comptime] clamp: bool,
) {
    let loaded_width = comptime!(loaded_width(radius));
    let loaded_height = comptime!(loaded_height(radius));
    let reach = comptime!(radius as i32);
    let first_column = (CUBE_POS_X * TILE_WIDTH) as i32 - reach;
    let first_row = (CUBE_POS_Y * TILE_HEIGHT) as i32 - reach;

    #[unroll]
    for row_step in 0..comptime!(loaded_height / UNITS_DOWN) {
        let local_row = UNIT_POS_Y + row_step * UNITS_DOWN;
        let source_row = fold(first_row + local_row as i32, height, clamp);
        let row_start = source_row * width;

        #[unroll]
        for column_step in 0..comptime!(loaded_width / UNITS_ACROSS) {
            let local_column = UNIT_POS_X + column_step * UNITS_ACROSS;
            let source_column = fold(first_column + local_column as i32, width, clamp);
            let sample = input[(row_start + source_column) as usize];
            let slot = local_row * loaded_width + local_column;
            loaded[slot as usize] = f32::cast_from(sample);
        }
    }
}

/// Blurs every row of `loaded` across into `across`, which keeps the vertical border but not the
/// horizontal one.
#[cube]
fn blur_across(
    loaded: &SharedMemory<f32>,
    taps: &Array<f32>,
    across: &mut SharedMemory<f32>,
    #[comptime] radius: u32,
) {
    let loaded_width = comptime!(loaded_width(radius));
    let loaded_height = comptime!(loaded_height(radius));

    #[unroll]
    for row_step in 0..comptime!(loaded_height / UNITS_DOWN) {
        let local_row = UNIT_POS_Y + row_step * UNITS_DOWN;

        #[unroll]
        for column_step in 0..OUTPUTS_ACROSS {
            let local_column = UNIT_POS_X + column_step * UNITS_ACROSS;
            let window_start = local_row * loaded_width + local_column;

            let mut total = 0.0f32;
            #[unroll]
            for tap in 0..comptime!(2 * radius + 1) {
                total += loaded[(window_start + tap) as usize] * taps[tap as usize];
            }

            across[(local_row * TILE_WIDTH + local_column) as usize] = total;
        }
    }
}

/// Blurs down the columns of `across` and writes the outputs that fall inside the plane.
#[cube]
fn blur_down<F: Float>(
    across: &SharedMemory<f32>,
    taps: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    #[comptime] radius: u32,
) {
    let first_column = CUBE_POS_X * TILE_WIDTH;
    let first_row = CUBE_POS_Y * TILE_HEIGHT;

    #[unroll]
    for row_step in 0..OUTPUTS_DOWN {
        let local_row = UNIT_POS_Y + row_step * UNITS_DOWN;
        let row = first_row + local_row;

        #[unroll]
        for column_step in 0..OUTPUTS_ACROSS {
            let local_column = UNIT_POS_X + column_step * UNITS_ACROSS;
            let column = first_column + local_column;

            let mut total = 0.0f32;
            #[unroll]
            for tap in 0..comptime!(2 * radius + 1) {
                let slot = (local_row + tap) * TILE_WIDTH + local_column;
                total += across[slot as usize] * taps[tap as usize];
            }

            if row < height && column < width {
                output[(row * width + column) as usize] = F::cast_from(total);
            }
        }
    }
}

/// Blurs one tile of the plane in a single cube, staging it and its border through shared memory.
#[cube(launch_unchecked)]
fn blur_tile<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    #[comptime] radius: u32,
    #[comptime] clamp: bool,
) {
    let tap_count = comptime!(2 * radius + 1);
    let loaded_width = comptime!(loaded_width(radius));
    let loaded_height = comptime!(loaded_height(radius));
    let mut loaded = SharedMemory::<f32>::new(comptime!((loaded_width * loaded_height) as usize));
    let mut across = SharedMemory::<f32>::new(comptime!((TILE_WIDTH * loaded_height) as usize));

    let mut taps = Array::<f32>::new(comptime!(tap_count as usize));
    #[unroll]
    for tap in 0..tap_count {
        taps[tap as usize] = weights[tap as usize];
    }

    load_tile(input, &mut loaded, width, height, radius, clamp);

    sync_cube();

    blur_across(&loaded, &taps, &mut across, radius);

    sync_cube();

    blur_down(&across, &taps, output, width, height, radius);
}

/// Blurs `planes.input` into `planes.output` in one dispatch, one output tile a cube.
///
/// It reads past the edge as `edge` says. It panics when the plane needs more cubes than
/// [tile_cubes](crate::blur::tiled::tile_cubes) allows.
pub(super) fn blur_tiled<R: Runtime>(
    client: &ComputeClient<R>,
    weights: ArrayArg<R>,
    planes: FilterPlanes<'_>,
    radius: u32,
    edge: Edge,
    storage: Storage,
) {
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = tile_cubes(width, height).expect("the plane should fit the tiled blur's cube count");
    let shape = CubeDim::new_2d(UNITS_ACROSS, UNITS_DOWN);
    let clamp = edge == Edge::Clamp;
    let input = planes.input.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: both planes hold `width * height` values of `storage`, every read is folded into the
    // plane and every write is guarded to it.
    unsafe {
        match storage {
            Storage::F32 => blur_tile::launch_unchecked::<f32, R>(
                client, cubes, shape, input, weights, output, width, height, radius, clamp,
            ),
            Storage::F16 => blur_tile::launch_unchecked::<half::f16, R>(
                client, cubes, shape, input, weights, output, width, height, radius, clamp,
            ),
        }
    }
}
