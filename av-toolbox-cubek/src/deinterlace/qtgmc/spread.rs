// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::fmtc_gauss_taps;
use crate::blur::capped_tile_cubes;
use crate::border::fold;
use crate::plane::are_distinct;
use crate::resize::{Taps, Weights, upload};
use crate::{DevicePlane, Dispatch, assert_plane_sized};

/// How many samples each side of a sample the gaussian reaches.
const REACH: u32 = 4;
/// How many samples each output of the gaussian reads along an axis.
const TAP_COUNT: u32 = 2 * REACH + 1;
/// The border the input is loaded with, the gaussian's reach plus the binomial's.
const LOADED_BORDER: u32 = REACH + 1;

/// The units along a cube's rows, one for each column of the output tile.
const UNITS_ACROSS: u32 = 32;
/// The units down a cube's columns.
const UNITS_DOWN: u32 = 8;
/// Every unit of a cube.
const UNITS: u32 = UNITS_ACROSS * UNITS_DOWN;
/// The outputs each unit writes down its column.
const OUTPUTS_DOWN: u32 = 4;
/// The width of the output tile a cube writes.
const TILE_WIDTH: u32 = UNITS_ACROSS;
/// The height of the output tile a cube writes.
const TILE_HEIGHT: u32 = UNITS_DOWN * OUTPUTS_DOWN;

/// The width of the input a cube loads, the tile and its border.
const LOADED_WIDTH: u32 = TILE_WIDTH + 2 * LOADED_BORDER;
/// The height of the input a cube loads, the tile and its border.
const LOADED_HEIGHT: u32 = TILE_HEIGHT + 2 * LOADED_BORDER;
/// The width of the binomial blur a cube keeps, the tile and the gaussian's reach.
const BLURRED_WIDTH: u32 = TILE_WIDTH + 2 * REACH;
/// The height of the binomial blur a cube keeps, the tile and the gaussian's reach.
const BLURRED_HEIGHT: u32 = TILE_HEIGHT + 2 * REACH;

/// The gaussian tables of one plane size and sharpness, on the device.
///
/// Output `i` of a line reads the 9 samples from `i - 4`, each with its own weights.
#[derive(Debug)]
pub struct GaussTables {
    across: Weights,
    down: Weights,
}

impl GaussTables {
    /// Builds the tables of a `size` plane, as `(width, height)`, for fmtc's gaussian of `a1 = sharpness`.
    pub fn new<R: Runtime>(client: &ComputeClient<R>, size: (u32, u32), sharpness: f32) -> Self {
        let across = fmtc_gauss_taps(size.0, sharpness);
        let down = fmtc_gauss_taps(size.1, sharpness);
        assert_centred(&across);
        assert_centred(&down);

        Self {
            across: upload(client, &across),
            down: upload(client, &down),
        }
    }
}

/// Asserts that output `i` of `taps` reads [TAP_COUNT] samples from `i - REACH`.
fn assert_centred(taps: &Taps) {
    assert_eq!(
        taps.count, TAP_COUNT,
        "the gaussian reads {} taps, expected {TAP_COUNT}",
        taps.count
    );

    for (index, start) in taps.starts.iter().enumerate() {
        let expected = index as i32 - REACH as i32;
        assert_eq!(
            *start, expected,
            "gaussian output {index} starts at {start}, expected {expected}"
        );
    }
}

/// Copies the tile and its border from `input` into `loaded`, clamping every coordinate into the plane.
#[cube]
fn load_tile(
    input: &Array<f32>,
    loaded: &mut SharedMemory<f32>,
    width: u32,
    height: u32,
    first_column: u32,
    first_row: u32,
) {
    let samples = comptime!(LOADED_WIDTH * LOADED_HEIGHT);
    let border = comptime!(LOADED_BORDER as i32);
    let top = first_row as i32 - border;
    let left = first_column as i32 - border;

    #[unroll]
    for step in 0..comptime!(samples.div_ceil(UNITS)) {
        let slot = UNIT_POS + step * UNITS;

        if slot < samples {
            let row = fold(top + (slot / LOADED_WIDTH) as i32, height, true);
            let column = fold(left + (slot % LOADED_WIDTH) as i32, width, true);
            loaded[slot as usize] = input[(row * width + column) as usize];
        }
    }
}

/// The `1 2 1` sum of the three loaded samples around `column` in `row`.
#[cube]
fn row_sum(loaded: &SharedMemory<f32>, row: u32, column: u32) -> f32 {
    let centre = row * LOADED_WIDTH + column;
    let left = loaded[(centre - 1) as usize];
    let middle = loaded[centre as usize];
    let right = loaded[(centre + 1) as usize];

    left + 2.0 * middle + right
}

/// Blurs `loaded` with the 3x3 binomial into `blurred`, which keeps the gaussian's reach around the tile.
///
/// A sample past the plane's edge takes the blur of the edge sample it clamps to.
#[cube]
fn smooth_tile(
    loaded: &SharedMemory<f32>,
    blurred: &mut SharedMemory<f32>,
    width: u32,
    height: u32,
    first_column: u32,
    first_row: u32,
) {
    let samples = comptime!(BLURRED_WIDTH * BLURRED_HEIGHT);
    let reach = comptime!(REACH as i32);
    let top = first_row as i32 - reach;
    let left = first_column as i32 - reach;

    #[unroll]
    for step in 0..comptime!(samples.div_ceil(UNITS)) {
        let slot = UNIT_POS + step * UNITS;

        if slot < samples {
            let row = fold(top + (slot / BLURRED_WIDTH) as i32, height, true);
            let column = fold(left + (slot % BLURRED_WIDTH) as i32, width, true);
            let loaded_row = row + LOADED_BORDER - first_row;
            let loaded_column = column + LOADED_BORDER - first_column;

            let above = row_sum(loaded, loaded_row - 1, loaded_column);
            let centre = row_sum(loaded, loaded_row, loaded_column);
            let below = row_sum(loaded, loaded_row + 1, loaded_column);
            blurred[slot as usize] = (above + 2.0 * centre + below) / 16.0;
        }
    }
}

/// Runs the gaussian along every row of `blurred` into `across`, which keeps the vertical reach only.
#[cube]
fn spread_across(
    blurred: &SharedMemory<f32>,
    weights: &Array<f32>,
    across: &mut SharedMemory<f32>,
    width: u32,
    first_column: u32,
) {
    let column = u32::min(first_column + UNIT_POS_X, width - 1);
    let mut taps = Array::<f32>::new(comptime!(TAP_COUNT as usize));
    #[unroll]
    for tap in 0..TAP_COUNT {
        taps[tap as usize] = weights[(column * TAP_COUNT + tap) as usize];
    }

    #[unroll]
    for step in 0..comptime!(BLURRED_HEIGHT.div_ceil(UNITS_DOWN)) {
        let row = UNIT_POS_Y + step * UNITS_DOWN;

        if row < BLURRED_HEIGHT {
            let window_start = row * BLURRED_WIDTH + UNIT_POS_X;
            let mut total = 0.0f32;
            #[unroll]
            for tap in 0..TAP_COUNT {
                total += blurred[(window_start + tap) as usize] * taps[tap as usize];
            }

            across[(row * TILE_WIDTH + UNIT_POS_X) as usize] = total;
        }
    }
}

/// Runs the gaussian down the columns of `across` and writes the outputs that fall inside the plane.
#[cube]
fn spread_down(
    across: &SharedMemory<f32>,
    weights: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    first_column: u32,
    first_row: u32,
) {
    let column = first_column + UNIT_POS_X;

    #[unroll]
    for step in 0..OUTPUTS_DOWN {
        let local_row = UNIT_POS_Y + step * UNITS_DOWN;
        let row = first_row + local_row;
        let weight_start = u32::min(row, height - 1) * TAP_COUNT;

        let mut total = 0.0f32;
        #[unroll]
        for tap in 0..TAP_COUNT {
            let slot = (local_row + tap) * TILE_WIDTH + UNIT_POS_X;
            total += across[slot as usize] * weights[(weight_start + tap) as usize];
        }

        if row < height && column < width {
            output[(row * width + column) as usize] = total;
        }
    }
}

/// Blurs with the 3x3 binomial and then the gaussian, each cube walking the tiles its grid position lands on.
///
/// A cube usually has one tile. It walks more only when the plane needs more cubes than a device allows
/// along an axis.
#[cube(launch_unchecked)]
fn spread_kernel(
    input: &Array<f32>,
    across_weights: &Array<f32>,
    down_weights: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
) {
    let mut loaded = SharedMemory::<f32>::new(comptime!((LOADED_WIDTH * LOADED_HEIGHT) as usize));
    let mut blurred = SharedMemory::<f32>::new(comptime!((BLURRED_WIDTH * BLURRED_HEIGHT) as usize));
    let mut across = SharedMemory::<f32>::new(comptime!((TILE_WIDTH * BLURRED_HEIGHT) as usize));

    let mut tile_down = CUBE_POS_Y;
    while tile_down * TILE_HEIGHT < height {
        let first_row = tile_down * TILE_HEIGHT;
        let mut tile_across = CUBE_POS_X;

        while tile_across * TILE_WIDTH < width {
            let first_column = tile_across * TILE_WIDTH;

            load_tile(input, &mut loaded, width, height, first_column, first_row);

            sync_cube();

            smooth_tile(&loaded, &mut blurred, width, height, first_column, first_row);

            sync_cube();

            spread_across(&blurred, across_weights, &mut across, width, first_column);

            sync_cube();

            spread_down(
                &across,
                down_weights,
                output,
                width,
                height,
                first_column,
                first_row,
            );

            tile_across += CUBE_COUNT_X;
        }

        tile_down += CUBE_COUNT_Y;
    }
}

/// The cubes covering a `width` by `height` plane, one tile each, capped at what every device allows.
fn spread_cubes(width: u32, height: u32) -> CubeCount {
    capped_tile_cubes(width, height, TILE_WIDTH, TILE_HEIGHT)
}

/// Blurs `input` into `output` with the 3x3 binomial and then fmtc's gaussian from `gauss`, in one launch.
///
/// Both blurs clamp at the border, and the gaussian keeps each output's own weights. The planes must be
/// separate.
pub fn blur_and_spread<R: Runtime>(
    client: &ComputeClient<R>,
    _dispatch: &impl Dispatch,
    gauss: &GaussTables,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
) {
    let width = output.width();
    let height = output.height();
    let bytes = size_of::<f32>();
    assert_plane_sized(&input, width, height, bytes, "spread input");
    assert_plane_sized(&output, width, height, bytes, "spread output");
    assert!(
        are_distinct(&[input.handle(), output.handle()]),
        "the spread input and output must be separate planes"
    );
    assert_eq!(
        gauss.across.outputs(),
        width,
        "the across table does not match the plane width"
    );
    assert_eq!(
        gauss.down.outputs(),
        height,
        "the down table does not match the plane height"
    );

    let cubes = spread_cubes(width, height);
    let unit = CubeDim::new_2d(UNITS_ACROSS, UNITS_DOWN);
    let input = input.array::<R>();
    let across = gauss.across.weights::<R>();
    let down = gauss.down.weights::<R>();
    let output = output.array::<R>();

    // Safety: both planes hold `width * height` f32 values and the tables 9 weights for each output of
    // their axis. Every read is clamped into the plane or its table, and every write is guarded to the plane.
    unsafe {
        spread_kernel::launch_unchecked::<R>(client, cubes, unit, input, across, down, output, width, height);
    }
}
