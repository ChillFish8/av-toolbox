// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BlurKind, average};
use crate::blur::{MAX_SHARED_BYTES, capped_tile_cubes};
use crate::plane::are_distinct;
use crate::{DevicePlane, Dispatch, assert_plane_sized};

/// The width of the output tile a cube writes.
const TILE_WIDTH: u32 = 64;
/// The units of a cube.
const UNITS: u32 = 256;
/// The u32 values a cube's two staged buffers hold together.
const SHARED_VALUES: u32 = MAX_SHARED_BYTES / size_of::<u32>() as u32;
/// The outputs each unit writes along a row.
const RUN_ACROSS: u32 = 16;

const fn reach_of(wide: bool) -> u32 {
    if wide { 6 } else { 2 }
}

/// How far the staged square reaches past the tile, a reach for every pass.
const fn border_of(wide: bool, passes: u32) -> u32 {
    reach_of(wide) * passes
}

/// The slots between two staged rows, odd so units reading down a column hit distinct banks.
const fn stride_of(wide: bool, passes: u32) -> u32 {
    let staged_width = TILE_WIDTH + 2 * border_of(wide, passes);

    staged_width | 1
}

/// The height of the output tile a cube writes, as tall as the two staged buffers allow.
const fn tile_height_of(wide: bool, passes: u32) -> u32 {
    let staged_rows = SHARED_VALUES / 2 / stride_of(wide, passes);
    let tile_rows = staged_rows - 2 * border_of(wide, passes);

    tile_rows / 4 * 4
}

/// The outputs each unit writes down a column, so the last pass gives every unit one run.
const fn run_down_of(wide: bool, passes: u32) -> u32 {
    tile_height_of(wide, passes) * TILE_WIDTH / UNITS
}

const fn staged_height_of(wide: bool, passes: u32) -> u32 {
    tile_height_of(wide, passes) + 2 * border_of(wide, passes)
}

/// The most passes one launch runs.
///
/// A launch of several r6 passes is slower than one launch for each, as its border grows by 6 every pass.
const fn passes_per_launch(wide: bool) -> u32 {
    if wide { 1 } else { 2 }
}

/// Whether every launch of `wide` keeps a tile and fits its staged buffers in [SHARED_VALUES].
const fn fits(wide: bool) -> bool {
    let mut passes = 1;
    let mut fitting = true;

    while passes <= passes_per_launch(wide) {
        let staged = stride_of(wide, passes) * staged_height_of(wide, passes);
        fitting = fitting && tile_height_of(wide, passes) > 0 && 2 * staged <= SHARED_VALUES;
        passes += 1;
    }

    fitting
}

const _: () = assert!(fits(true) && fits(false));

/// Where a cube's tile and its staged square sit in the plane.
#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
struct Window {
    columns: i32,
    rows: i32,
    /// The plane column of the staged square's first column, negative past the plane's start.
    left: i32,
    /// The plane row of the staged square's first row, negative past the plane's start.
    top: i32,
    first_column: i32,
    first_row: i32,
}

/// The plane positions from `first - spread` up to `first + tile + spread`, kept inside a line of `length`.
#[cube]
fn span(first: i32, tile: i32, spread: i32, length: i32) -> (i32, i32) {
    let start = i32::max(first - spread, 0);
    let end = i32::min(first + tile + spread, length);

    (start, end)
}

/// One blur step of the sample at `centre` in `taps`, which holds the line around it.
///
/// `position` is the sample's place in its line of `length`. For r6, within 6 samples of an end only
/// the samples on the inner side are read, and where both ends are that close `start_wins` picks the start.
#[cube]
fn blur_taps(
    taps: &[i32],
    position: i32,
    length: i32,
    start_wins: bool,
    #[comptime] centre: u32,
    #[comptime] wide: bool,
) -> u32 {
    let middle_tap = taps[centre as usize];

    if wide {
        let near_start = position < 6;
        let near_end = position >= length - 6;

        let mut direction = 0i32;
        if near_start && (start_wins || !near_end) {
            direction = 1;
        } else if near_end {
            direction = -1;
        }

        let before_one = taps[(centre - 1) as usize];
        let after_one = taps[(centre + 1) as usize];
        let before_two = taps[(centre - 2) as usize];
        let after_two = taps[(centre + 2) as usize];
        let before_three = taps[(centre - 3) as usize];
        let after_three = taps[(centre + 3) as usize];
        let before_four = taps[(centre - 4) as usize];
        let after_four = taps[(centre + 4) as usize];
        let before_five = taps[(centre - 5) as usize];
        let after_five = taps[(centre + 5) as usize];
        let before_six = taps[(centre - 6) as usize];
        let after_six = taps[(centre + 6) as usize];

        let (near, middle, far) = if direction == 0 {
            let pair_one = average(before_one, after_one);
            let pair_two = average(before_two, after_two);
            let pair_three = average(before_three, after_three);
            let pair_four = average(before_four, after_four);
            let pair_five = average(before_five, after_five);
            let pair_six = average(before_six, after_six);

            let near = average(pair_one, pair_two);
            let middle = average(pair_three, pair_four);
            let far = average(pair_five, pair_six);
            (near, middle, far)
        } else {
            let forward = direction > 0;
            let tap_one = select(forward, after_one, before_one);
            let tap_two = select(forward, after_two, before_two);
            let tap_three = select(forward, after_three, before_three);
            let tap_four = select(forward, after_four, before_four);
            let tap_five = select(forward, after_five, before_five);
            let tap_six = select(forward, after_six, before_six);

            let near = average(tap_one, tap_two);
            let middle = average(tap_three, tap_four);
            let far = average(tap_five, tap_six);
            (near, middle, far)
        };

        let with_centre = average(middle_tap, near);
        let outer = average(middle, far);
        let everything = average(with_centre, outer);
        let blurred = average(with_centre, everything);

        u32::cast_from(blurred)
    } else {
        let before_one = taps[(centre - 1) as usize];
        let after_one = taps[(centre + 1) as usize];
        let before_two = taps[(centre - 2) as usize];
        let after_two = taps[(centre + 2) as usize];

        let inner = average(before_one, after_one);
        let outer = average(before_two, after_two);
        let first = average(outer, middle_tap);
        let second = average(first, middle_tap);
        let blurred = average(second, inner);

        u32::cast_from(blurred)
    }
}

/// Copies the staged square from `input`, clamping each read into the plane.
#[cube]
fn load_staged(
    input: &[u32],
    staged: &mut Shared<[u32]>,
    window: Window,
    #[comptime] wide: bool,
    #[comptime] passes: u32,
) {
    let stride = comptime!(stride_of(wide, passes));
    let staged_width = comptime!(TILE_WIDTH + 2 * border_of(wide, passes));
    let samples = comptime!(staged_width * staged_height_of(wide, passes));

    let last_row = window.rows - 1;
    let last_column = window.columns - 1;

    #[unroll]
    for step in 0..comptime!(samples.div_ceil(UNITS)) {
        let item = UNIT_POS + step * UNITS;
        let local_row = item / staged_width;
        let local_column = item % staged_width;
        let row = i32::clamp(window.top + local_row as i32, 0, last_row);
        let column = i32::clamp(window.left + local_column as i32, 0, last_column);
        let value = input[(row * window.columns + column) as usize];

        if item < samples {
            let slot = local_row * stride + local_column;
            staged[slot as usize] = value;
        }
    }
}

/// Blurs the staged rows within `row_spread` of the tile across, at the columns within `column_spread`.
///
/// Each unit slides along a run of `run` outputs, reading each staged value once. Reads are clamped into
/// the columns the step before wrote.
#[cube]
fn blur_staged_rows(
    staged: &Shared<[u32]>,
    across: &mut Shared<[u32]>,
    window: Window,
    row_spread: i32,
    column_spread: i32,
    #[comptime] wide: bool,
    #[comptime] passes: u32,
    #[comptime] run: u32,
) {
    let reach = comptime!(reach_of(wide));
    let stride = comptime!(stride_of(wide, passes) as i32);
    let tile_width = comptime!(TILE_WIDTH as i32);
    let tile_height = comptime!(tile_height_of(wide, passes) as i32);
    let run_length = comptime!(run as i32);
    let rows_most = tile_height + 2 * row_spread;
    let runs = (tile_width + 2 * column_spread + run_length - 1) / run_length;
    let items = rows_most * runs;
    let (row_start, row_end) = span(window.first_row, tile_height, row_spread, window.rows);
    let (column_start, column_end) = span(window.first_column, tile_width, column_spread, window.columns);
    let reach_spread = column_spread + reach as i32;
    let (read_start, read_end) = span(window.first_column, tile_width, reach_spread, window.columns);
    let read_last = read_end - 1;

    let mut item = UNIT_POS as i32;
    while item < items {
        let row = window.first_row - row_spread + item % rows_most;
        let first = window.first_column - column_spread + (item / rows_most) * run_length;

        if row >= row_start && row < row_end {
            let origin = (row - window.top) * stride - window.left;
            let mut taps = Array::<i32>::new(comptime!((run + 2 * reach) as usize));

            #[unroll]
            for tap in 0..comptime!(run + 2 * reach) {
                let position = first - reach as i32 + tap as i32;
                let clamped = i32::clamp(position, read_start, read_last);
                let value = staged[(origin + clamped) as usize];
                taps[tap as usize] = i32::cast_from(value);
            }

            #[unroll]
            for step in 0..run {
                let column = first + step as i32;

                if column >= column_start && column < column_end {
                    let blurred = blur_taps(
                        taps.as_slice(),
                        column,
                        window.columns,
                        false,
                        comptime!(step + reach),
                        wide,
                    );
                    across[(origin + column) as usize] = blurred;
                }
            }
        }

        item += UNITS as i32;
    }
}

/// Blurs the staged columns within `spread` of the tile down, at the rows within `spread`.
///
/// Each unit slides down a run of `run` outputs. The last pass writes its tile to `output`, and the others
/// write back into `staged`.
#[cube]
fn blur_staged_columns(
    across: &Shared<[u32]>,
    staged: &mut Shared<[u32]>,
    output: &mut [u32],
    window: Window,
    spread: i32,
    #[comptime] wide: bool,
    #[comptime] passes: u32,
    #[comptime] run: u32,
    #[comptime] last: bool,
) {
    let reach = comptime!(reach_of(wide));
    let stride = comptime!(stride_of(wide, passes) as i32);
    let tile_width = comptime!(TILE_WIDTH as i32);
    let tile_height = comptime!(tile_height_of(wide, passes) as i32);
    let run_length = comptime!(run as i32);
    let columns_most = tile_width + 2 * spread;
    let runs = (tile_height + 2 * spread + run_length - 1) / run_length;
    let items = columns_most * runs;
    let (row_start, row_end) = span(window.first_row, tile_height, spread, window.rows);
    let (column_start, column_end) = span(window.first_column, tile_width, spread, window.columns);
    let reach_spread = spread + reach as i32;
    let (read_start, read_end) = span(window.first_row, tile_height, reach_spread, window.rows);
    let read_last = read_end - 1;

    let mut item = UNIT_POS as i32;
    while item < items {
        let column = window.first_column - spread + item % columns_most;
        let first = window.first_row - spread + (item / columns_most) * run_length;

        if column >= column_start && column < column_end {
            let origin = column - window.left - window.top * stride;
            let mut taps = Array::<i32>::new(comptime!((run + 2 * reach) as usize));

            #[unroll]
            for tap in 0..comptime!(run + 2 * reach) {
                let position = first - reach as i32 + tap as i32;
                let clamped = i32::clamp(position, read_start, read_last);
                let value = across[(origin + clamped * stride) as usize];
                taps[tap as usize] = i32::cast_from(value);
            }

            #[unroll]
            for step in 0..run {
                let row = first + step as i32;

                if row >= row_start && row < row_end {
                    let blurred = blur_taps(
                        taps.as_slice(),
                        row,
                        window.rows,
                        true,
                        comptime!(step + reach),
                        wide,
                    );

                    if last {
                        output[(row * window.columns + column) as usize] = blurred;
                    } else {
                        staged[(origin + row * stride) as usize] = blurred;
                    }
                }
            }
        }

        item += UNITS as i32;
    }
}

/// Runs `passes` blurs, each cube staging a tile and its border through shared memory and walking the
/// tiles its grid position lands on.
///
/// Each pass leaves the outermost reach of what it read stale, so the border is a reach deep for every
/// pass. A cube usually has one tile. It walks more only when the plane needs more cubes than a device
/// allows along an axis.
#[cube(launch_unchecked)]
fn blur_staged_kernel(
    input: &[u32],
    output: &mut [u32],
    width: u32,
    height: u32,
    #[comptime] wide: bool,
    #[comptime] passes: u32,
) {
    let run_down = comptime!(run_down_of(wide, passes));
    let reach = comptime!(reach_of(wide) as i32);
    let border = comptime!(border_of(wide, passes) as i32);
    let tile_height = comptime!(tile_height_of(wide, passes));
    let staged_samples = comptime!((stride_of(wide, passes) * staged_height_of(wide, passes)) as usize);
    let mut staged = Shared::<[u32]>::new_slice(staged_samples);
    let mut across = Shared::<[u32]>::new_slice(staged_samples);

    let mut tile_down = CUBE_POS_Y;
    while tile_down * tile_height < height {
        let first_row = (tile_down * tile_height) as i32;
        let mut tile_across = CUBE_POS_X;

        while tile_across * TILE_WIDTH < width {
            let first_column = (tile_across * TILE_WIDTH) as i32;
            let window = Window {
                columns: width as i32,
                rows: height as i32,
                left: first_column - border,
                top: first_row - border,
                first_column,
                first_row,
            };

            load_staged(input, &mut staged, window, wide, passes);

            sync_cube();

            #[unroll]
            for pass in 0..comptime!(passes - 1) {
                let row_spread = border - reach * pass as i32;
                let column_spread = row_spread - reach;

                blur_staged_rows(
                    &staged,
                    &mut across,
                    window,
                    row_spread,
                    column_spread,
                    wide,
                    passes,
                    RUN_ACROSS,
                );

                sync_cube();

                blur_staged_columns(
                    &across,
                    &mut staged,
                    output,
                    window,
                    column_spread,
                    wide,
                    passes,
                    run_down,
                    false,
                );

                sync_cube();
            }

            blur_staged_rows(&staged, &mut across, window, reach, 0, wide, passes, RUN_ACROSS);

            sync_cube();

            blur_staged_columns(
                &across,
                &mut staged,
                output,
                window,
                0,
                wide,
                passes,
                run_down,
                true,
            );

            tile_across += CUBE_COUNT_X;
        }

        tile_down += CUBE_COUNT_Y;
    }
}

/// Blurs `input` into `output` `passes` times in one launch.
fn launch_staged(
    client: &Client,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    kind: BlurKind,
    passes: u32,
) {
    let width = output.width();
    let height = output.height();
    let bytes = size_of::<u32>();
    assert_plane_sized(&input, width, height, bytes, "mask blur input");
    assert_plane_sized(&output, width, height, bytes, "mask blur output");
    assert!(
        are_distinct(&[input.handle(), output.handle()]),
        "the mask blur input and output must be separate planes"
    );
    assert!(
        output.samples() <= i32::MAX as usize,
        "the mask blur takes at most i32::MAX samples"
    );

    let wide = kind == BlurKind::R6;
    let tile_height = tile_height_of(wide, passes);
    let cubes = capped_tile_cubes(width, height, TILE_WIDTH, tile_height);
    let unit = CubeDim::new_1d(UNITS);
    let input = input.array();
    let output = output.array();

    // Safety: both planes hold `width * height` u32 values, every read is clamped into the plane and the
    // staged square, and every write is guarded to the plane.
    unsafe {
        blur_staged_kernel::launch_unchecked(client, cubes, unit, input, output, width, height, wide, passes);
    }
}

#[cube(launch_unchecked)]
fn copy_kernel(input: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS];
}

/// Blurs `mask` `passes` times with `kind`, passing between `mask` and `scratch`, and hands back the plane
/// holding the result.
///
/// Both planes hold u32 values and are one size. Every pass gives the same result as [blur_mask].
///
/// # Panics
///
/// Panics when the planes differ in size, share a handle, or hold more than `i32::MAX` samples.
pub fn blur_mask_passes<'a>(
    client: &Client,
    _dispatch: &impl Dispatch,
    mask: DevicePlane<'a>,
    scratch: DevicePlane<'a>,
    kind: BlurKind,
    passes: u32,
) -> DevicePlane<'a> {
    let most = passes_per_launch(kind == BlurKind::R6);
    let mut source = mask;
    let mut target = scratch;
    let mut remaining = passes;

    while remaining > 0 {
        let chunk = remaining.min(most);
        launch_staged(client, source, target, kind, chunk);

        remaining -= chunk;
        std::mem::swap(&mut source, &mut target);
    }

    source
}

/// Blurs `mask` once with `kind`, passing through `scratch`.
///
/// Both planes hold u32 values and are one size. The pass lands in `scratch` and is copied back into `mask`.
pub fn blur_mask(
    client: &Client,
    dispatch: &impl Dispatch,
    mask: DevicePlane<'_>,
    scratch: DevicePlane<'_>,
    kind: BlurKind,
) {
    let blurred = blur_mask_passes(client, dispatch, mask, scratch, kind, 1);

    let samples = mask.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let blurred = blurred.array();
    let mask = mask.array();

    // Safety: both planes hold `samples` u32 values, and the kernel stays below the output's length.
    unsafe {
        copy_kernel::launch_unchecked(client, cubes, unit, blurred, mask);
    }
}
