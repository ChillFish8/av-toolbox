use av_toolbox_cubek::DevicePlane;
use av_toolbox_gpu::FrameContext;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::sizing::{Scratch, Sizing};

/// Tiles across and down the grid.
const TILES: u32 = 8;

/// How many levels a sample is binned into.
const LEVELS: u32 = 256;

#[derive(Debug, Clone, Copy)]
pub(super) struct EqualisePlanes<'a> {
    pub(super) input: DevicePlane<'a>,
    pub(super) output: DevicePlane<'a>,
}

#[cube]
fn level_of(value: f32) -> u32 {
    let rounded = f32::round(value * 255.0);
    let clamped = f32::clamp(rounded, 0.0, 255.0);

    u32::cast_from(clamped)
}

#[cube(launch_unchecked)]
fn clear_histograms(histograms: &mut Array<u32>) {
    if ABSOLUTE_POS >= histograms.len() {
        terminate!();
    }

    histograms[ABSOLUTE_POS] = 0u32;
}

#[cube(launch_unchecked)]
fn count_levels(
    input: &Array<f32>,
    histograms: &mut Array<Atomic<u32>>,
    width: u32,
    tile_width: u32,
    tile_height: u32,
) {
    if ABSOLUTE_POS >= input.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let tile = (row / tile_height) * 8u32 + column / tile_width;
    let level = level_of(input[ABSOLUTE_POS]);

    histograms[(tile * 256u32 + level) as usize].fetch_add(1u32);
}

/// Clips one tile's histogram, spreads the excess evenly, and writes its running count as a table.
#[cube(launch_unchecked)]
fn build_tables(histograms: &Array<u32>, tables: &mut Array<f32>, limit: f32) {
    let tile = ABSOLUTE_POS as u32;
    if tile >= 64u32 {
        terminate!();
    }

    let base = tile * 256u32;
    let mut total = 0.0f32;
    let mut excess = 0.0f32;
    for level in 0..256u32 {
        let count = f32::cast_from(histograms[(base + level) as usize]);
        total += count;
        excess += f32::max(count - limit, 0.0);
    }

    let share = excess / 256.0;
    let mut running = 0.0f32;
    for level in 0..256u32 {
        let count = f32::cast_from(histograms[(base + level) as usize]);
        running += f32::min(count, limit) + share;

        if total > 0.0 {
            tables[(base + level) as usize] = running / total;
        } else {
            tables[(base + level) as usize] = 0.0;
        }
    }
}

#[cube]
fn table_at(tables: &Array<f32>, tile_x: u32, tile_y: u32, level: u32) -> f32 {
    let tile = tile_y * 8u32 + tile_x;

    tables[(tile * 256u32 + level) as usize]
}

/// The tile whose centre sits at or before `position`, counted in tiles, kept inside the grid.
#[cube]
fn tile_before(position: f32, tiles: u32) -> u32 {
    let floored = f32::floor(position);
    let clamped = f32::max(floored, 0.0);
    let tile = u32::cast_from(clamped);

    u32::min(tile, tiles - 1)
}

/// Maps each sample through the tables of the four nearest tile centres, blended bilinearly.
#[cube(launch_unchecked)]
fn equalise_samples(
    input: &Array<f32>,
    tables: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    tile_width: u32,
    tile_height: u32,
    tiles_across: u32,
    tiles_down: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let row = position / width;
    let level = level_of(input[ABSOLUTE_POS]);

    let across = (f32::cast_from(column) + 0.5) / f32::cast_from(tile_width) - 0.5;
    let down = (f32::cast_from(row) + 0.5) / f32::cast_from(tile_height) - 0.5;
    let left = tile_before(across, tiles_across);
    let top = tile_before(down, tiles_down);
    let right = u32::min(left + 1, tiles_across - 1);
    let bottom = u32::min(top + 1, tiles_down - 1);
    let left_tile = f32::cast_from(left);
    let top_tile = f32::cast_from(top);
    let blend_x = f32::clamp(across - left_tile, 0.0, 1.0);
    let blend_y = f32::clamp(down - top_tile, 0.0, 1.0);

    let top_left = table_at(tables, left, top, level);
    let top_right = table_at(tables, right, top, level);
    let bottom_left = table_at(tables, left, bottom, level);
    let bottom_right = table_at(tables, right, bottom, level);
    let upper = (1.0 - blend_x) * top_left + blend_x * top_right;
    let lower = (1.0 - blend_x) * bottom_left + blend_x * bottom_right;

    output[ABSOLUTE_POS] = (1.0 - blend_y) * upper + blend_y * lower;
}

/// Equalises contrast tile by tile.
///
/// Each tile's histogram is clipped at `clip` times its average bin, rounded down and at least 1.
pub(super) fn equalise<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    sizing: &Sizing<'_>,
    planes: EqualisePlanes<'_>,
    clip: f32,
) {
    let width = planes.input.width();
    let height = planes.input.height();
    let samples = planes.input.samples();
    let tile_width = width.div_ceil(TILES);
    let tile_height = height.div_ceil(TILES);
    let tiles_across = width.div_ceil(tile_width);
    let tiles_down = height.div_ceil(tile_height);
    let tile_samples = (tile_width * tile_height) as f32;
    let limit = (clip * tile_samples / LEVELS as f32).floor().max(1.0);

    let entries = (TILES * TILES * LEVELS) as usize;
    let histograms = Scratch::line::<u32>(context, entries);
    let tables = Scratch::line::<f32>(context, entries);

    let cleared = histograms.plane().array::<R>();
    let cubes = sizing.0.cube_count(entries);
    let shape = sizing.0.cube_dim();

    // Safety: the histograms hold `entries` u32 values and the kernel writes nothing past them.
    unsafe {
        clear_histograms::launch_unchecked::<R>(client, cubes, shape, cleared);
    }

    let input = planes.input.array::<R>();
    let counted = histograms.plane().array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: the input holds `samples` f32 values, and every tile and level index stays below
    // `entries` because a tile is at most 7 across and 7 down.
    unsafe {
        count_levels::launch_unchecked::<R>(
            client,
            cubes,
            shape,
            input,
            counted,
            width,
            tile_width,
            tile_height,
        );
    }

    let counts = histograms.plane().array::<R>();
    let built = tables.plane().array::<R>();
    let tile_count = (TILES * TILES) as usize;
    let cubes = sizing.0.cube_count(tile_count);
    let shape = sizing.0.cube_dim();

    // Safety: both buffers hold `entries` values and each of the 64 units touches only its own tile.
    unsafe {
        build_tables::launch_unchecked::<R>(client, cubes, shape, counts, built, limit);
    }

    let input = planes.input.array::<R>();
    let looked_up = tables.plane().array::<R>();
    let output = planes.output.array::<R>();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: the input and output hold `samples` f32 values, and every table index stays below
    // `entries`.
    unsafe {
        equalise_samples::launch_unchecked::<R>(
            client,
            cubes,
            shape,
            input,
            looked_up,
            output,
            width,
            tile_width,
            tile_height,
            tiles_across,
            tiles_down,
        );
    }
}
