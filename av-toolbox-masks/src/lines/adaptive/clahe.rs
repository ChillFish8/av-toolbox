use av_toolbox_cubek::{DevicePlane, reflect};
use av_toolbox_gpu::FrameContext;
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::sizing::{Scratch, Sizing};

/// Tiles across and down the grid.
const TILES: u32 = 8;

/// How many levels a sample is binned into.
const LEVELS: u32 = 256;

/// [LEVELS] as a float, for spreading clipped counts.
const LEVEL_COUNT: f32 = LEVELS as f32;

/// The highest level a sample can land in, as a float.
const TOP_LEVEL: f32 = (LEVELS - 1) as f32;

#[derive(Debug, Clone, Copy)]
pub(super) struct EqualisePlanes<'a> {
    pub(super) input: DevicePlane<'a>,
    pub(super) output: DevicePlane<'a>,
}

#[cube]
fn level_of(value: f32) -> u32 {
    let rounded = f32::round(value * TOP_LEVEL);
    let clamped = f32::clamp(rounded, 0.0, TOP_LEVEL);

    u32::cast_from(clamped)
}

#[cube(launch_unchecked)]
fn clear_histograms(histograms: &mut [u32]) {
    if ABSOLUTE_POS >= histograms.len() {
        terminate!();
    }

    histograms[ABSOLUTE_POS] = 0u32;
}

/// Bins every sample of the image, extended with reflect-101 samples to a whole number of tiles,
/// into its tile's histogram.
#[cube(launch_unchecked)]
fn count_levels(
    input: &[f32],
    histograms: &mut [Atomic<u32>],
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
) {
    let extended_width = tile_width * TILES;
    let extended_samples = extended_width * tile_height * TILES;
    let position = ABSOLUTE_POS as u32;
    if position >= extended_samples {
        terminate!();
    }

    let column = position % extended_width;
    let row = position / extended_width;
    let source_column = reflect(column as i32, width as i32);
    let source_row = reflect(row as i32, height as i32);
    let source = source_row as u32 * width + source_column as u32;
    let tile = (row / tile_height) * TILES + column / tile_width;
    let level = level_of(input[source as usize]);

    histograms[(tile * LEVELS + level) as usize].fetch_add(1u32);
}

/// Clips one tile's histogram, spreads the excess evenly, and writes its running count as a table.
#[cube(launch_unchecked)]
fn build_tables(histograms: &[u32], tables: &mut [f32], limit: f32, tile_samples: f32) {
    let tile = ABSOLUTE_POS as u32;
    if tile >= TILES * TILES {
        terminate!();
    }

    let base = tile * LEVELS;
    let mut excess = 0.0f32;
    for level in 0..LEVELS {
        let count = f32::cast_from(histograms[(base + level) as usize]);
        excess += f32::max(count - limit, 0.0);
    }

    let share = excess / LEVEL_COUNT;
    let mut running = 0.0f32;
    for level in 0..LEVELS {
        let count = f32::cast_from(histograms[(base + level) as usize]);
        running += f32::min(count, limit) + share;
        tables[(base + level) as usize] = running / tile_samples;
    }
}

#[cube]
fn table_at(tables: &[f32], tile_x: u32, tile_y: u32, level: u32) -> f32 {
    let tile = tile_y * TILES + tile_x;

    tables[(tile * LEVELS + level) as usize]
}

/// The tile whose centre sits at or before `position`, counted in tiles, kept inside the grid.
#[cube]
fn tile_before(position: f32) -> u32 {
    let floored = f32::floor(position);
    let clamped = f32::max(floored, 0.0);
    let tile = u32::cast_from(clamped);

    u32::min(tile, TILES - 1)
}

/// Maps each sample through the tables of the four nearest tile centres, blended bilinearly.
#[cube(launch_unchecked)]
fn equalise_samples(
    input: &[f32],
    tables: &[f32],
    output: &mut [f32],
    width: u32,
    tile_width: u32,
    tile_height: u32,
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
    let left = tile_before(across);
    let top = tile_before(down);
    let right = u32::min(left + 1, TILES - 1);
    let bottom = u32::min(top + 1, TILES - 1);
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
/// The image is split into an 8 by 8 grid of equal tiles, extended with reflect-101 samples on the
/// right and bottom until each side divides by the grid. Each tile's histogram is clipped at `clip`
/// times its average bin, rounded down and at least 1.
pub(super) fn equalise(
    client: &Client,
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
    let extended_samples = (tile_width * TILES * tile_height * TILES) as usize;
    let tile_samples = (tile_width * tile_height) as f32;
    let limit = (clip * tile_samples / LEVEL_COUNT).floor().max(1.0);

    let entries = (TILES * TILES * LEVELS) as usize;
    let histograms = Scratch::line::<u32>(context, entries);
    let tables = Scratch::line::<f32>(context, entries);

    let cleared = histograms.plane().array();
    let cubes = sizing.0.cube_count(entries);
    let shape = sizing.0.cube_dim();

    // Safety: the histograms hold `entries` u32 values and the kernel writes nothing past them.
    unsafe {
        clear_histograms::launch_unchecked(client, cubes, shape, cleared);
    }

    let input = planes.input.array();
    let counted = histograms.plane().array();
    let cubes = sizing.0.cube_count(extended_samples);
    let shape = sizing.0.cube_dim();

    // Safety: every read is reflected into the input's `samples` f32 values. Each extended row and
    // column falls in a tile below `TILES` on its axis and each level is below `LEVELS`, so every
    // histogram index stays below `entries`.
    unsafe {
        count_levels::launch_unchecked(
            client,
            cubes,
            shape,
            input,
            counted,
            width,
            height,
            tile_width,
            tile_height,
        );
    }

    let counts = histograms.plane().array();
    let built = tables.plane().array();
    let tile_count = (TILES * TILES) as usize;
    let cubes = sizing.0.cube_count(tile_count);
    let shape = sizing.0.cube_dim();

    // Safety: both buffers hold `entries` values and each of the `TILES * TILES` units touches only
    // its own tile's `LEVELS` entries.
    unsafe {
        build_tables::launch_unchecked(client, cubes, shape, counts, built, limit, tile_samples);
    }

    let input = planes.input.array();
    let looked_up = tables.plane().array();
    let output = planes.output.array();
    let cubes = sizing.0.cube_count(samples);
    let shape = sizing.0.cube_dim();

    // Safety: the input and output hold `samples` f32 values, and tiles are clamped below `TILES`
    // and levels below `LEVELS`, so every table index stays below `entries`.
    unsafe {
        equalise_samples::launch_unchecked(
            client,
            cubes,
            shape,
            input,
            looked_up,
            output,
            width,
            tile_width,
            tile_height,
        );
    }
}
