// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::lines::{both_pass_cubes, line_shape, sample_index};
use super::{BlurPlanes, Gaussian, MAX_CUBES_PER_AXIS, MAX_SHARED_BYTES, Storage};
use crate::Edge;
use crate::border::fold;

/// The lines, rows or columns, a cube blurs side by side.
const LANES: u32 = 16;
/// The units a cube spreads along each line.
const GROUPS: u32 = 16;
/// The neighbouring outputs each unit writes along its line.
const RUN: u32 = 16;
/// The outputs a cube writes along each line.
const SEGMENT: u32 = GROUPS * RUN;
/// The taps a cube sums from one staged chunk.
const SPAN: u32 = 224;
/// The samples of each line one chunk stages.
const CHUNK: u32 = SEGMENT + SPAN - 1;
/// The distance between neighbouring line samples in shared memory.
///
/// It is one longer than the lanes, so a cube storing along a line spreads over the banks.
const STRIDE: u32 = LANES + 1;

const _: () = assert!(SPAN.is_multiple_of(RUN));
const _: () = assert!(LANES == GROUPS);
const _: () = assert!(CHUNK * STRIDE * size_of::<f32>() as u32 <= MAX_SHARED_BYTES);

/// The taps of a blur reaching `radius`, rounded up to a whole count of runs.
#[cube]
pub(super) fn padded_tap_count(radius: u32) -> u32 {
    let tap_count = 2 * radius + 1;

    tap_count.div_ceil(RUN) * RUN
}

/// The cubes one pass along lines of `line_length` samples takes, `line_count` lines in all.
fn pass_cubes(line_length: u32, line_count: u32) -> Option<CubeCount> {
    let segments = line_length.div_ceil(SEGMENT);
    let blocks = line_count.div_ceil(LANES);
    if segments > MAX_CUBES_PER_AXIS || blocks > MAX_CUBES_PER_AXIS {
        return None;
    }

    Some(CubeCount::Static(segments, blocks, 1))
}

/// The cubes both passes over a `width` by `height` plane take.
///
/// It is `None` when either pass needs more cubes along an axis than every device allows.
pub(super) fn staged_cubes(width: u32, height: u32) -> Option<(CubeCount, CubeCount)> {
    both_pass_cubes(width, height, pass_cubes)
}

/// The line and position, within the cube, that a unit loads or stores on `step`.
///
/// Neighbouring units take neighbouring samples of the plane, so global reads and writes coalesce.
#[cube]
fn moved_slot(step: u32, #[comptime] along_rows: bool) -> (u32, u32) {
    let mut local_line = UNIT_POS_X;
    let mut local_position = UNIT_POS_Y + step * GROUPS;

    if along_rows {
        local_line = UNIT_POS_Y;
        local_position = UNIT_POS_X + step * LANES;
    }

    (local_line, local_position)
}

/// Copies `count` samples of each of the cube's lines, from `first_position` on, into `staged`.
///
/// Positions are folded into the line, clamped when `clamp` is set and reflected otherwise. Lines
/// past the plane repeat its last line.
#[cube]
fn load_lines<F: Float>(
    input: &Array<F>,
    staged: &mut SharedMemory<f32>,
    first_position: i32,
    count: u32,
    width: u32,
    height: u32,
    #[comptime] along_rows: bool,
    #[comptime] clamp: bool,
) {
    let (line_length, line_count) = line_shape(width, height, along_rows);
    let first_line = CUBE_POS_Y * LANES;
    let last_line = line_count - 1;
    let steps = count.div_ceil(GROUPS);

    for step in 0..steps {
        let (local_line, local_position) = moved_slot(step, along_rows);
        if local_position < count {
            let line = min(first_line + local_line, last_line);
            let position = fold(first_position + local_position as i32, line_length, clamp);
            let index = sample_index(line, position, width, along_rows);
            let sample = f32::cast_from(input[index as usize]);
            staged[(local_position * STRIDE + local_line) as usize] = sample;
        }
    }
}

/// Writes the cube's `SEGMENT` outputs of each line from `staged` to the ones inside the plane.
#[cube]
fn store_lines<F: Float>(
    staged: &SharedMemory<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    #[comptime] along_rows: bool,
) {
    let (line_length, line_count) = line_shape(width, height, along_rows);
    let first_line = CUBE_POS_Y * LANES;
    let first_position = CUBE_POS_X * SEGMENT;

    #[unroll]
    for step in 0..comptime!(SEGMENT / GROUPS) {
        let (local_line, local_position) = moved_slot(step, along_rows);
        let line = first_line + local_line;
        let position = first_position + local_position;
        if line < line_count && position < line_length {
            let index = sample_index(line, position, width, along_rows);
            let total = staged[(local_position * STRIDE + local_line) as usize];
            output[index as usize] = F::cast_from(total);
        }
    }
}

/// Adds `tap_count` taps of the staged chunk, from `first_tap` on, into the unit's `totals`.
///
/// The unit slides a window of `RUN` samples down its line, so each staged sample is read once and
/// weighs into every output it reaches. `tap_count` is a multiple of `RUN`, so the window turns a
/// whole lap each step and its slots stay fixed.
#[cube]
fn add_taps(
    staged: &SharedMemory<f32>,
    weights: &Array<f32>,
    totals: &mut Array<f32>,
    first_tap: u32,
    tap_count: u32,
) {
    let lane = UNIT_POS_X;
    let first_output = UNIT_POS_Y * RUN;
    let mut window = Array::<f32>::new(comptime!(RUN as usize));

    #[unroll]
    for slot in 0..comptime!(RUN - 1) {
        window[slot as usize] = staged[((first_output + slot) * STRIDE + lane) as usize];
    }

    for lap in 0..tap_count / RUN {
        let lap_start = lap * RUN;

        #[unroll]
        for turn in 0..RUN {
            let newest = first_output + lap_start + turn + RUN - 1;
            let newest_slot = comptime!((turn + RUN - 1) % RUN);
            window[newest_slot as usize] = staged[(newest * STRIDE + lane) as usize];
            let weight = weights[(first_tap + lap_start + turn) as usize];

            #[unroll]
            for output in 0..RUN {
                let slot = comptime!((output + turn) % RUN);
                totals[output as usize] += window[slot as usize] * weight;
            }
        }
    }
}

/// Blurs `LANES` lines of the plane over one segment of `SEGMENT` outputs each.
///
/// The window is walked in chunks of `SPAN` taps, each staged through shared memory, so any radius fits.
#[cube(launch_unchecked)]
fn blur_lines<F: Float>(
    input: &Array<F>,
    weights: &Array<f32>,
    output: &mut Array<F>,
    width: u32,
    height: u32,
    radius: u32,
    #[comptime] along_rows: bool,
    #[comptime] clamp: bool,
) {
    let mut staged = SharedMemory::<f32>::new(comptime!((CHUNK * STRIDE) as usize));
    let mut totals = Array::<f32>::new(comptime!(RUN as usize));

    #[unroll]
    for slot in 0..RUN {
        totals[slot as usize] = 0.0;
    }

    let tap_count = padded_tap_count(radius);
    let chunk_count = tap_count.div_ceil(SPAN);
    let window_start = (CUBE_POS_X * SEGMENT) as i32 - radius as i32;

    for chunk in 0..chunk_count {
        let first_tap = chunk * SPAN;
        let chunk_taps = min(tap_count - first_tap, SPAN);
        let first_position = window_start + first_tap as i32;
        let count = SEGMENT + chunk_taps - 1;

        sync_cube();

        load_lines(
            input,
            &mut staged,
            first_position,
            count,
            width,
            height,
            along_rows,
            clamp,
        );

        sync_cube();

        add_taps(&staged, weights, &mut totals, first_tap, chunk_taps);
    }

    sync_cube();

    let first_output = UNIT_POS_Y * RUN;
    #[unroll]
    for slot in 0..RUN {
        let local_position = first_output + slot;
        staged[(local_position * STRIDE + UNIT_POS_X) as usize] = totals[slot as usize];
    }

    sync_cube();

    store_lines(&staged, output, width, height, along_rows);
}

/// Blurs `planes.input` across into `planes.scratch`, then down into `planes.output`.
///
/// Each pass stages its lines through shared memory, reading past the edge as `edge` says. It panics
/// when the plane needs more cubes than [staged_cubes](crate::blur::staged::staged_cubes) allows.
pub(super) fn blur_staged<R: Runtime>(
    client: &ComputeClient<R>,
    gaussian: &Gaussian,
    planes: BlurPlanes<'_>,
    edge: Edge,
    storage: Storage,
) {
    let width = planes.output.width();
    let height = planes.output.height();
    let cubes = staged_cubes(width, height);
    let (row_cubes, column_cubes) = cubes.expect("the plane should fit the staged blur's cube count");
    let shape = CubeDim::new_2d(LANES, GROUPS);
    let radius = gaussian.radius();
    let clamp = edge == Edge::Clamp;

    let weights = gaussian.padded_weights::<R>();
    let input = planes.input.array::<R>();
    let scratch = planes.scratch.array::<R>();

    // Safety: every plane holds `width * height` values of `storage`, every read is reflected or
    // clamped into the plane and every write is guarded to it.
    unsafe {
        match storage {
            Storage::F32 => blur_lines::launch_unchecked::<f32, R>(
                client, row_cubes, shape, input, weights, scratch, width, height, radius, true, clamp,
            ),
            Storage::F16 => blur_lines::launch_unchecked::<half::f16, R>(
                client, row_cubes, shape, input, weights, scratch, width, height, radius, true, clamp,
            ),
        }
    }

    let column_weights = gaussian.padded_weights::<R>();
    let scratch = planes.scratch.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: every plane holds `width * height` values of `storage`, every read is reflected or
    // clamped into the plane and every write is guarded to it.
    unsafe {
        match storage {
            Storage::F32 => blur_lines::launch_unchecked::<f32, R>(
                client,
                column_cubes,
                shape,
                scratch,
                column_weights,
                output,
                width,
                height,
                radius,
                false,
                clamp,
            ),
            Storage::F16 => blur_lines::launch_unchecked::<half::f16, R>(
                client,
                column_cubes,
                shape,
                scratch,
                column_weights,
                output,
                width,
                height,
                radius,
                false,
                clamp,
            ),
        }
    }
}
