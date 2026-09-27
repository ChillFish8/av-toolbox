// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{BlurKind, average};
use crate::{DevicePlane, Dispatch};

/// The mask value `offset` steps from `position` along one line, clamped into the line.
#[cube]
fn along(values: &Array<u32>, base: u32, step: u32, position: i32, offset: i32, length: i32) -> i32 {
    let reached = i32::clamp(position + offset, 0, length - 1);
    let index = base + reached as u32 * step;

    i32::cast_from(values[index as usize])
}

/// One r6 blur step.
///
/// Within 6 samples of an end only the samples on the inner side are read. Where both ends are
/// that close, `start_wins` picks the start.
#[cube]
fn r6_at(values: &Array<u32>, base: u32, step: u32, position: i32, length: i32, start_wins: bool) -> u32 {
    let centre = along(values, base, step, position, 0, length);
    let near_start = position < 6;
    let near_end = position >= length - 6;

    let mut direction = 0i32;
    if near_start && (start_wins || !near_end) {
        direction = 1;
    } else if near_end {
        direction = -1;
    }

    let (near, middle, far) = if direction == 0 {
        let before_one = along(values, base, step, position, -1, length);
        let after_one = along(values, base, step, position, 1, length);
        let before_two = along(values, base, step, position, -2, length);
        let after_two = along(values, base, step, position, 2, length);
        let before_three = along(values, base, step, position, -3, length);
        let after_three = along(values, base, step, position, 3, length);
        let before_four = along(values, base, step, position, -4, length);
        let after_four = along(values, base, step, position, 4, length);
        let before_five = along(values, base, step, position, -5, length);
        let after_five = along(values, base, step, position, 5, length);
        let before_six = along(values, base, step, position, -6, length);
        let after_six = along(values, base, step, position, 6, length);

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
        let tap_one = along(values, base, step, position, direction, length);
        let tap_two = along(values, base, step, position, direction * 2, length);
        let tap_three = along(values, base, step, position, direction * 3, length);
        let tap_four = along(values, base, step, position, direction * 4, length);
        let tap_five = along(values, base, step, position, direction * 5, length);
        let tap_six = along(values, base, step, position, direction * 6, length);

        let near = average(tap_one, tap_two);
        let middle = average(tap_three, tap_four);
        let far = average(tap_five, tap_six);
        (near, middle, far)
    };

    let with_centre = average(centre, near);
    let outer = average(middle, far);
    let everything = average(with_centre, outer);
    let blurred = average(with_centre, everything);

    u32::cast_from(blurred)
}

/// One r2 blur step.
#[cube]
fn r2_at(values: &Array<u32>, base: u32, step: u32, position: i32, length: i32) -> u32 {
    let centre = along(values, base, step, position, 0, length);
    let before_one = along(values, base, step, position, -1, length);
    let after_one = along(values, base, step, position, 1, length);
    let before_two = along(values, base, step, position, -2, length);
    let after_two = along(values, base, step, position, 2, length);

    let inner = average(before_one, after_one);
    let outer = average(before_two, after_two);
    let first = average(outer, centre);
    let second = average(first, centre);
    let blurred = average(second, inner);

    u32::cast_from(blurred)
}

#[cube(launch_unchecked)]
fn blur_rows(input: &Array<u32>, output: &mut Array<u32>, width: u32, wide: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % width;
    let base = position - column;
    let length = width as i32;
    let along_row = column as i32;

    if wide == 1u32 {
        output[ABSOLUTE_POS] = r6_at(input, base, 1u32, along_row, length, false);
    } else {
        output[ABSOLUTE_POS] = r2_at(input, base, 1u32, along_row, length);
    }
}

#[cube(launch_unchecked)]
fn blur_columns(input: &Array<u32>, output: &mut Array<u32>, width: u32, height: u32, wide: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let base = position % width;
    let length = height as i32;
    let along_column = (position / width) as i32;

    if wide == 1u32 {
        output[ABSOLUTE_POS] = r6_at(input, base, width, along_column, length, true);
    } else {
        output[ABSOLUTE_POS] = r2_at(input, base, width, along_column, length);
    }
}

/// Blurs `mask` once with `kind`, rows then columns, passing through `scratch`.
///
/// Both planes hold u32 values and are one size.
pub fn blur_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    mask: DevicePlane<'_>,
    scratch: DevicePlane<'_>,
    kind: BlurKind,
) {
    let width = mask.width();
    let height = mask.height();
    let samples = mask.samples();
    let wide = u32::from(kind == BlurKind::R6);

    let mask_read = mask.array::<R>();
    let scratch_write = scratch.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` u32 values, and every read is clamped inside its line.
    unsafe {
        blur_rows::launch_unchecked::<R>(client, cubes, unit, mask_read, scratch_write, width, wide);
    }

    let scratch_read = scratch.array::<R>();
    let mask_write = mask.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: both planes hold `samples` u32 values, and every read is clamped inside its line.
    unsafe {
        blur_columns::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            scratch_read,
            mask_write,
            width,
            height,
            wide,
        );
    }
}
