// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{Eedi2Settings, FieldBuffer, FieldPass, field_sample, is_interior};
use crate::Dispatch;

/// Whether the three field rows around `row` stay within `ten` of each other at `column`.
#[cube]
fn is_flat(input: &Array<f32>, width: u32, offset: u32, row: u32, column: u32, ten: f32) -> bool {
    let above = field_sample(input, width, offset, row - 1, column);
    let centre = field_sample(input, width, offset, row, column);
    let below = field_sample(input, width, offset, row + 1, column);

    let above_centre = f32::abs(above - centre) < ten;
    let centre_below = f32::abs(centre - below) < ten;
    let above_below = f32::abs(above - below) < ten;

    above_centre && centre_below && above_below
}

/// Whether the 3 by 3 block around a pixel varies too little to hold an edge.
///
/// The sum shifts after summing while each square shifts first, so above 8 bits the variance can be
/// negative. The plugin's unsigned arithmetic wraps that to a huge value, so a negative variance is not low.
#[cube]
fn has_low_variance(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    row: u32,
    column: u32,
    reduce: f32,
    vthresh_81: f32,
) -> bool {
    let mut sum = 0.0f32;
    let mut sum_of_squares = 0.0f32;

    for row_step in 0..3u32 {
        for column_step in 0..3u32 {
            let sample_row = row + row_step - 1;
            let sample_column = column + column_step - 1;
            let sample = field_sample(input, width, offset, sample_row, sample_column);
            let reduced = f32::floor(sample * reduce);
            sum += sample;
            sum_of_squares += reduced * reduced;
        }
    }

    let reduced_sum = f32::floor(sum * reduce);
    let variance = 9.0 * sum_of_squares - reduced_sum * reduced_sum;

    variance >= 0.0 && variance < vthresh_81
}

/// The thresholds pass 1 compares against.
#[derive(CubeType)]
struct MaskThresholds {
    reduce: f32,
    ten: f32,
    mthresh_squared: f32,
    lthresh: f32,
    vthresh_81: f32,
}

/// Whether a pixel's gradient or curvature is strong enough to hold an edge.
#[cube]
fn is_steep_or_curved(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    row: u32,
    column: u32,
    settings: &MaskThresholds,
) -> bool {
    let above = field_sample(input, width, offset, row - 1, column);
    let centre = field_sample(input, width, offset, row, column);
    let below = field_sample(input, width, offset, row + 1, column);
    let before = field_sample(input, width, offset, row, column - 1);
    let after = field_sample(input, width, offset, row, column + 1);

    let horizontal_step = f32::abs(after - before);
    let horizontal = f32::floor(horizontal_step * settings.reduce);
    let above_below = f32::abs(above - below);
    let above_centre = f32::abs(above - centre);
    let centre_below = f32::abs(centre - below);
    let vertical_step = f32::max(above_below, above_centre);
    let vertical_step = f32::max(vertical_step, centre_below);
    let vertical = f32::floor(vertical_step * settings.reduce);
    let gradient = horizontal * horizontal + vertical * vertical;

    let horizontal_bend = f32::abs(before - 2.0 * centre + after);
    let vertical_bend = f32::abs(above - 2.0 * centre + below);
    let horizontal_curve = f32::floor(horizontal_bend * settings.reduce);
    let vertical_curve = f32::floor(vertical_bend * settings.reduce);
    let curve = horizontal_curve + vertical_curve;

    gradient >= settings.mthresh_squared || curve >= settings.lthresh
}

/// Pass 1's test for one interior pixel.
#[cube]
fn is_edge(
    input: &Array<f32>,
    width: u32,
    offset: u32,
    row: u32,
    column: u32,
    settings: &MaskThresholds,
) -> bool {
    let flat_here = is_flat(input, width, offset, row, column, settings.ten);
    let flat_before = is_flat(input, width, offset, row, column - 1, settings.ten);
    let flat_after = is_flat(input, width, offset, row, column + 1, settings.ten);
    let mut edge = false;

    if !(flat_here || (flat_before && flat_after)) {
        let low_variance = has_low_variance(
            input,
            width,
            offset,
            row,
            column,
            settings.reduce,
            settings.vthresh_81,
        );

        if !low_variance {
            edge = is_steep_or_curved(input, width, offset, row, column, settings);
        }
    }

    edge
}

/// Pass 1, the raw edge mask `E0`, 1 on an edge and 0 elsewhere. The border stays clear.
#[cube(launch_unchecked)]
fn edge_mask_kernel(
    input: &Array<f32>,
    mask: &mut Array<f32>,
    width: u32,
    rows: u32,
    offset: u32,
    reduce: f32,
    ten: f32,
    mthresh_squared: f32,
    lthresh: f32,
    vthresh_81: f32,
) {
    if ABSOLUTE_POS >= mask.len() {
        terminate!();
    }

    let settings = MaskThresholds {
        reduce,
        ten,
        mthresh_squared,
        lthresh,
        vthresh_81,
    };

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let mut value = 0.0f32;

    // The test sits in its own block because the device may evaluate both sides of `&&`, and a border
    // pixel's neighbours can lie outside the plane.
    if is_interior(position, width, rows) {
        let edge = is_edge(input, width, offset, row, column, &settings);

        if edge {
            value = 1.0;
        }
    }

    mask[ABSOLUTE_POS] = value;
}

/// The set pixels among the 8 neighbours of the interior pixel at `position`.
#[cube]
fn neighbours_set(mask: &Array<f32>, width: u32, position: u32) -> f32 {
    let above = position - width;
    let below = position + width;

    let mut count = mask[(above - 1) as usize] + mask[above as usize] + mask[(above + 1) as usize];
    count += mask[(position - 1) as usize] + mask[(position + 1) as usize];
    count += mask[(below - 1) as usize] + mask[below as usize] + mask[(below + 1) as usize];

    count
}

/// Passes 2 and 4, clearing interior pixels with fewer than `strength` set neighbours.
#[cube(launch_unchecked)]
fn erode_kernel(source: &Array<f32>, destination: &mut Array<f32>, width: u32, rows: u32, strength: f32) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let mut value = source[ABSOLUTE_POS];

    if value == 1.0 && is_interior(position, width, rows) {
        let count = neighbours_set(source, width, position);

        if count < strength {
            value = 0.0;
        }
    }

    destination[ABSOLUTE_POS] = value;
}

/// Pass 3, setting interior pixels with at least `strength` set neighbours.
#[cube(launch_unchecked)]
fn dilate_kernel(source: &Array<f32>, destination: &mut Array<f32>, width: u32, rows: u32, strength: f32) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let mut value = source[ABSOLUTE_POS];

    if value == 0.0 && is_interior(position, width, rows) {
        let count = neighbours_set(source, width, position);

        if count >= strength {
            value = 1.0;
        }
    }

    destination[ABSOLUTE_POS] = value;
}

/// Pass 5, the final edge mask `E`, closing gaps of up to 3 pixels in a row and dropping lone pixels.
#[cube(launch_unchecked)]
fn remove_small_gaps_kernel(source: &Array<f32>, destination: &mut Array<f32>, width: u32, rows: u32) {
    if ABSOLUTE_POS >= destination.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;
    let inside = row >= 1 && row + 1 < rows && column >= 3 && column + 3 < width;
    let mut value = source[ABSOLUTE_POS];

    if inside {
        let row_start = row * width;
        let back_3 = source[(row_start + column - 3) as usize] == 1.0;
        let back_2 = source[(row_start + column - 2) as usize] == 1.0;
        let back_1 = source[(row_start + column - 1) as usize] == 1.0;
        let forward_1 = source[(row_start + column + 1) as usize] == 1.0;
        let forward_2 = source[(row_start + column + 2) as usize] == 1.0;
        let forward_3 = source[(row_start + column + 3) as usize] == 1.0;

        if value == 1.0 {
            let alone = !(back_3 || back_2 || back_1 || forward_1 || forward_2 || forward_3);

            if alone {
                value = 0.0;
            }
        } else {
            let bridged_1 = forward_1 && (back_1 || back_2 || back_3);
            let bridged_2 = forward_2 && (back_1 || back_2);
            let bridged_3 = forward_3 && back_1;

            if bridged_1 || bridged_2 || bridged_3 {
                value = 1.0;
            }
        }
    }

    destination[ABSOLUTE_POS] = value;
}

/// Launches pass 1, writing `E0` into the `edges` scratch.
pub(super) fn edge_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input plane holds `width * 2 * rows` values and `edges` holds `width * rows`, as
    // `FieldPass::new` checked. Each unit writes only its own mask pixel and reads field rows within one
    // of its own, at columns within one of its own, only for interior pixels.
    unsafe {
        let input = pass.input.array::<R>();
        let mask = pass.field_array::<R>(FieldBuffer::Edges);

        edge_mask_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            mask,
            pass.width,
            pass.rows,
            pass.offset,
            settings.reduce,
            settings.ten,
            settings.mthresh_squared,
            settings.lthresh,
            settings.vthresh_81,
        );
    }
}

/// Launches passes 2 and 4, eroding the mask in `edges` into `working`.
pub(super) fn erode<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `edges` and `working` each hold `width * rows` values, as `FieldPass::new` checked. Each unit
    // writes only its own pixel and reads its 8 neighbours only for interior pixels.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Edges);
        let destination = pass.field_array::<R>(FieldBuffer::Working);

        erode_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            destination,
            pass.width,
            pass.rows,
            settings.estr,
        );
    }
}

/// Launches pass 3, dilating the mask in `working` into `edges`.
pub(super) fn dilate<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
    settings: &Eedi2Settings,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `working` and `edges` each hold `width * rows` values, as `FieldPass::new` checked. Each unit
    // writes only its own pixel and reads its 8 neighbours only for interior pixels.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Working);
        let destination = pass.field_array::<R>(FieldBuffer::Edges);

        dilate_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            destination,
            pass.width,
            pass.rows,
            settings.dstr,
        );
    }
}

/// Launches pass 5, closing the small gaps of the mask in `working` into `edges`.
pub(super) fn remove_small_gaps<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
) {
    let samples = pass.field_samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: `working` and `edges` each hold `width * rows` values, as `FieldPass::new` checked. Each unit
    // writes only its own pixel and reads 3 columns either side only when they lie inside its row.
    unsafe {
        let source = pass.field_array::<R>(FieldBuffer::Working);
        let destination = pass.field_array::<R>(FieldBuffer::Edges);

        remove_small_gaps_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            source,
            destination,
            pass.width,
            pass.rows,
        );
    }
}
