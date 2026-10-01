// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::exact::mul_exact;
use super::interpolate::clamped_sixteenth;
use super::{FieldPass, plane_row, row_start, sample_at};
use crate::Dispatch;

/// The assembly, one unit per output sample, which copies the kept rows from the input and the rebuilt
/// rows from the scratch.
#[cube(launch_unchecked)]
fn assemble_kernel(
    input: &Array<f32>,
    rebuilt: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    offset: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let row = position / width;
    let column = position % width;

    let mut value = input[ABSOLUTE_POS];
    if row % 2 != offset {
        let field_row = row / 2;
        value = rebuilt[(field_row * width + column) as usize];
    }

    output[ABSOLUTE_POS] = value;
}

/// How far the interpolation strays, combined the way `vcheck` picks.
#[cube]
fn combine(first: i32, second: i32, vcheck: u32) -> i32 {
    let mut combined = i32::max(first, second);
    if vcheck == 1 {
        combined = i32::min(first, second);
    } else if vcheck == 2 {
        combined = (first + second + 1) >> 1;
    }

    combined
}

/// The weights of the reliability check's blend.
#[derive(CubeType)]
struct CheckWeights {
    reciprocal0: f32,
    reciprocal1: f32,
    reciprocal2: f32,
    vthresh2: f32,
}

/// The samples one checked sample reads, beside itself.
///
/// `checked_above` comes from the row two above after its check. The others come from the rebuilt rows
/// and the kept rows before any check.
#[derive(CubeType)]
struct CheckReads {
    here: i32,
    here_ahead: i32,
    here_behind: i32,
    checked_above: i32,
    unchecked_below: i32,
    kept_above: i32,
    kept_below: i32,
    kept_above_ahead: i32,
    kept_below_behind: i32,
}

/// The checked value of a sample from its reads, blending towards `fallback` where the interpolation looks
/// unreliable.
#[cube]
fn blend_check(
    reads: &CheckReads,
    fallback: i32,
    direction: i32,
    vcheck: u32,
    weights: &CheckWeights,
) -> i32 {
    let top = (reads.checked_above + reads.here_behind + 1) >> 1;
    let bottom = (reads.here_ahead + reads.unchecked_below + 1) >> 1;
    let top_variation = i32::abs(reads.checked_above - reads.kept_above_ahead)
        + i32::abs(reads.here_ahead - reads.kept_above_ahead);
    let bottom_variation = i32::abs(reads.unchecked_below - reads.kept_below_behind)
        + i32::abs(reads.here_behind - reads.kept_below_behind);
    let centre_variation = i32::abs(reads.here - reads.kept_above) + i32::abs(reads.here - reads.kept_below);
    let top_offset = i32::abs(top - reads.kept_above);
    let bottom_offset = i32::abs(bottom - reads.kept_below);
    let top_change = i32::abs(top_variation - centre_variation);
    let bottom_change = i32::abs(bottom_variation - centre_variation);
    let offset = combine(top_offset, bottom_offset, vcheck);
    let change = combine(top_change, bottom_change, vcheck);

    let offset_weight = offset as f32 * weights.reciprocal0;
    let change_weight = change as f32 * weights.reciprocal1;
    let lean_left = weights.vthresh2 - i32::abs(direction) as f32;
    let lean_weight = f32::max(lean_left * weights.reciprocal2, 0.0);
    let stronger = f32::max(offset_weight, change_weight);
    let strongest = f32::max(stronger, lean_weight);
    let weight = f32::min(strongest, 1.0);
    let keep = 1.0 - weight;
    let here_sample = reads.here as f32;
    let fallback_sample = fallback as f32;

    // Both products are exact, so no multiply can fuse into the sum and it rounds as the plugin's does.
    let kept_part = mul_exact(keep, here_sample);
    let fallback_part = mul_exact(weight, fallback_sample);
    let blended = kept_part + fallback_part;

    blended as i32
}

/// `column` moved by `shift`, clamped into the row.
#[cube]
fn shifted(column: u32, shift: i32, width: u32) -> u32 {
    let moved = column as i32 + shift;
    let inside = i32::clamp(moved, 0, width as i32 - 1);

    inside as u32
}

/// The value the check falls back to at `column` of plane row `row`, from `sclip` or the vertical cubic.
#[cube]
fn fallback_value(
    input: &Array<f32>,
    sclip: &Array<f32>,
    row: u32,
    column: u32,
    width: u32,
    height: u32,
    peak: u32,
    #[comptime] has_sclip: bool,
) -> i32 {
    if comptime!(has_sclip) {
        sclip[(row * width + column) as usize] as i32
    } else {
        let signed_row = row as i32;
        let signed_column = column as i32;
        let near_above_start = (row - 1) * width;
        let near_below_start = (row + 1) * width;
        let far_above_start = row_start(signed_row - 3, width, height);
        let far_below_start = row_start(signed_row + 3, width, height);
        let near_above = sample_at(input, near_above_start, signed_column, width);
        let near_below = sample_at(input, near_below_start, signed_column, width);
        let far_above = sample_at(input, far_above_start, signed_column, width);
        let far_below = sample_at(input, far_below_start, signed_column, width);
        let numerator = 9 * (near_above + near_below) - (far_above + far_below) + 8;
        clamped_sixteenth(numerator, peak)
    }
}

/// The reliability check of one rebuilt row, one unit per column.
///
/// It reads the row two above from `output`, where it has already been checked, and itself and the row
/// two below from `rebuilt`, before their checks. Without `sclip`, the input stands in for it and is never
/// read as one.
#[cube(launch_unchecked)]
fn check_kernel(
    input: &Array<f32>,
    rebuilt: &Array<f32>,
    directions: &Array<i32>,
    sclip: &Array<f32>,
    output: &mut Array<f32>,
    width: u32,
    height: u32,
    field_row: u32,
    first_rebuilt: u32,
    peak: u32,
    vcheck: u32,
    reciprocal0: f32,
    reciprocal1: f32,
    reciprocal2: f32,
    vthresh2: f32,
    #[comptime] has_sclip: bool,
) {
    let column = ABSOLUTE_POS as u32;
    if column >= width {
        terminate!();
    }

    let row = plane_row(field_row, first_rebuilt);
    let here_start = field_row * width;
    let above_start = here_start - width;
    let below_start = here_start + width;
    let kept_above_start = (row - 1) * width;
    let kept_below_start = (row + 1) * width;

    let fallback = fallback_value(input, sclip, row, column, width, height, peak, has_sclip);

    let direction = directions[(here_start + column) as usize];
    let direction_above = directions[(above_start + column) as usize];
    let direction_below = directions[(below_start + column) as usize];
    let opposed = i32::max(direction * direction_above, direction * direction_below) < 0;
    let neither = direction_above == 0 && direction_below == 0;

    let mut value = fallback;
    if direction != 0 && !opposed && !neither {
        let ahead = shifted(column, direction, width);
        let behind = shifted(column, -direction, width);
        let checked_row = (row - 2) * width;
        let reads = CheckReads {
            here: rebuilt[(here_start + column) as usize] as i32,
            here_ahead: rebuilt[(here_start + ahead) as usize] as i32,
            here_behind: rebuilt[(here_start + behind) as usize] as i32,
            checked_above: output[(checked_row + ahead) as usize] as i32,
            unchecked_below: rebuilt[(below_start + behind) as usize] as i32,
            kept_above: input[(kept_above_start + column) as usize] as i32,
            kept_below: input[(kept_below_start + column) as usize] as i32,
            kept_above_ahead: input[(kept_above_start + ahead) as usize] as i32,
            kept_below_behind: input[(kept_below_start + behind) as usize] as i32,
        };
        let weights = CheckWeights {
            reciprocal0,
            reciprocal1,
            reciprocal2,
            vthresh2,
        };
        value = blend_check(&reads, fallback, direction, vcheck, &weights);
    }

    output[(row * width + column) as usize] = value as f32;
}

/// Launches the assembly of the output from the kept rows and the rebuilt rows.
pub(super) fn assemble<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
) {
    let samples = pass.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and output hold `width * height` f32 values and `rebuilt` a value per rebuilt
    // sample, as `FieldPass::new` checked. Each unit writes only its own output sample, and reads the
    // input or the scratch at the same position.
    unsafe {
        let input = pass.input.array::<R>();
        let rebuilt = pass.rebuilt_array::<R>();
        let output = pass.output.array::<R>();

        assemble_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            rebuilt,
            output,
            pass.width,
            pass.offset,
        );
    }
}

/// Launches the reliability check down the rebuilt rows that have a rebuilt row two above and two below.
///
/// Each row reads the row above after its check, so every row is its own launch, which the device finishes
/// before the next starts.
pub(super) fn reliability_check<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    pass: &FieldPass<'_>,
) {
    let cubes = dispatch.cube_count(pass.width as usize);
    let unit = dispatch.cube_dim();
    let weights = pass.settings.weights;
    let has_sclip = pass.sclip.is_some();
    let sclip = pass.sclip.unwrap_or(pass.input);

    for field_row in 0..pass.rows {
        let row = 2 * field_row + pass.first_rebuilt;
        if row < 2 || row + 3 > pass.height {
            continue;
        }

        // Safety: the input, output and `sclip` hold `width * height` f32 values, and `rebuilt` and
        // `directions` a value per rebuilt sample, as `FieldPass::new` checked. Each unit writes only its own
        // sample of plane row `row`, reads the rebuilt rows either side, which exist for every row checked,
        // and clamps every column it reads into the row.
        unsafe {
            let input = pass.input.array::<R>();
            let rebuilt = pass.rebuilt_array::<R>();
            let directions = pass.directions_array::<R>();
            let sclip = sclip.array::<R>();
            let output = pass.output.array::<R>();

            check_kernel::launch_unchecked::<R>(
                client,
                cubes.clone(),
                unit,
                input,
                rebuilt,
                directions,
                sclip,
                output,
                pass.width,
                pass.height,
                field_row,
                pass.first_rebuilt,
                pass.settings.peak,
                pass.settings.vcheck,
                weights.reciprocal0,
                weights.reciprocal1,
                weights.reciprocal2,
                weights.vthresh2,
                has_sclip,
            );
        }
    }
}
