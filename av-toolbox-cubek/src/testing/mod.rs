mod artcnn;
mod based;
mod bilateral;
mod bwdif;
mod degrain;
mod dehalo;
mod dering;
mod edges;
mod eedi2;
mod eedi3;
mod hysteresis;
mod ivtc;
mod lineart;
mod motion;
mod neighbourhood;
mod normalise;
mod padding;
mod qtgmc;
mod resize;
mod retinex;
mod rgvs;
mod sangnom;
#[cfg(test)]
mod tests;
mod tuning;

use std::fmt::{Debug, Display};

pub use self::artcnn::{
    GOLDEN_R8F64,
    golden_input,
    reference_artcnn,
    reference_conv3x3,
    reference_depth_to_space,
    reference_silu,
    synthetic_weights,
};
pub use self::based::{
    BOUND_MASK_PEAK,
    BasedFinish,
    BasedParameters,
    BasedStages,
    Direction,
    peak_of,
    reference_based_aa,
    reference_based_finish,
    reference_based_mask,
    reference_binarize_prewitt,
    reference_box_halve,
    reference_double_mclip,
    reference_median_of_three,
    reference_pscale_blend,
    reference_supersample,
};
pub use self::bilateral::{reference_bilateral, reference_range_weight};
pub use self::bwdif::{BWDIF_FIXTURES, BwdifFixture, BwdifReferencePlanes, reference_bwdif};
pub use self::degrain::{ReferenceNeighbour, reference_degrain, reference_scene_changed};
pub use self::dehalo::{
    reference_dehalo_alpha,
    reference_edge_clean,
    reference_fine_dehalo,
    reference_merge_masked,
    reference_padded_warp,
    reference_yahr,
};
pub use self::dering::{reference_hq_dering, reference_ring_mask};
pub use self::edges::{reference_add_capped, reference_binarize, reference_edges};
pub use self::eedi2::{
    EEDI2_FIXTURES,
    Eedi2Fixture,
    Eedi2Parameters,
    Eedi2Stages,
    reference_eedi2_antialias,
    reference_eedi2_field,
    reference_eedi2_field_stages,
    reference_fill_round,
};
pub use self::eedi3::{
    CheckOrder,
    CloseDecisionTally,
    Contraction,
    EEDI3_FIXTURES,
    Eedi3Antialias,
    Eedi3Clips,
    Eedi3Fixture,
    Eedi3Parameters,
    Eedi3Pass,
    Eedi3PlaneKind,
    Eedi3Variant,
    Lending,
    SplitMix,
    TieOrder,
    eedi3_checked_rows,
    merge_eedi3_fields,
    random_eedi3_mask,
    random_eedi3_parameters,
    random_eedi3_plane,
    reference_eedi3_antialias,
    reference_eedi3_field,
    reference_eedi3_field_variant,
    worked_edge_plane,
};
pub use self::hysteresis::reference_hysteresis;
pub use self::ivtc::{
    DecimateDropFixture,
    DecimateMetricFixture,
    DecisionFixture,
    IntPlane,
    MapFixture,
    MaskFixture,
    SyntheticClip,
    VfmMetrics,
    VfmSettings,
    decimate_drop_fixtures,
    decimate_metric_fixtures,
    excluded_map_fixtures,
    map_fixtures,
    mask_fixtures,
    reference_chroma_into_luma,
    reference_comb_mask,
    reference_compare_plane,
    reference_decimate_metric,
    reference_difference,
    reference_field_sums,
    reference_match_weave,
    reference_mic,
    reference_vfm_metrics,
    reference_weave,
    scene_difference,
    synthetic_clips,
    synthetic_film_frame,
    vfm_decisions,
    vfm_matched_frames,
};
pub use self::lineart::{
    reference_clamped_merge,
    reference_darken,
    reference_merge,
    reference_thin,
    reference_toon,
    reference_vertical_121,
    reference_vinverse,
};
pub use self::motion::{
    MOTION_FIXTURES,
    MotionClip,
    MotionFixture,
    ReferenceField,
    SettingFixture,
    fixture_f32s,
    fixture_i32s,
    half_pel_sample,
    motion_clips,
    normalised,
    object_mask,
    psnr,
    psnr_within,
    reference_chain_seed,
    reference_clip_pyramids,
    reference_compensate,
    reference_confidence,
    reference_covering_blocks,
    reference_downscale,
    reference_nlm_mc_warp,
    reference_pyramid,
    reference_refine_half_pel,
    reference_refine_seed,
    reference_search,
    vector_accuracy,
    write_luma_y4m,
    write_motion_inputs,
};
pub use self::neighbourhood::{
    reference_chain,
    reference_chain_with_edge,
    reference_extreme,
    reference_median,
    reference_min_blur,
    reference_prewitt,
    reference_smooth,
    reference_spread,
    reference_spread_with_edge,
};
pub use self::normalise::reference_normalise;
pub use self::padding::{reference_crop, reference_pad};
pub use self::qtgmc::{
    FixturePlane,
    QtgmcFixture,
    qtgmc_fixture,
    reference_bob,
    reference_shimmer,
    reference_tr0,
    write_qtgmc_inputs,
};
pub use self::resize::{reference_resize, reference_resize_float, reference_resize_wide};
pub use self::retinex::{
    reference_log_response,
    reference_percentile,
    reference_recursive_gaussian,
    reference_recursive_gaussian_cascade,
    reference_retinex,
};
pub use self::rgvs::{reference_remove_grain, reference_repair};
pub use self::sangnom::{
    SANGNOM_FIXTURES,
    SangnomFixture,
    reference_sangnom_antialias,
    reference_sangnom_field,
};
pub use self::tuning::{failing_kernels, fresh_tune_label};
use crate::deband::{DebandParams, advance, unit};
use crate::depth::{Conversion, Shift, Thresholds, convert_sample};
pub use crate::ivtc::{FieldSums, MatchSums};
use crate::warp::{BlurKind, WarpParams};

/// Mirrors [reflect](crate::reflect) on the CPU.
pub fn reflect(index: i32, length: i32) -> i32 {
    if length <= 1 {
        return 0;
    }

    let period = 2 * length - 2;
    let folded = index.abs() % period;
    if folded >= length { period - folded } else { folded }
}

/// `values`, `width` samples wide, with its rows and columns swapped.
pub fn transposed_plane<T: Copy + Default>(values: &[T], width: u32, height: u32) -> Vec<T> {
    let columns = width as usize;
    let rows = height as usize;

    let mut swapped = vec![T::default(); values.len()];
    for row in 0..rows {
        for column in 0..columns {
            swapped[column * rows + row] = values[row * columns + column];
        }
    }

    swapped
}

/// A separable blur on the CPU with `weights` along each axis and reflect-101 borders.
pub fn reference_blur(input: &[f32], width: u32, height: u32, weights: &[f32]) -> Vec<f32> {
    let radius = (weights.len() / 2) as i32;
    let width = width as i32;
    let height = height as i32;

    let mut rows = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(column + tap as i32 - radius, width);
                total += input[(row * width + source) as usize] * weight;
            }

            rows[(row * width + column) as usize] = total;
        }
    }

    let mut output = vec![0.0f32; input.len()];
    for row in 0..height {
        for column in 0..width {
            let mut total = 0.0f32;
            for (tap, weight) in weights.iter().enumerate() {
                let source = reflect(row + tap as i32 - radius, height);
                total += rows[(source * width + column) as usize] * weight;
            }

            output[(row * width + column) as usize] = total;
        }
    }

    output
}

/// Converts a plane `width` samples across on the CPU, mirroring [convert](crate::depth::convert).
pub fn reference_depth(
    input: &[u16],
    width: u32,
    conversion: &Conversion,
    thresholds: &Thresholds,
    shift: Shift,
) -> Vec<u16> {
    input
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let column = index as u32 % width;
            let row = index as u32 / width;
            let threshold = thresholds.at(column, row, shift);

            convert_sample(*value, conversion.multiplier, threshold, conversion.largest)
        })
        .collect()
}

/// The sample nearest to `offset_x`, `offset_y` from the centre of `column`, `row`, clamped to the
/// plane.
fn nearest(
    input: &[u16],
    width: u32,
    height: u32,
    column: u32,
    row: u32,
    offset_x: f32,
    offset_y: f32,
) -> f32 {
    let x = (column as f32 + 0.5 + offset_x).floor();
    let y = (row as f32 + 0.5 + offset_y).floor();
    let cast_x = x.clamp(0.0, (width - 1) as f32) as u32;
    let cast_y = y.clamp(0.0, (height - 1) as f32) as u32;
    let clamped_x = u32::min(cast_x, width - 1);
    let clamped_y = u32::min(cast_y, height - 1);

    f32::from(input[(clamped_y * width + clamped_x) as usize])
}

/// Debands a plane on the CPU, mirroring [deband](crate::deband::deband).
pub fn reference_deband(
    input: &[u16],
    width: u32,
    height: u32,
    params: &DebandParams,
    salt: u32,
) -> Vec<u16> {
    let threshold = params.threshold / 1000.0;
    let grain = params.grain / 1000.0;
    let neutral = params.neutral / params.peak;
    let draws = params.draws();

    let mut output = Vec::with_capacity(input.len());
    for row in 0..height {
        for column in 0..width {
            let centre = f32::from(input[(row * width + column) as usize]);
            let mut state = [column, row, salt];
            let mut result = centre / params.peak;

            for draw in 0..draws {
                advance(&mut state);
                let first = unit(state[0]);
                let second = unit(state[1]);

                if draw < params.iterations {
                    let step = (draw + 1) as f32;
                    let distance = first * step * params.radius;
                    let angle = second * std::f32::consts::TAU;
                    let offset_x = distance * angle.cos();
                    let offset_y = distance * angle.sin();

                    let total = nearest(input, width, height, column, row, offset_x, offset_y)
                        + nearest(input, width, height, column, row, -offset_x, offset_y)
                        + nearest(input, width, height, column, row, -offset_x, -offset_y)
                        + nearest(input, width, height, column, row, offset_x, -offset_y);
                    let average = total / (4.0 * params.peak);
                    let bound = threshold / step;

                    if (result - average).abs() <= bound {
                        result = average;
                    }
                } else {
                    let strength = (result - neutral).abs().min(grain);
                    result += strength * (first - 0.5);
                }
            }

            let scaled = (result * params.peak + 0.5).floor();
            let clamped = scaled.clamp(0.0, params.peak);
            output.push(clamped as u16);
        }
    }

    output
}

/// The rounded mean of two values, the way aWarpSharp2 averages.
fn rounded_mean(first: i32, second: i32) -> i32 {
    (first + second + 1) >> 1
}

fn sample_at(input: &[u16], width: i32, column: i32, row: i32) -> i32 {
    let index = row * width + column;

    i32::from(input[index as usize])
}

/// Builds aWarpSharp2's edge mask on the CPU, mirroring [edge_mask](crate::warp::edge_mask).
///
/// Every plane is one size. A plane under 3 samples across or down gets an all-zero mask.
pub fn reference_edge_mask(input: &[u16], width: u32, height: u32, params: &WarpParams) -> Vec<u32> {
    if width < 3 || height < 3 {
        return vec![0; input.len()];
    }

    let peak = params.peak();
    let thresh = params.scaled_thresh();
    let columns = width as i32;
    let rows = height as i32;

    let mut mask = Vec::with_capacity(input.len());
    for row in 0..rows {
        for column in 0..columns {
            let x = column.clamp(1, columns - 2);
            let y = row.clamp(1, rows - 2);

            let top_left = sample_at(input, columns, x - 1, y - 1);
            let top = sample_at(input, columns, x, y - 1);
            let top_right = sample_at(input, columns, x + 1, y - 1);
            let left = sample_at(input, columns, x - 1, y);
            let right = sample_at(input, columns, x + 1, y);
            let bottom_left = sample_at(input, columns, x - 1, y + 1);
            let bottom = sample_at(input, columns, x, y + 1);
            let bottom_right = sample_at(input, columns, x + 1, y + 1);

            let top_corners = rounded_mean(top_left, top_right);
            let bottom_corners = rounded_mean(bottom_left, bottom_right);
            let left_corners = rounded_mean(bottom_left, top_left);
            let right_corners = rounded_mean(bottom_right, top_right);
            let above = rounded_mean(top, top_corners);
            let below = rounded_mean(bottom, bottom_corners);
            let before = rounded_mean(left, left_corners);
            let after = rounded_mean(right, right_corners);

            let vertical = (above - below).abs();
            let horizontal = (before - after).abs();
            let strongest = horizontal.max(vertical);
            let summed = (vertical + horizontal).min(peak);
            let boosted = (summed + strongest).min(peak);
            let doubled = (boosted * 2).min(peak);
            let tripled = (doubled + boosted).min(peak);
            let edge = (tripled * 2).min(peak);
            let capped = edge.min(thresh);

            mask.push(capped as u32);
        }
    }

    mask
}

/// The value `offset` steps from `position` along one line of `values`, clamped into the line.
fn along_line(values: &[u32], base: u32, step: u32, position: i32, offset: i32, length: i32) -> i32 {
    let reached = (position + offset).clamp(0, length - 1);
    let index = base + reached as u32 * step;

    values[index as usize] as i32
}

/// One r6 blur step. `start_wins` picks the start zone where the start and end zones overlap.
fn r6_step(values: &[u32], base: u32, step: u32, position: i32, length: i32, start_wins: bool) -> u32 {
    let centre = along_line(values, base, step, position, 0, length);
    let near_start = position < 6;
    let near_end = position >= length - 6;

    let direction = if near_start && (start_wins || !near_end) {
        1
    } else if near_end {
        -1
    } else {
        0
    };

    let (near, middle, far) = if direction == 0 {
        let pairs: Vec<i32> = (1..=6)
            .map(|reach| {
                let before = along_line(values, base, step, position, -reach, length);
                let after = along_line(values, base, step, position, reach, length);

                rounded_mean(before, after)
            })
            .collect();

        let near = rounded_mean(pairs[0], pairs[1]);
        let middle = rounded_mean(pairs[2], pairs[3]);
        let far = rounded_mean(pairs[4], pairs[5]);
        (near, middle, far)
    } else {
        let taps: Vec<i32> = (1..=6)
            .map(|reach| along_line(values, base, step, position, direction * reach, length))
            .collect();

        let near = rounded_mean(taps[0], taps[1]);
        let middle = rounded_mean(taps[2], taps[3]);
        let far = rounded_mean(taps[4], taps[5]);
        (near, middle, far)
    };

    let with_centre = rounded_mean(centre, near);
    let outer = rounded_mean(middle, far);
    let everything = rounded_mean(with_centre, outer);
    let blurred = rounded_mean(with_centre, everything);

    blurred as u32
}

/// One r2 blur step.
fn r2_step(values: &[u32], base: u32, step: u32, position: i32, length: i32) -> u32 {
    let centre = along_line(values, base, step, position, 0, length);
    let before_one = along_line(values, base, step, position, -1, length);
    let after_one = along_line(values, base, step, position, 1, length);
    let before_two = along_line(values, base, step, position, -2, length);
    let after_two = along_line(values, base, step, position, 2, length);

    let inner = rounded_mean(before_one, after_one);
    let outer = rounded_mean(before_two, after_two);
    let first = rounded_mean(outer, centre);
    let second = rounded_mean(first, centre);
    let blurred = rounded_mean(second, inner);

    blurred as u32
}

fn blur_step(
    values: &[u32],
    base: u32,
    step: u32,
    position: i32,
    length: i32,
    kind: BlurKind,
    start_wins: bool,
) -> u32 {
    match kind {
        BlurKind::R6 => r6_step(values, base, step, position, length, start_wins),
        BlurKind::R2 => r2_step(values, base, step, position, length),
    }
}

/// Blurs an edge mask once on the CPU, rows then columns, mirroring [blur_mask](crate::warp::blur_mask).
///
/// Across a row the end zone wins where both r6 zones reach, and down a column the start zone wins, as the
/// plugin's loop order has it.
pub fn reference_blur_mask(mask: &[u32], width: u32, height: u32, kind: BlurKind) -> Vec<u32> {
    let columns = width as i32;
    let rows = height as i32;

    let mut across = vec![0u32; mask.len()];
    for row in 0..height {
        let base = row * width;
        for column in 0..width {
            let blurred = blur_step(mask, base, 1, column as i32, columns, kind, false);
            across[(base + column) as usize] = blurred;
        }
    }

    let mut output = vec![0u32; mask.len()];
    for row in 0..height {
        for column in 0..width {
            let blurred = blur_step(&across, column, width, row as i32, rows, kind, true);
            output[(row * width + column) as usize] = blurred;
        }
    }

    output
}

fn mask_at(mask: &[u32], width: i32, column: i32, row: i32) -> i32 {
    let index = row * width + column;

    mask[index as usize] as i32
}

/// Moves each sample along the mask's slope on the CPU, mirroring [warp_plane](crate::warp::warp_plane).
pub fn reference_warp(input: &[u16], mask: &[u32], width: u32, height: u32, params: &WarpParams) -> Vec<u16> {
    let columns = width as i32;
    let rows = height as i32;
    let last_column = columns - 1;
    let last_row = rows - 1;
    let extra_bits = params.bits as i32 - 8;
    let scaled_depth = params.depth << 8;
    let peak = params.peak();

    let mut output = Vec::with_capacity(input.len());
    for y in 0..rows {
        for x in 0..columns {
            let row_above = (y - 1).max(0);
            let row_below = (y + 1).min(last_row);
            let column_before = (x - 1).max(0);
            let column_after = (x + 1).min(last_column);
            let above = mask_at(mask, columns, x, row_above);
            let below = mask_at(mask, columns, x, row_below);
            let before = mask_at(mask, columns, column_before, y);
            let after = mask_at(mask, columns, column_after, y);

            let horizontal_shifted = (before - after) >> extra_bits;
            let vertical_shifted = (above - below) >> extra_bits;
            let horizontal_fixed = ((horizontal_shifted << 7) * scaled_depth) >> 16;
            let vertical_fixed = ((vertical_shifted << 7) * scaled_depth) >> 16;

            let lowest = -y * 128;
            let highest = (rows - y) * 128 - 129;
            let vertical_limited = vertical_fixed.max(lowest).min(highest);

            let fraction_across = horizontal_fixed & 127;
            let fraction_down = vertical_limited & 127;
            let column_unclamped = x + (horizontal_fixed >> 7);
            let row_unclamped = y + (vertical_limited >> 7);

            let weight_across = if column_unclamped >= last_column || column_unclamped < 0 {
                0
            } else {
                fraction_across
            };

            let column = column_unclamped.clamp(0, last_column);
            let next_column = (column + 1).min(last_column);
            let row = row_unclamped.clamp(0, last_row);
            let next_row = (row_unclamped + 1).clamp(0, last_row);

            let top_left = sample_at(input, columns, column, row);
            let top_right = sample_at(input, columns, next_column, row);
            let bottom_left = sample_at(input, columns, column, next_row);
            let bottom_right = sample_at(input, columns, next_column, next_row);

            let top_mix = (top_left * (128 - weight_across) + top_right * weight_across + 64) >> 7;
            let bottom_mix = (bottom_left * (128 - weight_across) + bottom_right * weight_across + 64) >> 7;
            let blended = (top_mix * (128 - fraction_down) + bottom_mix * fraction_down + 64) >> 7;

            let clamped = blended.clamp(0, peak);
            output.push(clamped as u16);
        }
    }

    output
}

/// The whole aWarpSharp2 warp on the CPU, mirroring [warp_sharp](crate::warp::warp_sharp).
pub fn reference_warp_sharp(input: &[u16], width: u32, height: u32, params: &WarpParams) -> Vec<u16> {
    let mut mask = reference_edge_mask(input, width, height, params);
    for _ in 0..params.passes {
        mask = reference_blur_mask(&mask, width, height, params.blur);
    }

    reference_warp(input, &mask, width, height, params)
}

/// Rounds f32 code values on the CPU, mirroring [narrow](crate::values::narrow).
pub fn reference_narrow(values: &[f32], peak: f32) -> Vec<u16> {
    values
        .iter()
        .map(|value| {
            let rounded = (value + 0.5).floor();
            rounded.clamp(0.0, peak) as u16
        })
        .collect()
}

/// The position of `value` along the line of every f32, so neighbouring floats are one apart.
fn ordered_bits(value: f32) -> i64 {
    let bits = value.to_bits() as i32;
    let ordered = if bits < 0 { i32::MIN - bits } else { bits };

    i64::from(ordered)
}

/// Whether `found` is at most `ulps` representable f32 values from `expected`.
///
/// Motion confidence ends in an f32 division, and devices lower division differently, so its last bits
/// differ between devices and from the CPU.
pub fn within_ulps(found: f32, expected: f32, ulps: u32) -> bool {
    let found_position = ordered_bits(found);
    let expected_position = ordered_bits(expected);
    let distance = (found_position - expected_position).abs();

    distance <= i64::from(ulps)
}

/// Panics at the first sample of a `width` wide plane where `found` and `expected` differ.
pub fn assert_same_plane<T: PartialEq + Display>(found: &[T], expected: &[T], width: u32, label: &str) {
    assert_first_match(found, expected, width, label, |actual, wanted| actual == wanted);
}

/// Panics at the first sample of a `width` wide plane where `found` is further than `tolerance` from
/// `expected`.
pub fn assert_plane_within(found: &[f32], expected: &[f32], tolerance: f32, width: u32, label: &str) {
    let within = |actual: &f32, wanted: &f32| (actual - wanted).abs() <= tolerance;

    assert_first_match(found, expected, width, label, within);
}

/// How far a motion confidence may sit from another device's or the CPU's, as division rounds differently.
pub const CONFIDENCE_ULPS: u32 = 2;

/// How far a device may sit from the CPU on a compensated or degrained sample.
pub const OUTPUT_TOLERANCE: f32 = 1e-3;

/// Panics at the first value where `found` differs from `expected`, naming its field and place in it.
///
/// Every field holds `per_field` values.
pub fn assert_fields<T: PartialEq + Debug>(found: &[T], expected: &[T], per_field: usize, label: &str) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| actual != wanted);
    if let Some(index) = differing {
        let field = index / per_field;
        let within = index % per_field;
        panic!(
            "{label} differs first in field {field} at value {within}: found {:?}, expected {:?}",
            found[index], expected[index],
        );
    }
}

/// Panics at the first confidence further than [CONFIDENCE_ULPS] from `expected`.
///
/// Every field holds `per_field` confidences.
pub fn assert_confidences(found: &[f32], expected: &[f32], per_field: usize, label: &str) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong length");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| !within_ulps(*actual, *wanted, CONFIDENCE_ULPS));
    if let Some(index) = differing {
        let field = index / per_field;
        let within = index % per_field;
        panic!(
            "{label} differs first in field {field} at block {within}: found {}, expected {}",
            found[index], expected[index],
        );
    }
}

/// The largest difference between `found` and `expected`, a `width` wide plane, panicking past `tolerance`.
pub fn largest_difference(found: &[f32], expected: &[f32], tolerance: f32, width: u32, label: &str) -> f32 {
    assert_plane_within(found, expected, tolerance, width, label);

    found
        .iter()
        .zip(expected)
        .map(|(actual, wanted)| (actual - wanted).abs())
        .fold(0.0f32, f32::max)
}

/// Panics at the first sample of a `width` wide plane where `matches` rejects `found` against `expected`.
fn assert_first_match<T: Display>(
    found: &[T],
    expected: &[T],
    width: u32,
    label: &str,
    matches: impl Fn(&T, &T) -> bool,
) {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong sample count");

    let differing = found
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| !matches(actual, wanted));
    if let Some(index) = differing {
        let column = index as u32 % width;
        let row = index as u32 / width;
        panic!(
            "{label} differs first at {column},{row}: found {}, expected {}",
            found[index], expected[index],
        );
    }
}

/// Panics where the codes `found` and `expected` of a `width` wide plane differ, unless they differ by one
/// code and `merged`, the value before rounding, lies within `window` of a half code. Gives how many samples
/// differ.
pub fn assert_close_roundings(
    found: &[f32],
    expected: &[f32],
    merged: &[f32],
    window: f32,
    width: u32,
    label: &str,
) -> usize {
    assert_eq!(found.len(), expected.len(), "{label} has the wrong sample count");

    let mut differing = 0;
    for (index, (actual, wanted)) in found.iter().zip(expected).enumerate() {
        if actual == wanted {
            continue;
        }

        let column = index as u32 % width;
        let row = index as u32 / width;
        let fraction = merged[index] - merged[index].floor();
        let distance = (fraction - 0.5).abs();
        assert!(
            (actual - wanted).abs() == 1.0 && distance <= window,
            "{label} differs at {column},{row}: found {actual}, expected {wanted}, merged {}",
            merged[index],
        );
        differing += 1;
    }

    differing
}
