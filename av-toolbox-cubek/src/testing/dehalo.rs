use super::{
    reference_crop,
    reference_extreme,
    reference_median,
    reference_min_blur,
    reference_narrow,
    reference_pad,
    reference_prewitt,
    reference_remove_grain,
    reference_repair,
    reference_resize_float,
    reference_smooth,
    reference_warp_sharp,
};
use crate::dehalo::PADDING;
use crate::dehalo::alpha::{AlphaParams, alpha_sizes, halo_mask_sample, strength_sample};
use crate::dehalo::edge_clean::{EdgeCleanParams, levels_sample, threshold_sample};
use crate::dehalo::fine::{FineParams, contrast_sample, outside_sample};
use crate::dehalo::yahr::YahrParams;
use crate::neighbourhood::{Expansion, Extreme, Shape, Smooth, expansion_passes};
use crate::resize::{Kernel, centred_taps};
use crate::rgvs::{RemoveGrainMode, RepairMode};
use crate::warp::WarpParams;

const MITCHELL: Kernel = Kernel::Bicubic {
    b: 1.0 / 3.0,
    c: 1.0 / 3.0,
};

const SOFT_CUBIC: Kernel = Kernel::Bicubic { b: 1.0, c: 0.0 };

const LANCZOS: Kernel = Kernel::Lanczos { taps: 3 };

/// Merges on the CPU, mirroring [merge_masked](crate::dehalo::merge_masked).
pub fn reference_merge_masked(clip: &[f32], filtered: &[f32], mask: &[f32], peak: f32) -> Vec<f32> {
    clip.iter()
        .zip(filtered)
        .zip(mask)
        .map(|((value, target), weight)| value + (target - value) * weight / peak)
        .collect()
}

/// Resizes an f32 plane from `from` to `to` with centred taps of `kernel`.
fn resized(input: &[f32], from: (u32, u32), to: (u32, u32), kernel: &Kernel) -> Vec<f32> {
    let across = centred_taps(from.0, to.0, kernel);
    let down = centred_taps(from.1, to.1, kernel);

    reference_resize_float(input, from.0, from.1, &across, &down)
}

/// The whole of `dehalo/alpha` on one f32 plane on the CPU.
pub fn reference_dehalo_alpha(input: &[f32], width: u32, height: u32, params: &AlphaParams) -> Vec<f32> {
    let sizes = alpha_sizes(width, height, params).expect("a valid dehalo/alpha size");
    let full = (width, height);

    let small = resized(input, full, sizes.small, &MITCHELL);
    let halos = resized(&small, sizes.small, full, &SOFT_CUBIC);

    let clip_max = reference_extreme(input, width, height, Extreme::Maximum, Shape::Square);
    let clip_min = reference_extreme(input, width, height, Extreme::Minimum, Shape::Square);
    let halo_max = reference_extreme(&halos, width, height, Extreme::Maximum, Shape::Square);
    let halo_min = reference_extreme(&halos, width, height, Extreme::Minimum, Shape::Square);

    let lets: Vec<f32> = (0..input.len())
        .map(|index| {
            let clip_range = clip_max[index] - clip_min[index];
            let halo_range = halo_max[index] - halo_min[index];
            halo_mask_sample(input[index], halos[index], clip_range, halo_range, params)
        })
        .collect();

    let remove = match sizes.supersampled {
        None => reference_repair(input, &lets, width, height, RepairMode::One),
        Some(large) => {
            let lets_max = reference_extreme(&lets, width, height, Extreme::Maximum, Shape::Square);
            let lets_min = reference_extreme(&lets, width, height, Extreme::Minimum, Shape::Square);
            let clip_up = resized(input, full, large, &LANCZOS);
            let highest = resized(&lets_max, full, large, &MITCHELL);
            let lowest = resized(&lets_min, full, large, &MITCHELL);
            let clamped: Vec<f32> = (0..clip_up.len())
                .map(|index| clip_up[index].min(highest[index]).max(lowest[index]))
                .collect();

            resized(&clamped, large, full, &LANCZOS)
        },
    };

    input
        .iter()
        .zip(&remove)
        .map(|(clip, target)| strength_sample(*clip, *target, params))
        .collect()
}

/// Runs a chain of maximum or minimum passes on the CPU.
fn expanded(input: &[f32], width: u32, height: u32, passes: &[Shape], extreme: Extreme) -> Vec<f32> {
    let mut plane = input.to_vec();
    for shape in passes {
        plane = reference_extreme(&plane, width, height, extreme, *shape);
    }

    plane
}

/// The whole of `dehalo/fine` on one f32 plane on the CPU.
pub fn reference_fine_dehalo(input: &[f32], width: u32, height: u32, params: &FineParams) -> Vec<f32> {
    let peak = params.alpha.peak();
    let (across, down) = params.radii();

    let mut dehaloed = reference_dehalo_alpha(input, width, height, &params.alpha);
    if params.contra > 0.0 {
        let blurred = reference_smooth(&dehaloed, width, height, Smooth::Binomial);
        let median = reference_median(&blurred, width, height, 2);
        let first = reference_repair(&blurred, &median, width, height, RepairMode::One);
        let repaired = reference_repair(&blurred, &first, width, height, RepairMode::One);
        dehaloed = (0..input.len())
            .map(|index| {
                let smoothed = dehaloed[index];
                let blurred_value = blurred[index];
                let repaired_value = repaired[index];
                let clip = input[index];

                contrast_sample(smoothed, blurred_value, repaired_value, clip, params.contra)
            })
            .collect();
    }

    let edges = reference_prewitt(input, width, height, peak);
    let strong_low = params.alpha.scale(params.thmi);
    let light_low = params.alpha.scale(params.thlimi);
    let strong: Vec<f32> = edges
        .iter()
        .map(|edge| ((edge - strong_low) / (params.thma - params.thmi) * 255.0).clamp(0.0, peak))
        .collect();
    let light: Vec<f32> = edges
        .iter()
        .map(|edge| ((edge - light_low) / (params.thlima - params.thlimi) * 255.0).clamp(0.0, peak))
        .collect();

    let rectangle = expansion_passes(across, down, Expansion::Rectangle);
    let ellipse = expansion_passes(across, down, Expansion::Ellipse);
    let large = expanded(&strong, width, height, &rectangle, Extreme::Maximum);

    let grown = expanded(&light, width, height, &ellipse, Extreme::Maximum);
    let brightened: Vec<f32> = grown.iter().map(|value| (value * 4.0).clamp(0.0, peak)).collect();
    let thinned = expanded(&brightened, width, height, &ellipse, Extreme::Minimum);
    let once = reference_smooth(&thinned, width, height, Smooth::Box);
    let shrink = reference_smooth(&once, width, height, Smooth::Box);

    let band: Vec<f32> = (0..input.len())
        .map(|index| {
            let strong_value = strong[index];
            let large_value = large[index];
            let shrink_value = shrink[index];

            outside_sample(
                large_value,
                strong_value,
                shrink_value,
                params.excl,
                params.edgeproc,
                peak,
            )
        })
        .collect();
    let blurred_band = reference_smooth(&band, width, height, Smooth::Box);
    let outside: Vec<f32> = blurred_band
        .iter()
        .map(|value| (value * 2.0).clamp(0.0, peak))
        .collect();

    reference_merge_masked(input, &dehaloed, &outside, peak)
}

/// Zips two planes and subtracts, sample by sample.
pub(super) fn subtract(first: &[f32], second: &[f32]) -> Vec<f32> {
    first
        .iter()
        .zip(second)
        .map(|(value, other)| value - other)
        .collect()
}

/// Runs [reference_warp_sharp] on a plane grown by [PADDING] on every side, on the CPU.
///
/// The plane is rounded to `u16` before the warp and back to `f32` after, mirroring how
/// aWarpSharp2's own mask and warp work in integer samples.
pub fn reference_padded_warp(input: &[f32], width: u32, height: u32, params: &WarpParams) -> Vec<f32> {
    let padded = reference_pad(input, width, height, PADDING);
    let padded_width = width + 2 * PADDING;
    let padded_height = height + 2 * PADDING;
    let peak = params.peak() as f32;

    let narrowed = reference_narrow(&padded, peak);
    let warped = reference_warp_sharp(&narrowed, padded_width, padded_height, params);
    let widened: Vec<f32> = warped.iter().map(|value| f32::from(*value)).collect();

    reference_crop(&widened, padded_width, padded_height, PADDING)
}

/// The whole of havsfunc's YAHR on one f32 plane on the CPU.
pub fn reference_yahr(input: &[f32], width: u32, height: u32, params: &YahrParams) -> Vec<f32> {
    let min_blurred = reference_min_blur(input, width, height, 2);
    let blurred = reference_smooth(&min_blurred, width, height, Smooth::Binomial);
    let detail = subtract(input, &blurred);

    let warp_params = params.warp_params();
    let warped = reference_padded_warp(input, width, height, &warp_params);

    let warped_min_blurred = reference_min_blur(&warped, width, height, 2);
    let warped_blurred = reference_smooth(&warped_min_blurred, width, height, Smooth::Binomial);
    let warped_detail = subtract(&warped, &warped_blurred);

    let limited = reference_repair(&detail, &warped_detail, width, height, RepairMode::Thirteen);
    let change = subtract(&detail, &limited);

    subtract(input, &change)
}

/// The whole of havsfunc's EdgeCleaner on one f32 plane on the CPU.
pub fn reference_edge_clean(input: &[f32], width: u32, height: u32, params: &EdgeCleanParams) -> Vec<f32> {
    let peak = params.peak();
    let warp_params = params.warp_params();
    let warped = reference_padded_warp(input, width, height, &warp_params);
    let main = if params.rep {
        reference_repair(&warped, input, width, height, params.rmode)
    } else {
        warped
    };

    let edges = reference_prewitt(input, width, height, peak);
    let low = params.scale(4.0);
    let high = params.scale(32.0);
    let thresholded: Vec<f32> = edges
        .iter()
        .map(|edge| threshold_sample(*edge, low, high, peak, true))
        .collect();
    let mask = reference_smooth(&thresholded, width, height, Smooth::Box);
    let mut result = reference_merge_masked(input, &main, &mask, peak);

    if params.hot {
        result = reference_repair(&result, input, width, height, RepairMode::Two);
    }

    if params.smode {
        let smoothed = reference_remove_grain(input, width, height, RemoveGrainMode::Seventeen);
        let neutral = params.scale(128.0);
        let diff: Vec<f32> = input
            .iter()
            .zip(&smoothed)
            .map(|(value, other)| value - other + neutral)
            .collect();
        let levels_low = params.scale(40.0);
        let levels_span = params.scale(128.0);
        let exponent = 1.0 / 0.35;
        let brightened: Vec<f32> = diff
            .iter()
            .map(|value| levels_sample(*value, levels_low, levels_span, exponent, peak))
            .collect();
        let cleaned = reference_remove_grain(&brightened, width, height, RemoveGrainMode::Seven);
        let particle_edges = reference_prewitt(&cleaned, width, height, peak);
        let particle_high = params.scale(16.0);
        let particles: Vec<f32> = particle_edges
            .iter()
            .map(|edge| threshold_sample(*edge, low, particle_high, peak, false))
            .collect();
        result = reference_merge_masked(&result, input, &particles, peak);
    }

    result
}
