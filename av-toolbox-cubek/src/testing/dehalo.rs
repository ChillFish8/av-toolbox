use super::{
    reference_extreme,
    reference_median,
    reference_prewitt,
    reference_repair,
    reference_resize_float,
    reference_smooth,
};
use crate::dehalo::alpha::{AlphaParams, alpha_sizes, halo_mask_sample, strength_sample};
use crate::dehalo::fine::{FineParams, contrast_sample, outside_sample};
use crate::neighbourhood::{Expansion, Extreme, Shape, Smooth, expansion_passes};
use crate::resize::{Kernel, centred_taps};
use crate::rgvs::RepairMode;

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
