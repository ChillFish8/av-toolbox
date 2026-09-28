use super::dehalo::subtract;
use super::{
    reference_extreme,
    reference_hysteresis,
    reference_median,
    reference_merge_masked,
    reference_min_blur,
    reference_prewitt,
    reference_repair,
    reference_smooth,
    reference_spread,
};
use crate::dehalo::dering::{DeringParams, contra_limit_sample, limit_sample, ring_mask_sample};
use crate::dehalo::edge_clean::threshold_sample;
use crate::neighbourhood::{Extreme, Shape, Smooth, Spread};
use crate::rgvs::RepairMode;

/// The inner ring mask for `minp` 1, inflate then square minimum.
fn inner_minp_one(fmask: &[f32], width: u32, height: u32) -> Vec<f32> {
    let inflated = reference_spread(fmask, width, height, Spread::Inflate);

    reference_extreme(&inflated, width, height, Extreme::Minimum, Shape::Square)
}

/// The inner ring mask for `minp` 3, inflate then square minimum twice.
fn inner_minp_three(fmask: &[f32], width: u32, height: u32) -> Vec<f32> {
    let inflated = reference_spread(fmask, width, height, Spread::Inflate);
    let once = reference_extreme(&inflated, width, height, Extreme::Minimum, Shape::Square);

    reference_extreme(&once, width, height, Extreme::Minimum, Shape::Square)
}

/// The inner ring mask for `minp` 4, square minimum twice.
fn inner_minp_four(fmask: &[f32], width: u32, height: u32) -> Vec<f32> {
    let once = reference_extreme(fmask, width, height, Extreme::Minimum, Shape::Square);

    reference_extreme(&once, width, height, Extreme::Minimum, Shape::Square)
}

/// The ring mask's inner edge for `minp`.
fn inner_mask(fmask: &[f32], width: u32, height: u32, minp: u32) -> Vec<f32> {
    debug_assert!((0..=4).contains(&minp));

    match minp {
        0 => fmask.to_vec(),
        1 => inner_minp_one(fmask, width, height),
        2 => reference_extreme(fmask, width, height, Extreme::Minimum, Shape::Square),
        3 => inner_minp_three(fmask, width, height),
        4 => inner_minp_four(fmask, width, height),
        _ => unreachable!("minp is 0, 1, 2, 3 or 4"),
    }
}

/// The ring mask's outer edge, `mrad` square maximum passes then `msmooth` inflate passes.
fn outer_mask(fmask: &[f32], width: u32, height: u32, mrad: u32, msmooth: u32) -> Vec<f32> {
    let mut outer = fmask.to_vec();
    for _ in 0..mrad {
        outer = reference_extreme(&outer, width, height, Extreme::Maximum, Shape::Square);
    }

    for _ in 0..msmooth {
        outer = reference_spread(&outer, width, height, Spread::Inflate);
    }

    outer
}

/// Contrasharpens `smoothed` against `input`, `sharp` passes deep.
fn contra_sharpen(input: &[f32], smoothed: &[f32], width: u32, height: u32, sharp: u32) -> Vec<f32> {
    let pre = reference_median(smoothed, width, height, 1);
    let mut method = reference_smooth(&pre, width, height, Smooth::Binomial);
    for _ in 1..sharp {
        method = reference_smooth(&method, width, height, Smooth::Box);
    }

    let sharpened = subtract(&pre, &method);
    let removed = subtract(input, smoothed);
    let repaired = reference_repair(&sharpened, &removed, width, height, RepairMode::One);

    (0..input.len())
        .map(|index| contra_limit_sample(smoothed[index], sharpened[index], repaired[index]))
        .collect()
}

/// The whole of havsfunc's HQDeringmod on one f32 plane on the CPU.
pub fn reference_hq_dering(input: &[f32], width: u32, height: u32, params: &DeringParams) -> Vec<f32> {
    let peak = params.peak();
    let min_blurred = reference_min_blur(input, width, height, params.nrmode);
    let mut smoothed = min_blurred;

    if params.sharp > 0 {
        smoothed = contra_sharpen(input, &smoothed, width, height, params.sharp);
    }

    let repaired = match params.drrep {
        Some(mode) => reference_repair(input, &smoothed, width, height, mode),
        None => smoothed,
    };
    let thr = params.scale(params.thr);
    let darkthr = params.scale(params.darkthr);
    let limited: Vec<f32> = (0..input.len())
        .map(|index| limit_sample(input[index], repaired[index], thr, darkthr, params.elast))
        .collect();

    let edges = reference_prewitt(input, width, height, peak);
    let floor = params.scale(params.mthr);
    let pm: Vec<f32> = edges
        .iter()
        .map(|edge| threshold_sample(*edge, floor, f32::MAX, peak, false))
        .collect();
    let pm_median = reference_median(&pm, width, height, 1);
    let fmask = reference_hysteresis(&pm_median, &pm, width, height, peak);

    let outer = outer_mask(&fmask, width, height, params.mrad, params.msmooth);
    let inner = inner_mask(&fmask, width, height, params.minp);
    let ringmask: Vec<f32> = (0..input.len())
        .map(|index| ring_mask_sample(outer[index], inner[index], params.incedge, peak))
        .collect();

    reference_merge_masked(input, &limited, &ringmask, peak)
}
