// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::rgvs::RepairMode;
use crate::{DevicePlane, Dispatch};

/// Everything havsfunc's HQDeringmod runs with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DeringParams {
    /// How many square maximum passes grow the ring mask's outer edge.
    pub mrad: u32,
    /// How many inflate passes soften the ring mask's outer edge, after `mrad`.
    pub msmooth: u32,
    /// Whether the ring mask keeps the edge itself, rather than cutting it out with `minp`.
    pub incedge: bool,
    /// Edges weaker than this, once scaled, are not part of the ring mask.
    pub mthr: f32,
    /// Which combination of inflate and square minimum passes shrinks the ring mask's inner edge.
    pub minp: u32,
    /// The MinBlur radius the smoothing pass runs.
    pub nrmode: u32,
    /// How many contrasharpening passes run. 0 skips contrasharpening.
    pub sharp: u32,
    /// The repair mode the smoothed plane is repaired against the clip with, when set.
    pub drrep: Option<RepairMode>,
    /// How far a darkening sample may move before it is limited back toward the plane.
    pub thr: f32,
    /// How far a brightening sample may move before it is limited back toward the plane.
    pub darkthr: f32,
    /// How far past `thr`/`darkthr` a sample blends rather than falling straight back.
    pub elast: f32,
    /// The bits one sample holds.
    pub bits: u32,
}

impl DeringParams {
    pub fn peak(&self) -> f32 {
        let levels = 1u32 << self.bits;

        (levels - 1) as f32
    }

    /// `value` on the 8-bit scale, moved to this depth.
    pub fn scale(&self, value: f32) -> f32 {
        let shift = self.bits - 8;
        let factor = (1u32 << shift) as f32;

        value * factor
    }
}

/// Moves `smoothed` by whichever of `sharpened` or `repaired` changes it least.
pub fn contra_limit_sample(smoothed: f32, sharpened: f32, repaired: f32) -> f32 {
    let chosen = if repaired.abs() > sharpened.abs() {
        sharpened
    } else {
        repaired
    };

    smoothed + chosen
}

/// Limits `filtered` back toward `plane` once it moves past `thr`/`darkthr`, blending between them
/// and `elast` times them.
pub fn limit_sample(plane: f32, filtered: f32, thr: f32, darkthr: f32, elast: f32) -> f32 {
    let difference = filtered - plane;
    let threshold = if difference > 0.0 { darkthr } else { thr };
    let size = difference.abs();
    let elastic_threshold = threshold * elast;

    if size <= threshold {
        filtered
    } else if size >= elastic_threshold {
        plane
    } else {
        plane + difference * (elastic_threshold - size) / (elastic_threshold - threshold)
    }
}

/// Cuts `inner` out of `outer`, or passes `outer` through unchanged when `incedge` is set.
pub fn ring_mask_sample(outer: f32, inner: f32, incedge: bool, peak: f32) -> f32 {
    if incedge {
        outer
    } else {
        outer * (peak - inner) / peak
    }
}

/// The planes one contra-limit reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ContraPlanes<'a> {
    pub smoothed: DevicePlane<'a>,
    pub sharpened: DevicePlane<'a>,
    pub repaired: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The planes one limit-filter reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct LimitPlanes<'a> {
    pub plane: DevicePlane<'a>,
    pub filtered: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The planes one ring-mask reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RingPlanes<'a> {
    pub outer: DevicePlane<'a>,
    pub inner: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn contra_limit_kernel(smoothed: &[f32], sharpened: &[f32], repaired: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let sharp_value = sharpened[ABSOLUTE_POS];
    let repair_value = repaired[ABSOLUTE_POS];

    let mut chosen = repair_value;
    if f32::abs(repair_value) > f32::abs(sharp_value) {
        chosen = sharp_value;
    }

    output[ABSOLUTE_POS] = smoothed[ABSOLUTE_POS] + chosen;
}

#[cube(launch_unchecked)]
fn limit_kernel(plane: &[f32], filtered: &[f32], output: &mut [f32], thr: f32, darkthr: f32, elast: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let plane_value = plane[ABSOLUTE_POS];
    let filtered_value = filtered[ABSOLUTE_POS];
    let difference = filtered_value - plane_value;

    let mut threshold = thr;
    if difference > 0.0 {
        threshold = darkthr;
    }

    let size = f32::abs(difference);
    let elastic_threshold = threshold * elast;

    let mut result = filtered_value;
    if size > threshold {
        result = plane_value;
        if size < elastic_threshold {
            result = plane_value + difference * (elastic_threshold - size) / (elastic_threshold - threshold);
        }
    }

    output[ABSOLUTE_POS] = result;
}

#[cube(launch_unchecked)]
fn ring_mask_kernel(outer: &[f32], inner: &[f32], output: &mut [f32], incedge: u32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let outer_value = outer[ABSOLUTE_POS];
    let inner_value = inner[ABSOLUTE_POS];

    let mut result = outer_value * (peak - inner_value) / peak;
    if incedge == 1u32 {
        result = outer_value;
    }

    output[ABSOLUTE_POS] = result;
}

/// Writes `contra_limit_sample` at each sample.
pub fn contra_limit(client: &Client, dispatch: &impl Dispatch, planes: ContraPlanes<'_>) {
    let samples = planes.output.samples();
    let smoothed = planes.smoothed.array();
    let sharpened = planes.sharpened.array();
    let repaired = planes.repaired.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        contra_limit_kernel::launch_unchecked(client, cubes, unit, smoothed, sharpened, repaired, output);
    }
}

/// Writes `limit_sample` at each sample.
pub fn limit_filter(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: LimitPlanes<'_>,
    thr: f32,
    darkthr: f32,
    elast: f32,
) {
    let samples = planes.output.samples();
    let plane = planes.plane.array();
    let filtered = planes.filtered.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        limit_kernel::launch_unchecked(client, cubes, unit, plane, filtered, output, thr, darkthr, elast);
    }
}

/// Writes `ring_mask_sample` at each sample.
pub fn ring_mask(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RingPlanes<'_>,
    incedge: bool,
    peak: f32,
) {
    let samples = planes.output.samples();
    let outer = planes.outer.array();
    let inner = planes.inner.array();
    let output = planes.output.array();
    let incedge_flag = u32::from(incedge);
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        ring_mask_kernel::launch_unchecked(client, cubes, unit, outer, inner, output, incedge_flag, peak);
    }
}
