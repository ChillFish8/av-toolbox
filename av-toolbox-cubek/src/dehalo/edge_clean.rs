// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::DifferencePlanes;
use super::fine::RampPlanes;
use crate::Dispatch;
use crate::depth::Element;
use crate::rgvs::RepairMode;
use crate::warp::{BlurKind, WarpParams};

/// Everything havsfunc's EdgeCleaner runs with.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct EdgeCleanParams {
    /// How far samples move in the warp.
    pub strength: u32,
    /// Whether the warp is repaired against the clip before merging.
    pub rep: bool,
    /// The repair mode `rep` runs, when set.
    pub rmode: RepairMode,
    /// Whether the particle-cleaning pass runs.
    pub smode: bool,
    /// Whether the merged result is repaired against hot pixels.
    pub hot: bool,
    /// How the plane's samples are stored.
    pub element: Element,
    /// The bits one sample holds.
    pub bits: u32,
}

impl EdgeCleanParams {
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

    /// The warp EdgeCleaner runs, with a depth of half `strength`, plus 4 in `smode`, rounded half
    /// up.
    pub fn warp_params(&self) -> WarpParams {
        let smode_bonus = 4 * u32::from(self.smode);
        let depth = (self.strength + smode_bonus).div_ceil(2);

        WarpParams {
            element: self.element,
            bits: self.bits,
            thresh: 128,
            blur: BlurKind::R6,
            passes: 1,
            depth: depth as i32,
        }
    }
}

/// Zeroes edges weaker than `low`, fills edges stronger than `high`, and passes the rest through,
/// then flips around `peak` when `invert` is set.
pub fn threshold_sample(value: f32, low: f32, high: f32, peak: f32, invert: bool) -> f32 {
    let clamped = if value < low {
        0.0
    } else if value > high {
        peak
    } else {
        value
    };

    if invert { peak - clamped } else { clamped }
}

/// Stretches `value` from the window `low..low + span` to `0..1`, curves it by `exponent`, then
/// scales it back up to `peak`.
pub fn levels_sample(value: f32, low: f32, span: f32, exponent: f32, peak: f32) -> f32 {
    let stretched = ((value - low) / span).clamp(0.0, 1.0);
    let curved = f32::powf(stretched, exponent);

    curved * peak
}

#[cube(launch_unchecked)]
fn threshold_kernel(input: &[f32], output: &mut [f32], low: f32, high: f32, peak: f32, invert: u32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let mut clamped = value;
    if value < low {
        clamped = 0.0;
    } else if value > high {
        clamped = peak;
    }

    let mut result = clamped;
    if invert == 1u32 {
        result = peak - clamped;
    }

    output[ABSOLUTE_POS] = result;
}

#[cube(launch_unchecked)]
fn levels_kernel(input: &[f32], output: &mut [f32], low: f32, span: f32, exponent: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let stretched = f32::clamp((input[ABSOLUTE_POS] - low) / span, 0.0, 1.0);
    let curved = f32::powf(stretched, exponent);

    output[ABSOLUTE_POS] = curved * peak;
}

#[cube(launch_unchecked)]
fn offset_difference_kernel(first: &[f32], second: &[f32], output: &mut [f32], offset: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    output[ABSOLUTE_POS] = first[ABSOLUTE_POS] - second[ABSOLUTE_POS] + offset;
}

/// Writes `threshold_sample` at each sample.
pub fn threshold_mask(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RampPlanes<'_>,
    low: f32,
    high: f32,
    peak: f32,
    invert: bool,
) {
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let invert_flag = u32::from(invert);
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        threshold_kernel::launch_unchecked(client, cubes, unit, input, output, low, high, peak, invert_flag);
    }
}

/// Writes `levels_sample` at each sample.
pub fn levels(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RampPlanes<'_>,
    low: f32,
    span: f32,
    exponent: f32,
    peak: f32,
) {
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        levels_kernel::launch_unchecked(client, cubes, unit, input, output, low, span, exponent, peak);
    }
}

/// Writes `first - second + offset` at each sample.
pub fn offset_difference(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: DifferencePlanes<'_>,
    offset: f32,
) {
    let samples = planes.output.samples();
    let first = planes.first.array();
    let second = planes.second.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        offset_difference_kernel::launch_unchecked(client, cubes, unit, first, second, output, offset);
    }
}
