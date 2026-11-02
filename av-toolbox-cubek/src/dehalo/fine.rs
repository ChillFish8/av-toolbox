// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::alpha::AlphaParams;
use crate::{DevicePlane, Dispatch};

/// Everything `dehalo/fine` runs with.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct FineParams {
    /// The inner `dehalo/alpha`, with `rx`, `ry`, `darkstr` and `brightstr` from `dehalo/fine`.
    pub alpha: AlphaParams,
    /// Edges weaker than this are not strong edges.
    pub thmi: f32,
    /// Edges stronger than this are fully strong edges.
    pub thma: f32,
    /// Edges weaker than this are not light edges.
    pub thlimi: f32,
    /// Edges stronger than this are fully light edges.
    pub thlima: f32,
    /// How far the dehaloed plane is contrasharpened back toward the clip. 0 skips it.
    pub contra: f32,
    /// Whether light edges are kept out of the mask as well as strong ones.
    pub excl: bool,
    /// How much of the strong edges themselves the mask takes in. 0 takes none.
    pub edgeproc: f32,
}

impl FineParams {
    /// The morphology radii, `rx` and `ry` rounded half away from zero.
    pub fn radii(&self) -> (u32, u32) {
        let across = self.alpha.rx.round() as u32;
        let down = self.alpha.ry.round() as u32;

        (across, down)
    }
}

/// Adds back the part of the blur's detail the dehalo took away, where the two agree in sign.
pub fn contrast_sample(dehaloed: f32, blurred: f32, repaired: f32, clip: f32, level: f32) -> f32 {
    let detail = (blurred - repaired) * 2.49 * level;
    let removed = clip - dehaloed;

    let restored = if detail * removed < 0.0 {
        0.0
    } else if detail.abs() < removed.abs() {
        detail
    } else {
        removed
    };

    dehaloed + restored
}

/// The band around strong edges the dehalo is kept to.
pub fn outside_sample(large: f32, strong: f32, shrink: f32, excl: bool, edgeproc: f32, peak: f32) -> f32 {
    let inside = if excl { strong.max(shrink) } else { strong };
    let band = ((large - inside) * 2.0).clamp(0.0, peak);

    if edgeproc <= 0.0 {
        return band;
    }

    let widened = band + strong * edgeproc * 0.66;

    widened.clamp(0.0, peak)
}

#[cube(launch_unchecked)]
fn contrast_kernel(
    dehaloed: &[f32],
    blurred: &[f32],
    repaired: &[f32],
    clip: &[f32],
    output: &mut [f32],
    level: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let smoothed = dehaloed[ABSOLUTE_POS];
    let detail = (blurred[ABSOLUTE_POS] - repaired[ABSOLUTE_POS]) * 2.49 * level;
    let removed = clip[ABSOLUTE_POS] - smoothed;
    let detail_size = f32::abs(detail);
    let removed_size = f32::abs(removed);

    let mut restored = removed;
    if detail * removed < 0.0 {
        restored = 0.0;
    } else if detail_size < removed_size {
        restored = detail;
    }

    output[ABSOLUTE_POS] = smoothed + restored;
}

#[cube(launch_unchecked)]
fn ramp_kernel(input: &[f32], output: &mut [f32], low: f32, range: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let stretched = (input[ABSOLUTE_POS] - low) / range * 255.0;

    output[ABSOLUTE_POS] = f32::clamp(stretched, 0.0, peak);
}

#[cube(launch_unchecked)]
fn scale_kernel(input: &[f32], output: &mut [f32], factor: f32, peak: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let scaled = input[ABSOLUTE_POS] * factor;

    output[ABSOLUTE_POS] = f32::clamp(scaled, 0.0, peak);
}

#[cube(launch_unchecked)]
fn outside_kernel(
    large: &[f32],
    strong: &[f32],
    shrink: &[f32],
    output: &mut [f32],
    excl: u32,
    edgeproc: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let edge = strong[ABSOLUTE_POS];
    let mut inside = edge;
    if excl == 1u32 {
        inside = f32::max(edge, shrink[ABSOLUTE_POS]);
    }

    let doubled = (large[ABSOLUTE_POS] - inside) * 2.0;
    let mut band = f32::clamp(doubled, 0.0, peak);
    if edgeproc > 0.0 {
        let widened = band + edge * edgeproc * 0.66;
        band = f32::clamp(widened, 0.0, peak);
    }

    output[ABSOLUTE_POS] = band;
}

/// The planes one contrasharpen step reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ContrastPlanes<'a> {
    pub dehaloed: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub repaired: DevicePlane<'a>,
    pub clip: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// Adds back the part of the blur's detail the dehalo took away, where the two agree in sign.
pub fn contrasharpen(client: &Client, dispatch: &impl Dispatch, planes: ContrastPlanes<'_>, level: f32) {
    let samples = planes.output.samples();
    let dehaloed = planes.dehaloed.array();
    let blurred = planes.blurred.array();
    let repaired = planes.repaired.array();
    let clip = planes.clip.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        contrast_kernel::launch_unchecked(
            client, cubes, unit, dehaloed, blurred, repaired, clip, output, level,
        );
    }
}

/// The plane one ramp or scale step reads and the plane it writes, both f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct RampPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// Writes `clamp((x - low) / range * 255, 0, peak)` at each sample.
pub fn ramp(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RampPlanes<'_>,
    low: f32,
    range: f32,
    peak: f32,
) {
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        ramp_kernel::launch_unchecked(client, cubes, unit, input, output, low, range, peak);
    }
}

/// Writes `clamp(x * factor, 0, peak)` at each sample.
pub fn scale_values(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: RampPlanes<'_>,
    factor: f32,
    peak: f32,
) {
    let samples = planes.output.samples();
    let input = planes.input.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        scale_kernel::launch_unchecked(client, cubes, unit, input, output, factor, peak);
    }
}

/// The planes one outside-mask step reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct OutsidePlanes<'a> {
    pub large: DevicePlane<'a>,
    pub strong: DevicePlane<'a>,
    pub shrink: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// Writes the band around strong edges the dehalo is kept to.
pub fn outside_mask(
    client: &Client,
    dispatch: &impl Dispatch,
    planes: OutsidePlanes<'_>,
    excl: bool,
    edgeproc: f32,
    peak: f32,
) {
    let samples = planes.output.samples();
    let large = planes.large.array();
    let strong = planes.strong.array();
    let shrink = planes.shrink.array();
    let output = planes.output.array();
    let excl_flag = u32::from(excl);
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        outside_kernel::launch_unchecked(
            client, cubes, unit, large, strong, shrink, output, excl_flag, edgeproc, peak,
        );
    }
}
