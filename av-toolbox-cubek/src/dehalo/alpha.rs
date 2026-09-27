// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::{DevicePlane, Dispatch};

/// Everything `dehalo/alpha` runs with, on one plane of `bits`-bit samples.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct AlphaParams {
    /// How far a halo reaches across, in samples.
    pub rx: f32,
    /// How far a halo reaches down, in samples.
    pub ry: f32,
    /// How strongly dark halos are removed, at least 0.
    pub darkstr: f32,
    /// How strongly bright halos are removed, at least 0.
    pub brightstr: f32,
    /// Raising this protects more low-contrast detail.
    pub lowsens: f32,
    /// Raising this removes more high-contrast halos.
    pub highsens: f32,
    /// How far the clamp supersamples. 1 clamps with Repair mode 1 instead.
    pub ss: f32,
    /// The bits one sample holds.
    pub bits: u32,
}

impl AlphaParams {
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

/// The largest width or height `dehalo/alpha` will supersample a plane to.
pub const MAX_SUPERSAMPLED: u32 = 16384;

/// 16 below 16, otherwise `value` rounded to the nearest multiple of 4.
///
/// `None` when `value` is not finite, or when rounding it up to a multiple of 4 would overflow `u32`.
pub fn m4(value: f32) -> Option<u32> {
    if !value.is_finite() {
        return None;
    }

    if value < 16.0 {
        return Some(16);
    }

    let quarters = (value / 4.0).round() as u32;

    quarters.checked_mul(4)
}

/// The sizes `dehalo/alpha` resizes one plane through.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct AlphaSizes {
    /// The halo-free estimate's size.
    pub small: (u32, u32),
    /// The clamp's size, when it supersamples.
    pub supersampled: Option<(u32, u32)>,
}

/// The sizes for one plane of `width` by `height`, or `None` when a size cannot be computed or a
/// supersampled axis would exceed [MAX_SUPERSAMPLED].
pub fn alpha_sizes(width: u32, height: u32, params: &AlphaParams) -> Option<AlphaSizes> {
    let small_width = m4(width as f32 / params.rx)?;
    let small_height = m4(height as f32 / params.ry)?;

    let supersampled = if params.ss <= 1.0 {
        None
    } else {
        let large_width = m4(width as f32 * params.ss)?;
        let large_height = m4(height as f32 * params.ss)?;

        if large_width > MAX_SUPERSAMPLED || large_height > MAX_SUPERSAMPLED {
            return None;
        }

        Some((large_width, large_height))
    };

    Some(AlphaSizes {
        small: (small_width, small_height),
        supersampled,
    })
}

/// The halo mask's constants at this depth.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct HaloConstants {
    pub(crate) full: f32,
    pub(crate) low: f32,
    pub(crate) offset: f32,
    pub(crate) span: f32,
    pub(crate) high: f32,
    pub(crate) peak: f32,
}

impl HaloConstants {
    pub(crate) fn new(params: &AlphaParams) -> Self {
        Self {
            full: params.scale(255.0),
            low: params.scale(params.lowsens),
            offset: params.scale(256.0),
            span: params.scale(512.0),
            high: params.highsens / 100.0,
            peak: params.peak(),
        }
    }
}

/// Moves the halo-free estimate toward the clip where the clip's local range outweighs the estimate's.
///
/// `clip_range` and `halo_range` are each plane's 3x3 maximum minus its minimum.
pub fn halo_mask_sample(
    clip: f32,
    halos: f32,
    clip_range: f32,
    halo_range: f32,
    params: &AlphaParams,
) -> f32 {
    let constants = HaloConstants::new(params);
    let contrast = (clip_range - halo_range) / (clip_range + 0.000001);
    let sensitivity = contrast * constants.full - constants.low;
    let weight = (clip_range + constants.offset) / constants.span + constants.high;
    let mask = (sensitivity * weight).clamp(0.0, constants.peak);

    halos + (clip - halos) * mask / constants.peak
}

/// Pulls `clip` toward `remove`.
///
/// It moves by `darkstr` where the clip is darker than `remove`, and by `brightstr` otherwise.
pub fn strength_sample(clip: f32, remove: f32, params: &AlphaParams) -> f32 {
    let strength = if clip < remove {
        params.darkstr
    } else {
        params.brightstr
    };
    let pulled = clip - (clip - remove) * strength;

    pulled.clamp(0.0, params.peak())
}

/// The planes one halo mask blend reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct HaloPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub halos: DevicePlane<'a>,
    pub clip_max: DevicePlane<'a>,
    pub clip_min: DevicePlane<'a>,
    pub halo_max: DevicePlane<'a>,
    pub halo_min: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn halo_mask_kernel(
    clip: &Array<f32>,
    halos: &Array<f32>,
    clip_max: &Array<f32>,
    clip_min: &Array<f32>,
    halo_max: &Array<f32>,
    halo_min: &Array<f32>,
    output: &mut Array<f32>,
    full: f32,
    low: f32,
    offset: f32,
    span: f32,
    high: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let clip_range = clip_max[ABSOLUTE_POS] - clip_min[ABSOLUTE_POS];
    let halo_range = halo_max[ABSOLUTE_POS] - halo_min[ABSOLUTE_POS];
    let contrast = (clip_range - halo_range) / (clip_range + 0.000001);
    let sensitivity = contrast * full - low;
    let weight = (clip_range + offset) / span + high;
    let mask = f32::clamp(sensitivity * weight, 0.0, peak);
    let estimate = halos[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = estimate + (clip[ABSOLUTE_POS] - estimate) * mask / peak;
}

/// Writes `lets`, the halo-free estimate moved toward the clip where the clip holds real detail.
pub fn blend_halo_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: HaloPlanes<'_>,
    params: &AlphaParams,
) {
    let constants = HaloConstants::new(params);
    let samples = planes.output.samples();
    let clip = planes.clip.array::<R>();
    let halos = planes.halos.array::<R>();
    let clip_max = planes.clip_max.array::<R>();
    let clip_min = planes.clip_min.array::<R>();
    let halo_max = planes.halo_max.array::<R>();
    let halo_min = planes.halo_min.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        halo_mask_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            clip,
            halos,
            clip_max,
            clip_min,
            halo_max,
            halo_min,
            output,
            constants.full,
            constants.low,
            constants.offset,
            constants.span,
            constants.high,
            constants.peak,
        );
    }
}

/// The planes one clamp reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ClampPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub highest: DevicePlane<'a>,
    pub lowest: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn clamp_kernel(clip: &Array<f32>, highest: &Array<f32>, lowest: &Array<f32>, output: &mut Array<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let capped = f32::min(clip[ABSOLUTE_POS], highest[ABSOLUTE_POS]);

    output[ABSOLUTE_POS] = f32::max(capped, lowest[ABSOLUTE_POS]);
}

/// Writes `max(min(clip, highest), lowest)` at each sample.
pub fn clamp_between<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ClampPlanes<'_>,
) {
    let samples = planes.output.samples();
    let clip = planes.clip.array::<R>();
    let highest = planes.highest.array::<R>();
    let lowest = planes.lowest.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        clamp_kernel::launch_unchecked::<R>(client, cubes, unit, clip, highest, lowest, output);
    }
}

/// The planes one strength step reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct StrengthPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub remove: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn strength_kernel(
    clip: &Array<f32>,
    remove: &Array<f32>,
    output: &mut Array<f32>,
    darkstr: f32,
    brightstr: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = clip[ABSOLUTE_POS];
    let target = remove[ABSOLUTE_POS];

    let mut strength = brightstr;
    if value < target {
        strength = darkstr;
    }

    let pulled = value - (value - target) * strength;

    output[ABSOLUTE_POS] = f32::clamp(pulled, 0.0, peak);
}

/// Pulls the clip toward `remove` by `darkstr` or `brightstr`, clamped to `0..=peak`.
pub fn apply_strength<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: StrengthPlanes<'_>,
    params: &AlphaParams,
) {
    let peak = params.peak();
    let samples = planes.output.samples();
    let clip = planes.clip.array::<R>();
    let remove = planes.remove.array::<R>();
    let output = planes.output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        strength_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            clip,
            remove,
            output,
            params.darkstr,
            params.brightstr,
            peak,
        );
    }
}
