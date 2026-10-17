#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::blur::{self, FilterPlanes, Storage};
use crate::dehalo::{DifferencePlanes, difference};
use crate::neighbourhood::{Extreme, NeighbourPlanes, Shape, Smooth, extreme, smooth};
use crate::warp::WarpParams;
use crate::{DevicePlane, Dispatch, Edge};

/// The planes one merge reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct MergePlanes<'a> {
    /// The unfiltered plane.
    pub input: DevicePlane<'a>,
    /// The filtered plane, overwritten with the merge.
    pub filtered: DevicePlane<'a>,
    /// The weight of the filtered plane at each sample, between 0 and `weight_peak`.
    pub weight: DevicePlane<'a>,
}

/// Mixes the two planes as `input * (1 - share) + filtered * share`, with `share = weight / weight_peak`.
///
/// A weight of 0 or `weight_peak` gives a share of exactly 0 or 1, so those weights hand back the
/// input or the filtered value exactly.
#[cube(launch_unchecked)]
fn merge_kernel(input: &Array<f32>, filtered: &mut Array<f32>, weight: &Array<f32>, weight_peak: f32) {
    if ABSOLUTE_POS >= filtered.len() {
        terminate!();
    }

    let share = weight[ABSOLUTE_POS] / weight_peak;
    let kept = input[ABSOLUTE_POS] * (1.0 - share);
    let taken = filtered[ABSOLUTE_POS] * share;

    filtered[ABSOLUTE_POS] = kept + taken;
}

/// Holds `filtered` to `0..=peak`, then mixes the two planes as `input * (1 - share) + filtered * share`,
/// with `share = weight / weight_peak`.
#[cube(launch_unchecked)]
fn clamped_merge_kernel(
    input: &Array<f32>,
    filtered: &mut Array<f32>,
    weight: &Array<f32>,
    weight_peak: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= filtered.len() {
        terminate!();
    }

    let share = weight[ABSOLUTE_POS] / weight_peak;
    let capped = f32::min(filtered[ABSOLUTE_POS], peak);
    let target = f32::max(capped, 0.0);
    let kept = input[ABSOLUTE_POS] * (1.0 - share);
    let taken = target * share;

    filtered[ABSOLUTE_POS] = kept + taken;
}

/// Writes `input + (filtered - input) * weight / weight_peak` into `filtered`.
pub fn merge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MergePlanes<'_>,
    weight_peak: f32,
) {
    let samples = planes.filtered.samples();
    let input = planes.input.array::<R>();
    let filtered = planes.filtered.array::<R>();
    let weight = planes.weight.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        merge_kernel::launch_unchecked::<R>(client, cubes, unit, input, filtered, weight, weight_peak);
    }
}

/// Writes `input + (clamp(filtered, 0, peak) - input) * weight / weight_peak` into `filtered`.
///
/// Holding `filtered` to the sample range first means a partial weight lands between the input and
/// the filtered plane as it would be stored.
pub fn clamped_merge<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: MergePlanes<'_>,
    weight_peak: f32,
    peak: f32,
) {
    let samples = planes.filtered.samples();
    let input = planes.input.array::<R>();
    let filtered = planes.filtered.array::<R>();
    let weight = planes.weight.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        clamped_merge_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            filtered,
            weight,
            weight_peak,
            peak,
        );
    }
}

/// The vertical `1 2 1` blur over 4, reading past the top and bottom through `reflect`.
pub fn vertical_121<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: NeighbourPlanes<'_>,
) {
    let filter_planes = FilterPlanes {
        input: planes.input,
        output: planes.output,
    };

    blur::vertical_121(client, dispatch, filter_planes, Edge::Reflect, Storage::F32);
}

/// The settings of one vinverse pass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VinverseParams {
    /// How strongly the remaining combing is countered.
    pub contra_str: f32,
    /// How much of a correction that fights the input is kept, between 0 and 1.
    pub scl: f32,
    /// The most a sample may move from the input, in code values. `None` leaves it unlimited.
    pub amnt: Option<f32>,
}

/// The planes one vinverse pass reads, writes and uses as scratch, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct VinversePlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    pub blurred: DevicePlane<'a>,
    pub diff: DevicePlane<'a>,
    pub diff_blurred: DevicePlane<'a>,
    pub sbr: DevicePlane<'a>,
}

/// Takes off the part of the comb a second blur removes, unless it points against the comb.
#[cube(launch_unchecked)]
fn sbr_kernel(input: &Array<f32>, diff: &Array<f32>, diff_blurred: &Array<f32>, sbr: &mut Array<f32>) {
    if ABSOLUTE_POS >= sbr.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let comb = diff[ABSOLUTE_POS];
    let residual = comb - diff_blurred[ABSOLUTE_POS];

    let mut removed = comb;
    if f32::abs(residual) < f32::abs(comb) {
        removed = residual;
    }

    let mut result = value - removed;
    if residual * comb < 0.0 {
        result = value;
    }

    sbr[ABSOLUTE_POS] = result;
}

/// Adds the contra-sharpening step onto `sbr` and limits the result to `limit` around `input`.
#[cube(launch_unchecked)]
fn vinverse_kernel(
    input: &Array<f32>,
    sbr: &Array<f32>,
    sbr_blurred: &Array<f32>,
    output: &mut Array<f32>,
    contra_str: f32,
    scl: f32,
    limit: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let base = sbr[ABSOLUTE_POS];
    let contra = (base - sbr_blurred[ABSOLUTE_POS]) * contra_str;
    let removed = value - base;

    let mut smaller = removed;
    if f32::abs(contra) < f32::abs(removed) {
        smaller = contra;
    }

    let mut step = smaller;
    if contra * removed < 0.0 {
        step = smaller * scl;
    }

    let result = base + step;
    let capped = f32::min(result, value + limit);

    output[ABSOLUTE_POS] = f32::max(capped, value - limit);
}

/// Removes residual combing from `input` the way vs-jetpack's vinverse does, writing `output`.
///
/// The comb is the part of `input` that a vertical `1 2 1` blur removes. The part of that comb a
/// second blur removes is taken off, unless it points against the comb. A contra-sharpening step
/// then restores detail the first step flattened.
pub fn vinverse<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: VinversePlanes<'_>,
    params: &VinverseParams,
) {
    let blur_input = NeighbourPlanes {
        input: planes.input,
        output: planes.blurred,
    };
    vertical_121(client, dispatch, blur_input);

    let comb = DifferencePlanes {
        first: planes.input,
        second: planes.blurred,
        output: planes.diff,
    };
    difference(client, dispatch, comb);

    let blur_comb = NeighbourPlanes {
        input: planes.diff,
        output: planes.diff_blurred,
    };
    vertical_121(client, dispatch, blur_comb);

    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let input = planes.input.array::<R>();
    let diff = planes.diff.array::<R>();
    let diff_blurred = planes.diff_blurred.array::<R>();
    let sbr = planes.sbr.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        sbr_kernel::launch_unchecked::<R>(client, cubes, unit, input, diff, diff_blurred, sbr);
    }

    let blur_sbr = NeighbourPlanes {
        input: planes.sbr,
        output: planes.blurred,
    };
    vertical_121(client, dispatch, blur_sbr);

    let cubes = dispatch.cube_count(samples);
    let input = planes.input.array::<R>();
    let sbr = planes.sbr.array::<R>();
    let sbr_blurred = planes.blurred.array::<R>();
    let output = planes.output.array::<R>();
    let limit = params.amnt.unwrap_or(f32::MAX);

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        vinverse_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            sbr,
            sbr_blurred,
            output,
            params.contra_str,
            params.scl,
            limit,
        );
    }
}

/// The settings of one darken pass, all in the code values of the plane being filtered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DarkenSettings {
    /// How strongly a line is deepened, `strength / 128`.
    pub s: f32,
    /// The most a sample may grow while looking for its closing, `peak / (protection + 1)`.
    pub limit: f32,
    /// The brightest a closing may be.
    pub cap: f32,
    /// How far a closing must sit above a sample before the sample counts as a line.
    pub thr: f32,
    /// How hard thinning acts, `thinning / 16`. Thinning is off when this is 0.
    pub t: f32,
    /// The middle of the sample range, where a difference of 0 sits.
    pub n: f32,
    /// The largest sample value.
    pub peak: f32,
}

/// The planes one darken pass reads, writes and uses as scratch, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct DarkenPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    /// The plane grown by one sample, then reused for the thinned plane.
    pub maxed: DevicePlane<'a>,
    /// The grown plane held to `limit` above the input.
    pub limited: DevicePlane<'a>,
    pub closed: DevicePlane<'a>,
    pub diff: DevicePlane<'a>,
    /// Where the thinning step finds the darkest difference. It may be the plane of `closed`.
    pub lines: DevicePlane<'a>,
    /// Where the thinning step blurs `lines` into a weight. It may be the plane of `limited`.
    pub blurred: DevicePlane<'a>,
}

/// Holds each sample's expansion to `limit` above the sample.
#[cube(launch_unchecked)]
fn limit_kernel(input: &Array<f32>, maxed: &Array<f32>, limited: &mut Array<f32>, limit: f32) {
    if ABSOLUTE_POS >= limited.len() {
        terminate!();
    }

    limited[ABSOLUTE_POS] = f32::min(maxed[ABSOLUTE_POS], input[ABSOLUTE_POS] + limit);
}

/// Deepens the samples that sit below their closing, and writes how far each one sits below it.
#[cube(launch_unchecked)]
fn darken_kernel(
    input: &Array<f32>,
    closed: &Array<f32>,
    sharp: &mut Array<f32>,
    diff: &mut Array<f32>,
    cap: f32,
    thr: f32,
    s: f32,
    n: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= sharp.len() {
        terminate!();
    }

    let value = input[ABSOLUTE_POS];
    let closing = f32::min(closed[ABSOLUTE_POS], cap);

    let mut deepened = value;
    let mut offset = n;
    if closing > value + thr {
        deepened = value + (value - closing) * s;
        offset = value - closing + n;
    }

    sharp[ABSOLUTE_POS] = deepened;
    diff[ABSOLUTE_POS] = f32::max(f32::min(offset, peak), 0.0);
}

/// Turns the darkest difference around a sample into how much of the sharp result it keeps.
#[cube(launch_unchecked)]
fn line_weight_kernel(lines: &mut Array<f32>, n: f32, t: f32, peak: f32) {
    if ABSOLUTE_POS >= lines.len() {
        terminate!();
    }

    let weight = (lines[ABSOLUTE_POS] - n) * t + peak;

    lines[ABSOLUTE_POS] = f32::max(f32::min(weight, peak), 0.0);
}

/// Pulls the expanded plane in by the depth of each line.
#[cube(launch_unchecked)]
fn thin_kernel(maxed: &mut Array<f32>, diff: &Array<f32>, s: f32, n: f32) {
    if ABSOLUTE_POS >= maxed.len() {
        terminate!();
    }

    maxed[ABSOLUTE_POS] = maxed[ABSOLUTE_POS] + (diff[ABSOLUTE_POS] - n) * (s + 1.0);
}

/// Darkens the thin lines of `input` the way vs-jetpack's fast_line_darken does, writing `output`.
///
/// A line is a sample whose closing, the plane grown and shrunk by one sample, sits more than `thr`
/// above it. Lines are deepened by `s` times how far they sit below their closing. When `t` is above
/// 0, lines are also thinned, in the areas where the darkest difference around them says a line is
/// present.
pub fn darken<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: DarkenPlanes<'_>,
    settings: &DarkenSettings,
) {
    let grow = NeighbourPlanes {
        input: planes.input,
        output: planes.maxed,
    };
    extreme(client, dispatch, grow, Extreme::Maximum, Shape::Square);

    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let input = planes.input.array::<R>();
    let maxed = planes.maxed.array::<R>();
    let limited = planes.limited.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        limit_kernel::launch_unchecked::<R>(client, cubes, unit, input, maxed, limited, settings.limit);
    }

    let shrink = NeighbourPlanes {
        input: planes.limited,
        output: planes.closed,
    };
    extreme(client, dispatch, shrink, Extreme::Minimum, Shape::Square);

    let cubes = dispatch.cube_count(samples);
    let input = planes.input.array::<R>();
    let closed = planes.closed.array::<R>();
    let sharp = planes.output.array::<R>();
    let diff = planes.diff.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        darken_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            closed,
            sharp,
            diff,
            settings.cap,
            settings.thr,
            settings.s,
            settings.n,
            settings.peak,
        );
    }

    if settings.t <= 0.0 {
        return;
    }

    let darkest = NeighbourPlanes {
        input: planes.diff,
        output: planes.lines,
    };
    extreme(client, dispatch, darkest, Extreme::Minimum, Shape::Square);

    let cubes = dispatch.cube_count(samples);
    let lines = planes.lines.array::<R>();

    // Safety: the plane holds `samples` f32 values.
    unsafe {
        line_weight_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            lines,
            settings.n,
            settings.t,
            settings.peak,
        );
    }

    let spread = NeighbourPlanes {
        input: planes.lines,
        output: planes.blurred,
    };
    smooth(client, dispatch, spread, Smooth::Box);

    let cubes = dispatch.cube_count(samples);
    let maxed = planes.maxed.array::<R>();
    let diff = planes.diff.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        thin_kernel::launch_unchecked::<R>(client, cubes, unit, maxed, diff, settings.s, settings.n);
    }

    let mix = MergePlanes {
        input: planes.maxed,
        filtered: planes.output,
        weight: planes.blurred,
    };
    merge(client, dispatch, mix, settings.peak);
}

/// The settings of one toon pass, all in the code values of the plane being filtered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToonSettings {
    /// How strongly a line is deepened.
    pub strength: f32,
    /// The darkness at or below which a line is left alone.
    pub l: f32,
    /// The darkness at or above which a line is deepened in full.
    pub u: f32,
    /// The middle of the sample range, where a difference of 0 sits.
    pub n: f32,
    /// The largest sample value.
    pub peak: f32,
    /// The warp that thins the darkness map.
    pub warp: WarpParams,
}

/// The planes finding the darkness map reads, writes and uses as scratch, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ToonDifferencePlanes<'a> {
    pub input: DevicePlane<'a>,
    /// The plane grown by one sample.
    pub grown: DevicePlane<'a>,
    /// The grown plane shrunk by one sample.
    pub closed: DevicePlane<'a>,
    /// How far each sample sits below its closing.
    pub diff: DevicePlane<'a>,
    /// `diff` shifted up by `n`, ready to warp.
    pub shifted: DevicePlane<'a>,
}

/// The planes finishing a toon pass reads, writes and uses as scratch, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct ToonFinishPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
    /// The darkness map, held to the warped map in place.
    pub diff: DevicePlane<'a>,
    /// The warped darkness map, still shifted up by `n`.
    pub warped: DevicePlane<'a>,
    /// Where the largest darkness around each sample is written.
    pub widest: DevicePlane<'a>,
}

/// Writes how far each sample sits below its closing, held to `peak - n`, and that map shifted by `n`.
#[cube(launch_unchecked)]
fn toon_difference_kernel(
    input: &Array<f32>,
    closed: &Array<f32>,
    diff: &mut Array<f32>,
    shifted: &mut Array<f32>,
    n: f32,
    peak: f32,
) {
    if ABSOLUTE_POS >= diff.len() {
        terminate!();
    }

    let below = closed[ABSOLUTE_POS] - input[ABSOLUTE_POS];
    let held = f32::min(below, peak - n);

    diff[ABSOLUTE_POS] = held;
    shifted[ABSOLUTE_POS] = held + n;
}

/// Holds the darkness to what the warp left of it.
#[cube(launch_unchecked)]
fn toon_limit_kernel(diff: &mut Array<f32>, warped: &Array<f32>, n: f32) {
    if ABSOLUTE_POS >= diff.len() {
        terminate!();
    }

    diff[ABSOLUTE_POS] = f32::min(diff[ABSOLUTE_POS], warped[ABSOLUTE_POS] - n);
}

/// Deepens each sample by its darkness, scaled by how dark the line around it is.
#[cube(launch_unchecked)]
fn toon_kernel(
    input: &Array<f32>,
    diff: &Array<f32>,
    widest: &Array<f32>,
    output: &mut Array<f32>,
    l: f32,
    u: f32,
    strength: f32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let darkness = diff[ABSOLUTE_POS];
    let around = widest[ABSOLUTE_POS];

    let mut dark = darkness;
    if around <= l {
        dark = 0.0;
    } else if around < u {
        dark = darkness * (around - l) / (u - l);
    }

    output[ABSOLUTE_POS] = input[ABSOLUTE_POS] - dark * strength;
}

/// Finds how dark each sample is against its closing, and shifts that map up by `n` for the warp.
///
/// The closing is the plane grown then shrunk by one sample. The darkness is how far a sample
/// sits below it, held to `peak - n` so the shifted map stays inside the sample range.
pub fn toon_difference<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ToonDifferencePlanes<'_>,
    settings: &ToonSettings,
) {
    let grow = NeighbourPlanes {
        input: planes.input,
        output: planes.grown,
    };
    extreme(client, dispatch, grow, Extreme::Maximum, Shape::Square);

    let shrink = NeighbourPlanes {
        input: planes.grown,
        output: planes.closed,
    };
    extreme(client, dispatch, shrink, Extreme::Minimum, Shape::Square);

    let samples = planes.diff.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let input = planes.input.array::<R>();
    let closed = planes.closed.array::<R>();
    let diff = planes.diff.array::<R>();
    let shifted = planes.shifted.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        toon_difference_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            closed,
            diff,
            shifted,
            settings.n,
            settings.peak,
        );
    }
}

/// Deepens lines by their darkness, given the darkness map after the warp.
///
/// The darkness is held to what the warp left of it. Where the largest darkness around a sample is
/// at most `l` nothing changes, from `u` up the whole darkness is taken off, and between the two a
/// share of it is, growing evenly. That is scaled by `strength` and taken off the input.
pub fn toon_finish<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: ToonFinishPlanes<'_>,
    settings: &ToonSettings,
) {
    let samples = planes.output.samples();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();
    let diff = planes.diff.array::<R>();
    let warped = planes.warped.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        toon_limit_kernel::launch_unchecked::<R>(client, cubes.clone(), unit, diff, warped, settings.n);
    }

    let spread = NeighbourPlanes {
        input: planes.diff,
        output: planes.widest,
    };
    extreme(client, dispatch, spread, Extreme::Maximum, Shape::Square);

    let input = planes.input.array::<R>();
    let diff = planes.diff.array::<R>();
    let widest = planes.widest.array::<R>();
    let output = planes.output.array::<R>();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        toon_kernel::launch_unchecked::<R>(
            client,
            cubes,
            unit,
            input,
            diff,
            widest,
            output,
            settings.l,
            settings.u,
            settings.strength,
        );
    }
}

/// The settings of one thin pass, all in the code values of the plane being filtered.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThinSettings {
    /// The warp that thins lines.
    pub warp: WarpParams,
    /// The Prewitt magnitude at or above which a sample is part of a line.
    pub threshold: f32,
    /// How many square maximum passes grow the line mask.
    pub radius: u32,
    /// The largest sample value.
    pub peak: f32,
}
