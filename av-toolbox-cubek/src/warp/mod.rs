mod blur;
#[cfg(test)]
mod tests;

// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

pub use self::blur::blur_mask;
use crate::depth::Element;
use crate::{DevicePlane, Dispatch};

/// Which blur smooths an edge mask.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BlurKind {
    /// Reaches 6 samples either side.
    R6,
    /// Reaches 2 samples either side.
    R2,
}

/// Everything one aWarpSharp2 warp needs.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct WarpParams {
    pub element: Element,
    /// The bits one sample holds, between 8 and 16.
    pub bits: u32,
    /// The most an edge can weigh in the mask, on the 8-bit scale.
    pub thresh: u32,
    pub blur: BlurKind,
    /// How many times the blur runs.
    pub passes: u32,
    /// How far samples move toward edges, between -128 and 127. A negative depth moves them away.
    pub depth: i32,
}

impl WarpParams {
    /// The largest sample value.
    pub fn peak(&self) -> i32 {
        let levels = 1i32 << self.bits;

        levels - 1
    }

    /// `thresh` at the sample's own depth.
    pub fn scaled_thresh(&self) -> i32 {
        let shift = self.bits - 8;
        let scaled = self.thresh << shift;

        scaled as i32
    }
}

/// The planes one warp sharpen reads, passes through and writes.
///
/// `mask` and `scratch` hold u32 values. Every plane is one size.
#[derive(Debug, Clone, Copy)]
pub struct WarpPlanes<'a> {
    pub input: DevicePlane<'a>,
    pub mask: DevicePlane<'a>,
    pub scratch: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

/// The rounded mean of two values, the way aWarpSharp2 averages.
#[cube]
fn average(first: i32, second: i32) -> i32 {
    (first + second + 1) >> 1
}

#[cube]
fn sample_at<E: Int>(input: &Array<E>, width: u32, column: i32, row: i32) -> i32 {
    let index = row as u32 * width + column as u32;

    i32::cast_from(input[index as usize])
}

#[cube]
fn mask_at(mask: &Array<u32>, width: u32, column: i32, row: i32) -> i32 {
    let index = row as u32 * width + column as u32;

    i32::cast_from(mask[index as usize])
}

#[cube(launch_unchecked)]
fn edge_mask_kernel<E: Int>(
    input: &Array<E>,
    mask: &mut Array<u32>,
    width: u32,
    height: u32,
    peak: i32,
    thresh: i32,
) {
    if ABSOLUTE_POS >= mask.len() {
        terminate!();
    }

    if width < 3 || height < 3 {
        mask[ABSOLUTE_POS] = 0u32;
    } else {
        let position = ABSOLUTE_POS as u32;
        let column_position = (position % width) as i32;
        let row_position = (position / width) as i32;
        let last_inner_column = width as i32 - 2;
        let last_inner_row = height as i32 - 2;
        let x = i32::clamp(column_position, 1, last_inner_column);
        let y = i32::clamp(row_position, 1, last_inner_row);

        let top_left = sample_at::<E>(input, width, x - 1, y - 1);
        let top = sample_at::<E>(input, width, x, y - 1);
        let top_right = sample_at::<E>(input, width, x + 1, y - 1);
        let left = sample_at::<E>(input, width, x - 1, y);
        let right = sample_at::<E>(input, width, x + 1, y);
        let bottom_left = sample_at::<E>(input, width, x - 1, y + 1);
        let bottom = sample_at::<E>(input, width, x, y + 1);
        let bottom_right = sample_at::<E>(input, width, x + 1, y + 1);

        let top_corners = average(top_left, top_right);
        let bottom_corners = average(bottom_left, bottom_right);
        let left_corners = average(bottom_left, top_left);
        let right_corners = average(bottom_right, top_right);
        let above = average(top, top_corners);
        let below = average(bottom, bottom_corners);
        let before = average(left, left_corners);
        let after = average(right, right_corners);

        let vertical = i32::abs(above - below);
        let horizontal = i32::abs(before - after);
        let strongest = i32::max(horizontal, vertical);
        let summed = i32::min(vertical + horizontal, peak);
        let boosted = i32::min(summed + strongest, peak);
        let doubled = i32::min(boosted * 2, peak);
        let tripled = i32::min(doubled + boosted, peak);
        let edge = i32::min(tripled * 2, peak);
        let capped = i32::min(edge, thresh);

        mask[ABSOLUTE_POS] = u32::cast_from(capped);
    }
}

#[cube(launch_unchecked)]
fn warp_kernel<E: Int>(
    input: &Array<E>,
    mask: &Array<u32>,
    output: &mut Array<E>,
    width: u32,
    height: u32,
    extra_bits: i32,
    depth: i32,
    peak: i32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let x = (position % width) as i32;
    let y = (position / width) as i32;
    let last_column = width as i32 - 1;
    let last_row = height as i32 - 1;

    let row_above = i32::max(y - 1, 0);
    let row_below = i32::min(y + 1, last_row);
    let column_before = i32::max(x - 1, 0);
    let column_after = i32::min(x + 1, last_column);
    let above = mask_at(mask, width, x, row_above);
    let below = mask_at(mask, width, x, row_below);
    let before = mask_at(mask, width, column_before, y);
    let after = mask_at(mask, width, column_after, y);

    let scaled_depth = depth << 8;
    let horizontal_shifted = (before - after) >> extra_bits;
    let vertical_shifted = (above - below) >> extra_bits;
    let horizontal_fixed = ((horizontal_shifted << 7) * scaled_depth) >> 16;
    let vertical_fixed = ((vertical_shifted << 7) * scaled_depth) >> 16;

    let lowest = -y * 128;
    let highest = (height as i32 - y) * 128 - 129;
    let vertical_raised = i32::max(vertical_fixed, lowest);
    let vertical_limited = i32::min(vertical_raised, highest);

    let fraction_across = horizontal_fixed & 127;
    let fraction_down = vertical_limited & 127;
    let column_unclamped = x + (horizontal_fixed >> 7);
    let row_unclamped = y + (vertical_limited >> 7);

    let mut weight_across = fraction_across;
    if column_unclamped >= last_column || column_unclamped < 0 {
        weight_across = 0;
    }

    let column = i32::clamp(column_unclamped, 0, last_column);
    let next_column = i32::min(column + 1, last_column);
    let row = i32::clamp(row_unclamped, 0, last_row);
    let next_row = i32::clamp(row_unclamped + 1, 0, last_row);

    let top_left = sample_at::<E>(input, width, column, row);
    let top_right = sample_at::<E>(input, width, next_column, row);
    let bottom_left = sample_at::<E>(input, width, column, next_row);
    let bottom_right = sample_at::<E>(input, width, next_column, next_row);

    let top_mix = (top_left * (128 - weight_across) + top_right * weight_across + 64) >> 7;
    let bottom_mix = (bottom_left * (128 - weight_across) + bottom_right * weight_across + 64) >> 7;
    let blended = (top_mix * (128 - fraction_down) + bottom_mix * fraction_down + 64) >> 7;
    let clamped = i32::clamp(blended, 0, peak);

    output[ABSOLUTE_POS] = E::cast_from(clamped);
}

/// Writes aWarpSharp2's edge mask of `input` into `mask`, which holds u32 values.
pub fn edge_mask<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    mask: DevicePlane<'_>,
    params: &WarpParams,
) {
    match params.element {
        Element::U8 => launch_edge_mask::<u8, R>(client, dispatch, input, mask, params),
        Element::U16 => launch_edge_mask::<u16, R>(client, dispatch, input, mask, params),
    }
}

fn launch_edge_mask<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    mask: DevicePlane<'_>,
    params: &WarpParams,
) {
    let width = input.width();
    let height = input.height();
    let samples = mask.samples();
    let peak = params.peak();
    let thresh = params.scaled_thresh();
    let input_array = input.array::<R>();
    let mask_array = mask.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input holds `width * height` samples of `E` and the mask as many u32 values, and
    // every read is clamped inside the plane.
    unsafe {
        edge_mask_kernel::launch_unchecked::<E, R>(
            client,
            cubes,
            unit,
            input_array,
            mask_array,
            width,
            height,
            peak,
            thresh,
        );
    }
}

/// Moves each sample of `input` along the slope of `mask` into `output`.
pub fn warp_plane<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    mask: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &WarpParams,
) {
    match params.element {
        Element::U8 => launch_warp::<u8, R>(client, dispatch, input, mask, output, params),
        Element::U16 => launch_warp::<u16, R>(client, dispatch, input, mask, output, params),
    }
}

fn launch_warp<E: Int, R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    mask: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &WarpParams,
) {
    let width = input.width();
    let height = input.height();
    let samples = output.samples();
    let extra_bits = params.bits as i32 - 8;
    let peak = params.peak();
    let input_array = input.array::<R>();
    let mask_array = mask.array::<R>();
    let output_array = output.array::<R>();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: the input and output hold `width * height` samples of `E` and the mask as many u32
    // values, and every read is clamped inside the plane.
    unsafe {
        warp_kernel::launch_unchecked::<E, R>(
            client,
            cubes,
            unit,
            input_array,
            mask_array,
            output_array,
            width,
            height,
            extra_bits,
            params.depth,
            peak,
        );
    }
}

/// Runs the whole aWarpSharp2 warp on one plane.
///
/// It builds the edge mask, blurs it `params.passes` times, then warps the input along it.
pub fn warp_sharp<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    planes: WarpPlanes<'_>,
    params: &WarpParams,
) {
    edge_mask(client, dispatch, planes.input, planes.mask, params);

    for _ in 0..params.passes {
        blur_mask(client, dispatch, planes.mask, planes.scratch, params.blur);
    }

    warp_plane(client, dispatch, planes.input, planes.mask, planes.output, params);
}
