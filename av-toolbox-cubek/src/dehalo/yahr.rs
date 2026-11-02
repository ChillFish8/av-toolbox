// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use crate::depth::Element;
use crate::warp::{BlurKind, WarpParams};
use crate::{DevicePlane, Dispatch};

/// Everything havsfunc's YAHR runs its warp with.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct YahrParams {
    /// How many times YAHR's edge mask is blurred before it warps.
    pub blur: u32,
    /// How far samples move toward edges. A negative depth moves them away.
    pub depth: i32,
    /// How the plane's samples are stored.
    pub element: Element,
    /// The bits one sample holds.
    pub bits: u32,
}

impl YahrParams {
    /// The warp YAHR runs its own edge-aware pass through, with an r6 blur and a threshold of 128.
    pub fn warp_params(&self) -> WarpParams {
        WarpParams {
            element: self.element,
            bits: self.bits,
            thresh: 128,
            blur: BlurKind::R6,
            passes: self.blur,
            depth: self.depth,
        }
    }
}

/// The planes one YAHR combine reads and writes, all f32 and one size.
#[derive(Debug, Clone, Copy)]
pub struct YahrPlanes<'a> {
    pub clip: DevicePlane<'a>,
    pub detail: DevicePlane<'a>,
    pub limited: DevicePlane<'a>,
    pub output: DevicePlane<'a>,
}

#[cube(launch_unchecked)]
fn yahr_combine_kernel(clip: &[f32], detail: &[f32], limited: &[f32], output: &mut [f32]) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let change = detail[ABSOLUTE_POS] - limited[ABSOLUTE_POS];

    output[ABSOLUTE_POS] = clip[ABSOLUTE_POS] - change;
}

/// Writes `clip - (detail - limited)` at each sample.
pub fn yahr_combine(client: &Client, dispatch: &impl Dispatch, planes: YahrPlanes<'_>) {
    let samples = planes.output.samples();
    let clip = planes.clip.array();
    let detail = planes.detail.array();
    let limited = planes.limited.array();
    let output = planes.output.array();
    let cubes = dispatch.cube_count(samples);
    let unit = dispatch.cube_dim();

    // Safety: every plane holds `samples` f32 values.
    unsafe {
        yahr_combine_kernel::launch_unchecked(client, cubes, unit, clip, detail, limited, output);
    }
}
