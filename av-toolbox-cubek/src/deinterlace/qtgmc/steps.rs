use cubecl::prelude::*;

use super::{QtgmcTables, clamped_difference, merge, resharpen, search_clip, subtract, vertical_midpoint};
use crate::neighbourhood::{Edge, NeighbourPlanes, Smooth, smooth_with_edge};
use crate::resize::{ResizePlanes, Weights, apply_float_with_edge};
use crate::{DevicePlane, Dispatch};

/// Scratch planes the composite steps pass through, all of one plane size.
pub struct StepScratch<'a> {
    pub first: DevicePlane<'a>,
    pub second: DevicePlane<'a>,
    pub third: DevicePlane<'a>,
}

/// Whether every plane is held by a separate handle.
fn are_distinct(planes: &[DevicePlane<'_>]) -> bool {
    planes.iter().enumerate().all(|(index, plane)| {
        planes[index + 1..]
            .iter()
            .all(|other| !std::ptr::eq(plane.handle(), other.handle()))
    })
}

/// The weight of `repaired` in the search clip's blur.
const SEARCH_REPAIRED_WEIGHT: f32 = 0.1;

/// Blurs `input` into `output` with the 3x3 blur and then the gaussian, both clamped at the border.
fn blur_and_spread<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    scratch: (DevicePlane<'_>, DevicePlane<'_>),
    spread: (&Weights, &Weights),
    output: DevicePlane<'_>,
) {
    let (blurred, rows) = scratch;
    let smoothing = NeighbourPlanes {
        input,
        output: blurred,
    };
    smooth_with_edge(client, dispatch, smoothing, Smooth::Binomial, Edge::Clamp);

    let resizing = ResizePlanes {
        input: blurred,
        scratch: rows,
        output,
    };
    apply_float_with_edge(client, dispatch, resizing, spread.0, spread.1, Edge::Clamp);
}

/// The planes the search clip is built from, all of one size and all separate.
pub struct SearchPlanes<'a> {
    pub repaired: DevicePlane<'a>,
    pub bob: DevicePlane<'a>,
    pub scratch: StepScratch<'a>,
    pub output: DevicePlane<'a>,
}

/// Builds the clip motion is searched on from `repaired` and the `bob`.
pub fn search_clip_from_repaired<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    tables: &QtgmcTables,
    planes: SearchPlanes<'_>,
    bits: u32,
) {
    let SearchPlanes {
        repaired,
        bob,
        scratch,
        output,
    } = planes;
    let StepScratch { first, second, third } = scratch;
    debug_assert!(
        are_distinct(&[repaired, bob, first, second, third, output]),
        "the search clip planes must all be separate"
    );

    let gauss = &tables.narrow;
    let spread = (&gauss.across, &gauss.down);
    blur_and_spread(client, dispatch, repaired, (first, second), spread, third);

    merge(client, dispatch, third, repaired, first, SEARCH_REPAIRED_WEIGHT);
    search_clip(client, dispatch, first, repaired, bob, output, bits);
}

/// Resharpens `input` as SMode 2 does, by `strength`.
pub fn smode2_resharpen<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    input: DevicePlane<'_>,
    scratch: StepScratch<'_>,
    output: DevicePlane<'_>,
    strength: f32,
    bits: u32,
) {
    let StepScratch { first, second, .. } = scratch;
    debug_assert!(
        are_distinct(&[input, first, second, output]),
        "the resharpen planes must all be separate"
    );

    vertical_midpoint(client, dispatch, input, first);

    let smoothing = NeighbourPlanes {
        input: first,
        output: second,
    };
    smooth_with_edge(client, dispatch, smoothing, Smooth::Binomial, Edge::Clamp);

    resharpen(client, dispatch, input, second, output, strength, bits);
}

/// The planes the back-blend is built from, all of one size and all separate.
pub struct BackBlendPlanes<'a> {
    pub sharpened: DevicePlane<'a>,
    pub unsharpened: DevicePlane<'a>,
    pub scratch: StepScratch<'a>,
    pub output: DevicePlane<'a>,
}

/// Removes the blur of what sharpening added to luma from `sharpened`.
pub fn back_blend<R: Runtime>(
    client: &ComputeClient<R>,
    dispatch: &impl Dispatch,
    tables: &QtgmcTables,
    planes: BackBlendPlanes<'_>,
    bits: u32,
) {
    let BackBlendPlanes {
        sharpened,
        unsharpened,
        scratch,
        output,
    } = planes;
    let StepScratch { first, second, third } = scratch;
    debug_assert!(
        are_distinct(&[sharpened, unsharpened, first, second, third, output]),
        "the back-blend planes must all be separate"
    );

    clamped_difference(client, dispatch, sharpened, unsharpened, first, bits);

    let gauss = &tables.wide;
    let spread = (&gauss.across, &gauss.down);
    blur_and_spread(client, dispatch, first, (second, third), spread, first);
    subtract(client, dispatch, sharpened, first, output, bits);
}
