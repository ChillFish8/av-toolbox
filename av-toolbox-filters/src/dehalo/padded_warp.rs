use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::dehalo::PADDING;
use av_toolbox_cubek::padding::{crop, pad};
use av_toolbox_cubek::values::{narrow, widen};
use av_toolbox_cubek::warp::{WarpParams, WarpPlanes, warp_sharp};
use av_toolbox_gpu::{ComputeClient, FrameContext, Runtime};

use crate::scratch::{self, Scratch};
use crate::sizing::Sizing;

/// Runs [warp_sharp] on `input` grown by [PADDING] on every side, then crops the result into
/// `output`.
///
/// `input` is rounded to whole samples before the warp and widened back to code values after, the
/// way aWarpSharp2's own mask and warp work in integer samples.
pub(super) fn padded_warp<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    params: &WarpParams,
) {
    let sizing = Sizing(context);
    let width = input.width();
    let height = input.height();
    let padded_width = width + 2 * PADDING;
    let padded_height = height + 2 * PADDING;

    let padded = Scratch::new::<f32>(context, padded_width, padded_height);
    let padded_plane = padded.plane();
    pad(client, &sizing, input, padded_plane, PADDING);

    let peak = params.peak() as f32;
    let narrowed = scratch::samples(context, padded_width, padded_height, params.element);
    let narrowed_plane = narrowed.plane();
    narrow(
        client,
        &sizing,
        padded_plane,
        narrowed_plane,
        params.element,
        peak,
    );

    let mask = Scratch::new::<u32>(context, padded_width, padded_height);
    let warp_scratch = Scratch::new::<u32>(context, padded_width, padded_height);
    let warped = scratch::samples(context, padded_width, padded_height, params.element);
    let mask_plane = mask.plane();
    let warp_scratch_plane = warp_scratch.plane();
    let warped_plane = warped.plane();
    let warp_planes = WarpPlanes {
        input: narrowed_plane,
        mask: mask_plane,
        scratch: warp_scratch_plane,
        output: warped_plane,
    };
    warp_sharp(client, &sizing, warp_planes, params);

    let widened = Scratch::new::<f32>(context, padded_width, padded_height);
    let widened_plane = widened.plane();
    widen(client, &sizing, warped_plane, widened_plane, params.element);

    crop(client, &sizing, widened_plane, output, PADDING);
}
