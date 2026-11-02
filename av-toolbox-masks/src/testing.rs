use av_toolbox_cubek::DevicePlane;
use av_toolbox_gpu::FrameContext;
use cubecl::prelude::Client;

use crate::sizing::Sizing;

/// Writes `round(clamp(v, 0, peak) * 255 / peak)` from `input` into `output` as u8.
pub fn store_mask(
    client: &Client,
    context: &FrameContext,
    input: DevicePlane<'_>,
    output: DevicePlane<'_>,
    peak: f32,
) {
    let sizing = Sizing(context);

    crate::luma::store_mask(client, &sizing, input, output, peak);
}
