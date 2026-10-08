use av_toolbox_gpu::{Channel, FrameContext, FrameFormat, FrameLayout, FrameMeta, Plane};

/// Each plane's share of a `width` by `height` frame laid out as `format`, in channel order.
pub(super) fn channel_shares(format: &FrameFormat, width: u32, height: u32) -> Vec<(u32, u32)> {
    let channels = format.channels();

    channels
        .iter()
        .map(|channel| {
            let dimensions = format.layout.channel_dimensions(*channel, width, height);

            dimensions.expect("every channel of a format's own layout is carried")
        })
        .collect()
}

/// A fresh plane of `source`'s size in `format`'s sample format, or `None` when the frame allocated for it
/// carries no plane.
pub(super) fn plane_like(
    context: &FrameContext,
    format: &FrameFormat,
    source: &Plane,
    meta: &FrameMeta,
) -> Option<Plane> {
    let (width, height) = source.dimensions();
    let plane_format = FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        ..*format
    };
    let frame = context.alloc_frame(&plane_format, meta.clone());
    let plane = frame.plane(Channel::Y)?;

    Some(plane.clone())
}
