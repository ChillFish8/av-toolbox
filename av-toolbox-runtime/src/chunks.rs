use av_toolbox_gpu::{ChunkRoute, FrameMeta};

/// The chunk a frame belongs to on `route`.
///
/// Without a route nothing divides the stream, so every frame is in chunk 0.
pub(crate) fn chunk_in(meta: &FrameMeta, route: Option<ChunkRoute>) -> u64 {
    route.map_or(0, |route| route.chunk_of(meta))
}
