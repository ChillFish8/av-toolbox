use av_toolbox_gpu::{FrameMeta, LayerId};

/// The chunk a frame belongs to in `layer`.
///
/// Without a layer nothing divides the stream, so every frame is in chunk 0.
pub(crate) fn chunk_in(meta: &FrameMeta, layer: Option<LayerId>) -> u64 {
    match layer {
        Some(layer) => meta.chunks.get(layer),
        None => 0,
    }
}
