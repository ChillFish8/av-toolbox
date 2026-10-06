use av_toolbox_gpu::{FrameMeta, LayerId};

/// The chunk a frame belongs to in `layer`, or its plain chunk when there is no layer.
pub(crate) fn chunk_in(meta: &FrameMeta, layer: Option<LayerId>) -> u64 {
    match layer {
        Some(layer) => meta.chunks.get(layer),
        None => meta.chunk,
    }
}
