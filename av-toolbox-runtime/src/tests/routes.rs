use av_toolbox_core::Parallelism;
use av_toolbox_core::testing::{FixtureChunk, FixtureCuts};
use av_toolbox_gpu::{ChunkRoute, FrameMeta, LayerId};

/// A route reading [FixtureChunk](av_toolbox_core::testing::FixtureChunk) values from `layer`.
pub(crate) fn route(layer: u32) -> ChunkRoute {
    let parallelism = Parallelism::chunk::<FixtureCuts>(());
    let key = parallelism.chunk_key().expect("a chunk stage has a key");

    ChunkRoute {
        layer: LayerId(layer),
        key,
    }
}

/// The metadata of the frame at `index`, in `chunk` of `layer`.
pub(crate) fn meta_in(index: u64, layer: u32, chunk: u64) -> FrameMeta {
    let route = route(layer);
    let value = FixtureChunk(chunk);

    FrameMeta::at(index).with_provided(route.layer, value)
}

/// The chunk `meta` is in on `layer`.
pub(crate) fn chunk_on(meta: &FrameMeta, layer: u32) -> u64 {
    route(layer).chunk_of(meta)
}
