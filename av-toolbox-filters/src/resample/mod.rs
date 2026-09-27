mod chroma;
mod depth;
#[cfg(test)]
mod tests;

use av_toolbox_cubek::Dispatch;
use av_toolbox_gpu::{CubeCount, CubeDim, FrameContext};

pub(crate) use self::chroma::Chroma;
pub(crate) use self::depth::Depth;

/// Sizes launches from the frame context a stage is handed.
pub(super) struct Sizing<'a>(pub(super) &'a FrameContext);

impl Dispatch for Sizing<'_> {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}
