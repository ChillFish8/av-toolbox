use av_toolbox_cubek::Dispatch;
use av_toolbox_gpu::{CubeCount, CubeDim, FrameContext};

/// Sizes launches from the frame context a stage is handed.
pub(crate) struct Sizing<'a>(pub(crate) &'a FrameContext);

impl Dispatch for Sizing<'_> {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}
