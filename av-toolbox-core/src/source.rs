use av_toolbox_gpu::{Frame, FrameContext};

use crate::{StageError, StageInstance};

/// A [Source] produces video frames to be processed by the pipeline.
pub trait Source: StageInstance {
    /// The next frame, or `None` once the stream has ended.
    ///
    /// A source that is done never has to be called again, since nothing downstream calls it
    /// after this returns `None`.
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError>;
}
