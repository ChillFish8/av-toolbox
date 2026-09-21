use av_toolbox_gpu::{Frame, FrameContext};

use crate::{Inputs, StageError, StageInstance};

/// A [Filter] applies a set of changes to a video stream.
pub trait Filter: StageInstance {
    fn process(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<Frame, StageError>;

    /// Frames still held when the stream ends, in order.
    ///
    /// A filter holding a temporal window flushes what it buffered here, since nothing else asks
    /// it for more once the stream is over.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        Ok(Vec::new())
    }
}
