use av_toolbox_gpu::{DeviceSide, FrameContext, Side};

use crate::{Inputs, StageError, StageInstance};

/// A [Filter] applies a set of changes to a video stream.
///
/// The side defaults to the device.
pub trait Filter<S: Side = DeviceSide>: StageInstance {
    fn process(&mut self, context: &mut FrameContext, inputs: Inputs<S>) -> Result<S::Frame, StageError>;

    /// Frames still held when the stream ends, in order.
    ///
    /// A filter holding a temporal window flushes what it buffered here, since nothing else asks
    /// it for more once the stream is over. A filter that is given one chunk at a time
    /// flushes at the end of every chunk rather than once at the end of the run.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<S::Frame>, StageError> {
        Ok(Vec::new())
    }
}
