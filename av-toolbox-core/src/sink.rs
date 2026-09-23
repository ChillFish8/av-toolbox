use av_toolbox_gpu::{DeviceSide, FrameContext, Side};

use crate::{Inputs, StageError, StageInstance};

/// A [Sink] consumes a filtered stream of frames from the pipeline.
///
/// The side defaults to the device, so `impl Sink for X` still means a device sink.
pub trait Sink<S: Side = DeviceSide>: StageInstance {
    fn consume(&mut self, context: &mut FrameContext, inputs: Inputs<S>) -> Result<(), StageError>;

    /// Called once the stream has ended.
    ///
    /// A sink that is given one chunk at a time is called at the end of every chunk rather
    /// than once at the end of the run.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        Ok(())
    }
}
