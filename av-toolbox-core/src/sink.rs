use av_toolbox_gpu::FrameContext;

use crate::{Inputs, StageError, StageInstance};

/// A [Sink] consumes a filtered stream of frames from the pipeline.
pub trait Sink: StageInstance {
    fn consume(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<(), StageError>;

    /// Called once the stream has ended.
    ///
    /// A sink that is given one chunk at a time is called at the end of every chunk rather
    /// than once at the end of the run.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        Ok(())
    }
}
