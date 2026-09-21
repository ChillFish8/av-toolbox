use av_toolbox_gpu::FrameContext;

use crate::{Inputs, StageError, StageInstance};

/// A [Sink] consumes a filtered stream of frames from the pipeline.
pub trait Sink: StageInstance {
    fn consume(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<(), StageError>;

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        Ok(())
    }
}
