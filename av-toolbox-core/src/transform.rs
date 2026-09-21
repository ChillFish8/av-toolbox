use av_toolbox_gpu::{FrameContext, StreamItem};

use crate::{Inputs, StageError, StageInstance};

/// A [Transform] alters the pipeline stream itself rather than any content of the frames
/// within it.
///
/// A transform may emit more frames than it was given, or fewer, or group them into chunks so they
/// can be processed in parallel. What flows out of it is still frames.
pub trait Transform: StageInstance {
    /// What this emits for one input, which may be nothing.
    fn process(&mut self, context: &mut FrameContext, inputs: Inputs) -> Result<Vec<StreamItem>, StageError>;

    /// What this emits once the stream has ended, which may be nothing.
    ///
    /// A transform that is given one chunk at a time is asked at the end of every chunk
    /// rather than once at the end of the run.
    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<StreamItem>, StageError> {
        Ok(Vec::new())
    }
}
