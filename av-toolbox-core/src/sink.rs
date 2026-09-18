use crate::StageInstance;

/// A [Sink] consumes a filtered stream of frames from the pipeline.
pub trait Sink: StageInstance {}
