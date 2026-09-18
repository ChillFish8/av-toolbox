use crate::StageInstance;

/// A [Source] produces video frames to be processed by the pipeline.
pub trait Source: StageInstance {}
