use crate::StageInstance;

/// A [Transform] alters the pipeline stream itself rather than any content of the frames
/// within it.
///
/// A transform may emit more frames than it was given, or fewer, or group them into chunks so they
/// can be processed in parallel. What flows out of it is still frames.
pub trait Transform: StageInstance {}
