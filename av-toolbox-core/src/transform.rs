use crate::Stage;

/// A [Transform] alters the pipeline stream itself rather than any content of the frames
/// within it.
pub trait Transform: Stage {
    
}