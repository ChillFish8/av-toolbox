use av_toolbox_core::{PathEdge, PathView, PlaceError};

/// Places a scene provider after the stage `after` names, or directly after the source.
pub(crate) fn place(after: Option<&str>, path: &PathView<'_>) -> Result<PathEdge, PlaceError> {
    match after {
        Some(name) => path.after_named(name),
        None => Ok(path.after_source()),
    }
}
