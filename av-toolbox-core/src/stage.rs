use crate::tag::Tag;

/// A [Stage] is a single abstract point within a pipeline.
pub trait Stage {
    /// The identifier of the stage.
    ///
    /// This is used when referencing the stage in a pipeline.
    fn identifier() -> &'static str;

    /// Labelling tags used for organisation.
    fn tags() -> &'static [Tag];
}