use std::fmt;

use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use crate::{BoundInputs, InputSignature, StageError, StreamType, Tag};

/// The category a stage belongs to.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StageKind {
    Source,
    Filter,
    Mask,
    Transform,
    Sink,
}

impl StageKind {
    /// What a stage of this kind produces, or `None` for a sink.
    pub const fn output(&self) -> Option<StreamType> {
        match self {
            Self::Source | Self::Filter | Self::Transform => Some(StreamType::Frames),
            Self::Mask => Some(StreamType::Mask),
            Self::Sink => None,
        }
    }
}

impl fmt::Display for StageKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Source => "source",
            Self::Filter => "filter",
            Self::Mask => "mask",
            Self::Transform => "transform",
            Self::Sink => "sink",
        };

        formatter.write_str(name)
    }
}

/// A constructed stage ready to be placed in a pipeline.
pub trait StageInstance: Send + 'static {}

/// A type of stage that can be registered and built.
///
/// The arguments type supplies the JSON schema for a stage's arguments, and the input signature
/// declares which inputs it accepts.
///
/// # Example
///
/// ```
/// use av_toolbox_core::{
///     BoundInputs,
///     Filter,
///     InputParam,
///     InputSignature,
///     PipelineRegistry,
///     StageDefinition,
///     StageError,
///     StageInstance,
/// };
/// use schemars::JsonSchema;
/// use serde::Deserialize;
///
/// #[derive(Debug, Deserialize, JsonSchema)]
/// struct DenoiseArgs {
///     sigma_scale: f32,
/// }
///
/// struct DenoiseFilter {
///     sigma_scale: f32,
/// }
///
/// impl StageInstance for DenoiseFilter {}
/// impl Filter for DenoiseFilter {}
///
/// struct Denoise;
///
/// impl StageDefinition for Denoise {
///     type Args = DenoiseArgs;
///     type Instance = DenoiseFilter;
///
///     fn package() -> &'static str {
///         "av-denoise"
///     }
///
///     fn identifier() -> &'static str {
///         "nl4d"
///     }
///
///     fn inputs() -> &'static InputSignature {
///         const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
///         SIGNATURE
///     }
///
///     fn build(args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
///         Ok(DenoiseFilter {
///             sigma_scale: args.sigma_scale,
///         })
///     }
/// }
///
/// let mut registry = PipelineRegistry::new();
/// registry.register_filter::<Denoise>().expect("should register");
/// ```
pub trait StageDefinition: 'static {
    /// The arguments this stage takes, supplied by the pipeline as JSON.
    type Args: DeserializeOwned + JsonSchema;

    /// The stage this definition builds.
    type Instance: StageInstance;

    fn package() -> &'static str;

    fn identifier() -> &'static str;

    fn tags() -> &'static [Tag] {
        &[]
    }

    fn inputs() -> &'static InputSignature;

    /// Builds the stage from its arguments and bound inputs.
    fn build(args: Self::Args, inputs: BoundInputs) -> Result<Self::Instance, StageError>;
}

#[cfg(test)]
mod tests {
    use schemars::JsonSchema;
    use serde::Deserialize;

    use super::*;
    use crate::{InputParam, bind_inputs};

    #[derive(Debug, Deserialize, JsonSchema)]
    struct ExampleArgs {
        sigma_scale: f32,
    }

    struct ExampleInstance {
        sigma_scale: f32,
    }

    impl StageInstance for ExampleInstance {}

    struct ExampleDefinition;

    impl StageDefinition for ExampleDefinition {
        type Args = ExampleArgs;
        type Instance = ExampleInstance;

        fn package() -> &'static str {
            "av-denoise"
        }

        fn identifier() -> &'static str {
            "nl4d"
        }

        fn inputs() -> &'static InputSignature {
            const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
            SIGNATURE
        }

        fn build(args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
            Ok(ExampleInstance {
                sigma_scale: args.sigma_scale,
            })
        }
    }

    #[test]
    fn a_definition_builds_its_instance() {
        let arguments = [];
        let bound = bind_inputs(&InputSignature::EMPTY, &arguments).expect("should bind");
        let instance =
            ExampleDefinition::build(ExampleArgs { sigma_scale: 0.8 }, bound).expect("should build");

        assert_eq!(instance.sigma_scale, 0.8);
    }

    #[test]
    fn tags_default_to_empty() {
        assert!(ExampleDefinition::tags().is_empty());
    }

    #[test]
    fn every_kind_but_a_sink_produces_a_stream() {
        assert_eq!(StageKind::Source.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Filter.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Transform.output(), Some(StreamType::Frames));
        assert_eq!(StageKind::Mask.output(), Some(StreamType::Mask));
        assert_eq!(StageKind::Sink.output(), None);
    }
}
