use std::error::Error;
use std::fmt;

use crate::{StageKind, StreamType};

/// A failure raised by a stage while it is building itself.
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct StageError {
    message: String,
    #[source]
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl StageError {
    pub fn msg(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source: None,
        }
    }

    pub fn with_source(message: impl Into<String>, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

/// A failure raised while registering a stage.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("the {segment} of stage {package}/{identifier} is empty or contains a '/'")]
    InvalidIdentifier {
        package: String,
        identifier: String,
        segment: &'static str,
    },
    #[error("stage {identifier} has no description")]
    MissingDescription { identifier: String },
    #[error("a {kind} stage with the identifier {identifier} is already registered")]
    DuplicateStage { kind: StageKind, identifier: String },
    #[error("source {identifier} declares inputs, sources take no inputs")]
    SourceTakesInputs { identifier: String },
    #[error("artefact sink {identifier} must declare InputSignature::ARTEFACTS as its inputs")]
    ArtefactSinkSignature { identifier: String },
    #[error("the schema generated for stage {identifier} is not valid: {reason}")]
    InvalidSchema { identifier: String, reason: String },
    #[error("the identifier {identifier} is reserved for the transfers the graph inserts")]
    ReservedIdentifier { identifier: String },
    #[error("stage {identifier} input {param} has an invalid planes_of: {reason}")]
    InvalidPlanesOf {
        identifier: String,
        param: &'static str,
        reason: &'static str,
    },
    #[error("stage {identifier} demands something of `{input}`, which is no input of it")]
    InvalidDemand { identifier: String, input: &'static str },
    #[error("stage {identifier} probes `{input}`, which is no input of it")]
    InvalidProbe { identifier: String, input: &'static str },
    #[error("stage {identifier} makes more than one {kind} probe of `{input}`")]
    DuplicateProbeInput {
        identifier: String,
        input: &'static str,
        kind: &'static str,
    },
    #[error("a demand kind named {kind} is already registered")]
    DuplicateDemand { kind: &'static str },
    #[error("a probe kind named {kind} is already registered")]
    DuplicateProbe { kind: &'static str },
    #[error("a provider of {kind} demands is already registered")]
    DuplicateProvider { kind: &'static str },
    #[error("stage {identifier} declares the artefact {name} more than once")]
    DuplicateArtefact { identifier: String, name: &'static str },
    #[error(
        "stage {identifier} declares an artefact named {name:?}, which must be non-empty and hold no '.' or '/'"
    )]
    InvalidArtefactName { identifier: String, name: &'static str },
}

/// A failure raised while binding a pipeline's inputs to a stage's signature.
#[derive(Debug, thiserror::Error)]
pub enum BindInputsError {
    #[error("the input at position {position} is positional but follows a named input")]
    PositionalAfterNamed { position: usize },
    #[error("the input named {name} at position {position} is already bound")]
    DuplicateInput { name: String, position: usize },
    #[error("the required input {name} was not provided")]
    MissingRequiredInput { name: &'static str },
    #[error("unexpected input at position {position}, the stage takes at most {expected} inputs")]
    TooManyInputs { position: usize, expected: usize },
    #[error("the stage has no input named {name}, given at position {position}")]
    UnknownInput { name: String, position: usize },
    #[error("the input at position {position} produces {found}, but {expected} was expected")]
    InputTypeMismatch {
        name: Option<&'static str>,
        position: usize,
        expected: StreamType,
        found: StreamType,
    },
}

/// A single way in which some arguments failed to match a stage's schema.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SchemaViolation {
    path: String,
    message: String,
}

impl SchemaViolation {
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SchemaViolation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.path.is_empty() {
            return formatter.write_str(&self.message);
        }

        write!(formatter, "{}: {}", self.path, self.message)
    }
}

/// A failure raised while checking a stage's arguments.
#[derive(Debug, thiserror::Error)]
pub enum ArgsError {
    #[error("the arguments do not match the schema:\n{}", render_violations(.0))]
    SchemaViolations(Vec<SchemaViolation>),
    #[error("the arguments could not be deserialized")]
    Deserialize(#[source] serde_json::Error),
    #[error(transparent)]
    InvalidDemand(RegistryError),
    #[error(transparent)]
    InvalidProbe(RegistryError),
    #[error(
        "stage {identifier} is chunked on a variadic input given without a name, so its chunk input has no name"
    )]
    UnnamedChunkInput { identifier: String },
}

fn render_violations(violations: &[SchemaViolation]) -> String {
    let lines: Vec<String> = violations
        .iter()
        .map(|violation| format!("  - {violation}"))
        .collect();

    lines.join("\n")
}

/// A failure raised while building a stage from a pipeline definition.
#[derive(Debug, thiserror::Error)]
pub enum BuildStageError {
    #[error("no {kind} stage is registered with the identifier {identifier}")]
    UnknownStage { kind: StageKind, identifier: String },
    #[error("invalid arguments for stage {identifier}")]
    InvalidArgs {
        identifier: String,
        #[source]
        source: ArgsError,
    },
    #[error("invalid inputs for stage {identifier}")]
    InvalidInputs {
        identifier: String,
        #[source]
        source: BindInputsError,
    },
    #[error("stage {identifier} could not be built")]
    Build {
        identifier: String,
        #[source]
        source: StageError,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_error_exposes_its_source() {
        let cause = serde_json::from_str::<u32>("nope").expect_err("should not parse");
        let error = StageError::with_source("could not open the file", cause);

        assert_eq!(error.to_string(), "could not open the file");
        assert!(std::error::Error::source(&error).is_some());
    }

    #[test]
    fn unknown_stage_names_the_kind_it_looked_in() {
        let error = BuildStageError::UnknownStage {
            kind: StageKind::Sink,
            identifier: "av-denoise/nl4d".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "no sink stage is registered with the identifier av-denoise/nl4d",
        );
    }

    #[test]
    fn schema_violations_are_all_listed() {
        let violations = vec![
            SchemaViolation::new("/sigma_scale", "\"a\" is not of type \"number\""),
            SchemaViolation::new("/strength", "5 is greater than the maximum of 3"),
        ];
        let error = ArgsError::SchemaViolations(violations);

        let rendered = error.to_string();
        assert!(rendered.contains("/sigma_scale"));
        assert!(rendered.contains("/strength"));
    }
}
