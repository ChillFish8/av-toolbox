use std::fmt;
use std::sync::Arc;

use av_toolbox_core::StageKind;
use av_toolbox_graph::{Diagnostic, Severity, Span};
use av_toolbox_pipeline::StageRef;
use av_toolbox_runtime::RunError;
use codespan_reporting::term::termcolor::WriteColor;

use crate::report::{self, PipelineFiles};

/// A written stage, kept after the compiled pipeline that named it is gone.
#[derive(Debug, Clone)]
pub(crate) struct NamedStage {
    kind: StageKind,
    name: String,
    span: Span,
}

impl NamedStage {
    pub(crate) fn new(stage: StageRef<'_>) -> Self {
        Self {
            kind: stage.kind,
            name: stage.name.to_owned(),
            span: stage.span,
        }
    }

    pub(crate) fn as_ref(&self) -> StageRef<'_> {
        StageRef {
            kind: self.kind,
            name: &self.name,
            span: self.span,
        }
    }
}

enum Failure {
    Diagnostics {
        files: Arc<PipelineFiles>,
        found: Vec<Diagnostic>,
    },
    Run {
        files: Arc<PipelineFiles>,
        error: RunError,
        stage: Option<NamedStage>,
    },
    Message(String),
}

/// Why a session could not load, start or finish a stream.
///
/// Problems in the pipeline carry their diagnostics and the files they point into, so
/// [SessionError::render] can draw them against the TOML.
pub struct SessionError {
    failure: Box<Failure>,
}

impl SessionError {
    pub(crate) fn pipeline(files: Arc<PipelineFiles>, found: Vec<Diagnostic>) -> Self {
        let failure = Failure::Diagnostics { files, found };
        Self {
            failure: Box::new(failure),
        }
    }

    pub(crate) fn run(files: Arc<PipelineFiles>, error: RunError, stage: Option<NamedStage>) -> Self {
        let failure = Failure::Run { files, error, stage };
        Self {
            failure: Box::new(failure),
        }
    }

    pub(crate) fn message(message: impl Into<String>) -> Self {
        let failure = Failure::Message(message.into());
        Self {
            failure: Box::new(failure),
        }
    }

    /// The diagnostics a pipeline that failed to load or compile raised, empty for any other failure.
    pub fn diagnostics(&self) -> &[Diagnostic] {
        match self.failure.as_ref() {
            Failure::Diagnostics { found, .. } => found,
            Failure::Run { .. } | Failure::Message(_) => &[],
        }
    }

    /// The error a failed run ended with, `None` for any other failure.
    pub fn run_error(&self) -> Option<&RunError> {
        match self.failure.as_ref() {
            Failure::Run { error, .. } => Some(error),
            Failure::Diagnostics { .. } | Failure::Message(_) => None,
        }
    }

    /// Whether [SessionError::render] draws this error against the pipeline's files.
    ///
    /// When it does not, the error is a single line of text, the same as its `Display`.
    pub fn points_at_files(&self) -> bool {
        match self.failure.as_ref() {
            Failure::Diagnostics { .. } => true,
            Failure::Run { stage, .. } => stage.is_some(),
            Failure::Message(_) => false,
        }
    }

    /// Writes the error to `writer`, drawing diagnostics and the failing stage against the TOML.
    pub fn render(&self, writer: &mut dyn WriteColor) {
        match self.failure.as_ref() {
            Failure::Diagnostics { files, found } => report::diagnostics(writer, files, found),
            Failure::Run {
                files,
                error,
                stage: Some(stage),
            } => {
                let written = stage.as_ref();
                report::run_error(writer, files, error, Some(written));
            },
            Failure::Run { stage: None, .. } | Failure::Message(_) => {
                let summary = self.to_string();
                report::error_line(writer, &summary);
            },
        }
    }
}

impl fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.failure.as_ref() {
            Failure::Diagnostics { found, .. } => write_found(formatter, found),
            Failure::Run {
                error,
                stage: Some(stage),
                ..
            } => {
                let cause = report::innermost(error);
                write!(formatter, "{} {} failed: {cause}", stage.kind, stage.name)
            },
            Failure::Run {
                error, stage: None, ..
            } => {
                let message = report::plain_message(error);
                formatter.write_str(&message)
            },
            Failure::Message(message) => formatter.write_str(message),
        }
    }
}

/// Writes the first error in `found`, or the first warning when there is none, and how many more there are.
fn write_found(formatter: &mut fmt::Formatter<'_>, found: &[Diagnostic]) -> fmt::Result {
    let first_error = found
        .iter()
        .find(|diagnostic| diagnostic.severity() == Severity::Error);
    let Some(first) = first_error.or(found.first()) else {
        return formatter.write_str("the pipeline failed with no diagnostics");
    };

    let message = first.message();
    let others = found.len() - 1;
    match others {
        0 => formatter.write_str(message),
        1 => write!(formatter, "{message}, and 1 more problem"),
        others => write!(formatter, "{message}, and {others} more problems"),
    }
}

impl fmt::Debug for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let summary = self.to_string();
        formatter.debug_tuple("SessionError").field(&summary).finish()
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.failure.as_ref() {
            Failure::Run { error, .. } => Some(error),
            Failure::Diagnostics { .. } | Failure::Message(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageError;
    use av_toolbox_graph::{DiagnosticCode, NodeId};
    use av_toolbox_pipeline::SourceMap;
    use codespan_reporting::term::termcolor::NoColor;

    use super::*;

    const TEXT: &str = "[sink.output]\ntype = \"channel/frames\"\n";

    fn files() -> Arc<PipelineFiles> {
        let mut sources = SourceMap::new();
        sources.push("pipeline.toml".into(), TEXT.to_owned(), Vec::new());

        Arc::new(PipelineFiles::new(sources))
    }

    fn rendered(error: &SessionError) -> String {
        let mut writer = NoColor::new(Vec::new());
        error.render(&mut writer);

        String::from_utf8(writer.into_inner()).expect("utf-8")
    }

    fn stage_failure() -> RunError {
        RunError::Stage {
            node: NodeId::new(0),
            source: StageError::msg("the caller stopped receiving frames"),
        }
    }

    #[test]
    fn a_run_error_names_its_stage() {
        let stage = StageRef {
            kind: StageKind::Sink,
            name: "output",
            span: Span::new(0, 13),
        };
        let named = NamedStage::new(stage);

        let error = SessionError::run(files(), stage_failure(), Some(named));

        assert_eq!(
            error.to_string(),
            "sink output failed: the caller stopped receiving frames"
        );
        assert!(error.points_at_files());
        assert!(rendered(&error).contains("pipeline.toml:1:1"));
    }

    #[test]
    fn a_run_error_without_a_stage_renders_one_line() {
        let error = SessionError::run(files(), stage_failure(), None);

        let output = rendered(&error);

        assert!(!error.points_at_files());
        assert_eq!(
            output,
            "error: a transfer between host and device failed: the caller stopped receiving frames\n"
        );
    }

    #[test]
    fn the_summary_counts_the_other_problems() {
        let first = Diagnostic::error(
            DiagnosticCode::UnknownStage,
            "",
            Span::unknown(),
            "no sink is registered",
        );
        let second = Diagnostic::warning(DiagnosticCode::UnusedStage, "", Span::unknown(), "unused");

        let error = SessionError::pipeline(files(), vec![second, first]);

        assert_eq!(error.to_string(), "no sink is registered, and 1 more problem");
    }
}
