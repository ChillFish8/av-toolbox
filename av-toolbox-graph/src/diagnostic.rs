use std::fmt;

use av_toolbox_core::StageKind;

use crate::Span;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Severity {
    Error,
    Warning,
}

impl fmt::Display for Severity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Error => "error",
            Self::Warning => "warning",
        };

        formatter.write_str(name)
    }
}

/// The rule a diagnostic was raised by.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum DiagnosticCode {
    InvalidSyntax,
    InvalidDefinition,
    DuplicateStageName,
    UnknownStageReference,
    UnknownStage,
    KindMismatch,
    SourceTakesInputs,
    SinkHasNoOutput,
    NoSinks,
    InvalidInputs,
    Cycle,
    UnusedStage,
    InvalidArgs,
    StageBuildFailed,
    NoCommonFormat,
    StreamTypeMismatch,
    FrameRateMismatch,
    InvalidWorkers,
    ResidencyRoundTrip,
    TransfersNotPlaced,
    UnknownVariable,
    MissingVariable,
    InvalidVariable,
    UnknownCallOutput,
    UnreadableFile,
    CallCycle,
    ConflictingMaskPlanes,
    HostMaskPlanes,
    HeldInFrameRegion,
    UnsatisfiedDemand,
    PlaceFailed,
    RemovedTransforms,
}

impl DiagnosticCode {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::InvalidSyntax => "invalid-syntax",
            Self::InvalidDefinition => "invalid-definition",
            Self::DuplicateStageName => "duplicate-stage-name",
            Self::UnknownStageReference => "unknown-stage-reference",
            Self::UnknownStage => "unknown-stage",
            Self::KindMismatch => "kind-mismatch",
            Self::SourceTakesInputs => "source-takes-inputs",
            Self::SinkHasNoOutput => "sink-has-no-output",
            Self::NoSinks => "no-sinks",
            Self::InvalidInputs => "invalid-inputs",
            Self::Cycle => "cycle",
            Self::UnusedStage => "unused-stage",
            Self::InvalidArgs => "invalid-args",
            Self::StageBuildFailed => "stage-build-failed",
            Self::NoCommonFormat => "no-common-format",
            Self::StreamTypeMismatch => "stream-type-mismatch",
            Self::FrameRateMismatch => "frame-rate-mismatch",
            Self::InvalidWorkers => "invalid-workers",
            Self::ResidencyRoundTrip => "residency-round-trip",
            Self::TransfersNotPlaced => "transfers-not-placed",
            Self::UnknownVariable => "unknown-variable",
            Self::MissingVariable => "missing-variable",
            Self::InvalidVariable => "invalid-variable",
            Self::UnknownCallOutput => "unknown-call-output",
            Self::UnreadableFile => "unreadable-file",
            Self::CallCycle => "call-cycle",
            Self::ConflictingMaskPlanes => "conflicting-mask-planes",
            Self::HostMaskPlanes => "host-mask-planes",
            Self::HeldInFrameRegion => "held-in-frame-region",
            Self::UnsatisfiedDemand => "unsatisfied-demand",
            Self::PlaceFailed => "place-failed",
            Self::RemovedTransforms => "removed-transforms",
        }
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A second location a diagnostic wants to point at.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Note {
    pub message: String,
    pub span: Span,
}

/// One problem found in a set of definitions.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    severity: Severity,
    code: DiagnosticCode,
    message: String,
    path: String,
    span: Span,
    notes: Vec<Note>,
}

impl Diagnostic {
    pub fn error(
        code: DiagnosticCode,
        path: impl Into<String>,
        span: Span,
        message: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Error, code, path, span, message)
    }

    pub fn warning(
        code: DiagnosticCode,
        path: impl Into<String>,
        span: Span,
        message: impl Into<String>,
    ) -> Self {
        Self::new(Severity::Warning, code, path, span, message)
    }

    fn new(
        severity: Severity,
        code: DiagnosticCode,
        path: impl Into<String>,
        span: Span,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code,
            message: message.into(),
            path: path.into(),
            span,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, message: impl Into<String>, span: Span) -> Self {
        self.notes.push(Note {
            message: message.into(),
            span,
        });
        self
    }

    pub fn severity(&self) -> Severity {
        self.severity
    }

    pub fn code(&self) -> DiagnosticCode {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn span(&self) -> Span {
        self.span
    }

    pub fn notes(&self) -> &[Note] {
        &self.notes
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}[{}]", self.severity, self.code)?;
        if !self.path.is_empty() {
            write!(formatter, " {}", self.path)?;
        }

        write!(formatter, "\n  {}", self.message)?;
        for note in &self.notes {
            write!(formatter, "\n  note: {}", note.message)?;
        }

        Ok(())
    }
}

/// Every problem found by one pass over a set of definitions.
#[derive(Debug, Default, thiserror::Error)]
#[error("{}", render_diagnostics(&self.entries))]
pub struct Diagnostics {
    entries: Vec<Diagnostic>,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.entries.push(diagnostic);
    }

    pub fn extend(&mut self, other: Diagnostics) {
        self.entries.extend(other.entries);
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn has_errors(&self) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.severity() == Severity::Error)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.entries.iter()
    }

    pub fn into_inner(self) -> Vec<Diagnostic> {
        self.entries
    }
}

impl From<Vec<Diagnostic>> for Diagnostics {
    fn from(entries: Vec<Diagnostic>) -> Self {
        Self { entries }
    }
}

fn render_diagnostics(entries: &[Diagnostic]) -> String {
    let lines: Vec<String> = entries.iter().map(|entry| entry.to_string()).collect();
    lines.join("\n\n")
}

pub(crate) fn stage_path(kind: StageKind, name: &str) -> String {
    format!("{kind}.{name}")
}

pub(crate) fn type_path(kind: StageKind, name: &str) -> String {
    format!("{kind}.{name}.type")
}

pub(crate) fn args_path(kind: StageKind, name: &str) -> String {
    format!("{kind}.{name}.args")
}

pub(crate) fn input_path(kind: StageKind, name: &str, index: usize) -> String {
    format!("{kind}.{name}.inputs[{index}]")
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;

    #[test]
    fn a_code_renders_as_kebab_case() {
        assert_eq!(
            DiagnosticCode::UnknownStageReference.as_str(),
            "unknown-stage-reference"
        );
    }

    #[test]
    fn a_diagnostic_renders_its_severity_code_path_and_message() {
        let diagnostic = Diagnostic::error(
            DiagnosticCode::UnknownStageReference,
            "filter.deband.inputs[1]",
            Span::new(10, 20),
            "no stage named input_C is defined",
        );

        let rendered = diagnostic.to_string();

        assert_eq!(
            rendered,
            "error[unknown-stage-reference] filter.deband.inputs[1]\n  no stage named input_C is defined"
        );
    }

    #[test]
    fn a_note_renders_under_its_diagnostic() {
        let diagnostic = Diagnostic::error(
            DiagnosticCode::DuplicateStageName,
            "filter.deband",
            Span::new(10, 20),
            "a stage named deband is already defined",
        )
        .with_note("first defined here", Span::new(1, 5));

        assert!(diagnostic.to_string().ends_with("\n  note: first defined here"));
    }

    #[test]
    fn a_set_reports_whether_it_holds_an_error() {
        let mut diagnostics = Diagnostics::new();
        diagnostics.push(Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            "filter.grain",
            Span::unknown(),
            "no sink depends on this stage",
        ));

        assert!(!diagnostics.has_errors());

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::NoSinks,
            "",
            Span::unknown(),
            "the graph has no sink",
        ));

        assert!(diagnostics.has_errors());
    }

    #[test]
    fn the_new_codes_render_as_kebab_case() {
        assert_eq!(DiagnosticCode::UnknownVariable.as_str(), "unknown-variable");
        assert_eq!(DiagnosticCode::MissingVariable.as_str(), "missing-variable");
        assert_eq!(DiagnosticCode::InvalidVariable.as_str(), "invalid-variable");
        assert_eq!(DiagnosticCode::UnknownCallOutput.as_str(), "unknown-call-output");
        assert_eq!(DiagnosticCode::UnreadableFile.as_str(), "unreadable-file");
        assert_eq!(DiagnosticCode::CallCycle.as_str(), "call-cycle");
    }

    #[test]
    fn the_mask_plane_codes_render_as_kebab_case() {
        assert_eq!(
            DiagnosticCode::ConflictingMaskPlanes.as_str(),
            "conflicting-mask-planes"
        );
        assert_eq!(DiagnosticCode::HostMaskPlanes.as_str(), "host-mask-planes");
    }

    #[test]
    fn extending_a_set_keeps_every_entry_in_order() {
        let mut first = Diagnostics::new();
        first.push(Diagnostic::error(
            DiagnosticCode::NoSinks,
            "",
            Span::unknown(),
            "a",
        ));

        let mut second = Diagnostics::new();
        second.push(Diagnostic::error(DiagnosticCode::Cycle, "", Span::unknown(), "b"));

        first.extend(second);
        let codes: Vec<DiagnosticCode> = first.iter().map(|entry| entry.code()).collect();

        assert_eq!(codes, [DiagnosticCode::NoSinks, DiagnosticCode::Cycle]);
    }

    #[test]
    fn paths_name_the_kind_the_stage_and_the_field() {
        assert_eq!(stage_path(StageKind::Filter, "deband"), "filter.deband");
        assert_eq!(type_path(StageKind::Filter, "deband"), "filter.deband.type");
        assert_eq!(args_path(StageKind::Filter, "deband"), "filter.deband.args");
        assert_eq!(
            input_path(StageKind::Filter, "deband", 1),
            "filter.deband.inputs[1]"
        );
    }
}
