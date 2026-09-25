use av_toolbox_graph::{Diagnostic, Severity};
use av_toolbox_pipeline::{SourceMap, StageRef};
use av_toolbox_runtime::RunError;
use codespan_reporting::diagnostic::{Diagnostic as Report, Label, Severity as Level};
use codespan_reporting::files::{Error as FilesError, SimpleFiles};
use codespan_reporting::term::termcolor::{Color, ColorSpec, StandardStream, WriteColor};
use codespan_reporting::term::{self, Config};

use crate::cli::ColorMode;

pub(crate) fn stderr(mode: ColorMode) -> StandardStream {
    let choice = mode.choice();
    StandardStream::stderr(choice)
}

/// Prints `error: <message>` with the level coloured.
pub(crate) fn error_line(writer: &mut dyn WriteColor, message: &str) {
    // Nothing is left to report a failed write to stderr to.
    let _ = write_line(writer, "error", Color::Red, message);
}

fn write_line(writer: &mut dyn WriteColor, level: &str, colour: Color, message: &str) -> std::io::Result<()> {
    let mut spec = ColorSpec::new();
    spec.set_fg(Some(colour)).set_bold(true);

    writer.set_color(&spec)?;
    write!(writer, "{level}")?;
    writer.reset()?;
    writeln!(writer, ": {message}")
}

/// Every file a pipeline was read from, for pointing diagnostics at.
pub(crate) struct PipelineFiles {
    files: SimpleFiles<String, String>,
    sources: SourceMap,
}

impl PipelineFiles {
    pub(crate) fn new(sources: SourceMap) -> Self {
        let mut files = SimpleFiles::new();

        for (_, source) in sources.iter() {
            let display = source.path.display();
            let name = display.to_string();
            files.add(name, source.text.clone());
        }

        Self { files, sources }
    }
}

/// Prints every diagnostic against the files, errors first, then a count of what was found.
pub(crate) fn diagnostics(writer: &mut dyn WriteColor, files: &PipelineFiles, diagnostics: &[Diagnostic]) {
    // Nothing is left to report a failed write to stderr to.
    let _ = write_diagnostics(writer, files, diagnostics);
}

fn write_diagnostics(
    writer: &mut dyn WriteColor,
    files: &PipelineFiles,
    diagnostics: &[Diagnostic],
) -> Result<(), FilesError> {
    if diagnostics.is_empty() {
        return Ok(());
    }

    let config = Config::default();
    let errors = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity() == Severity::Error);
    let warnings = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity() == Severity::Warning);

    for diagnostic in errors.chain(warnings) {
        let noted = files.sources.with_call_notes(diagnostic.clone());
        let report = convert(&noted);
        term::emit_to_write_style(writer, &config, &files.files, &report)?;
    }

    write_summary(writer, diagnostics)?;

    Ok(())
}

fn convert(diagnostic: &Diagnostic) -> Report<usize> {
    let level = match diagnostic.severity() {
        Severity::Error => Level::Error,
        Severity::Warning => Level::Warning,
    };

    let mut labels = Vec::new();
    let mut notes = Vec::new();

    let span = diagnostic.span();
    let path = diagnostic.path();
    if !span.is_unknown() {
        let file_id = span.file.index();
        let label = Label::primary(file_id, span.start..span.end);
        labels.push(label);
    } else if !path.is_empty() {
        let located = format!("in {path}");
        notes.push(located);
    }

    for note in diagnostic.notes() {
        if note.span.is_unknown() {
            let message = note.message.clone();
            notes.push(message);
            continue;
        }

        let note_file = note.span.file.index();
        let label = Label::secondary(note_file, note.span.start..note.span.end).with_message(&note.message);
        labels.push(label);
    }

    let code = diagnostic.code();
    let message = diagnostic.message();

    Report::new(level)
        .with_code(code.as_str())
        .with_message(message)
        .with_labels(labels)
        .with_notes(notes)
}

fn write_summary(writer: &mut dyn WriteColor, diagnostics: &[Diagnostic]) -> std::io::Result<()> {
    let errors = count(diagnostics, Severity::Error);
    let warnings = count(diagnostics, Severity::Warning);

    let counted = match (errors, warnings) {
        (0, warnings) => plural(warnings, "warning"),
        (errors, 0) => plural(errors, "error"),
        (errors, warnings) => {
            let errors = plural(errors, "error");
            let warnings = plural(warnings, "warning");

            format!("{errors} and {warnings}")
        },
    };
    let message = format!("pipeline has {counted}");

    match errors {
        0 => write_line(writer, "warning", Color::Yellow, &message),
        _ => write_line(writer, "error", Color::Red, &message),
    }
}

fn count(diagnostics: &[Diagnostic], severity: Severity) -> usize {
    diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.severity() == severity)
        .count()
}

fn plural(count: usize, noun: &str) -> String {
    match count {
        1 => format!("1 {noun}"),
        count => format!("{count} {noun}s"),
    }
}

/// Prints a failed run at the stage that failed.
///
/// A node the compiler inserted, such as a host/device transfer, has no written stage to point
/// at, so it prints a plain line naming the transfer instead of the error's own `Display`, which
/// would otherwise leak the node's internal id. A node id nothing in the run names at all falls
/// back to the error's own `Display`.
pub(crate) fn run_error(
    writer: &mut dyn WriteColor,
    files: &PipelineFiles,
    error: &RunError,
    stage: Option<StageRef<'_>>,
) {
    let Some(stage) = stage else {
        let message = plain_message(error);
        error_line(writer, &message);

        return;
    };

    let cause = innermost(error);
    let message = format!("{} {} failed: {cause}", stage.kind, stage.name);
    let file_id = stage.span.file.index();
    let label = Label::primary(file_id, stage.span.start..stage.span.end);
    let mut labels = vec![label];

    let source = files.sources.get(stage.span.file);
    let called_from = source
        .map(|source| source.called_from.as_slice())
        .unwrap_or_default();
    for call in called_from {
        let call_file = call.file.index();
        let call_label = Label::secondary(call_file, call.start..call.end).with_message("called from here");
        labels.push(call_label);
    }

    let report = Report::error().with_message(message).with_labels(labels);
    let config = Config::default();

    // Nothing is left to report a failed write to stderr to.
    let _ = term::emit_to_write_style(writer, &config, &files.files, &report);
}

/// The message for a run error with no written stage to point at.
///
/// A node is known but unwritten when the compiler inserted it, and that is always a transfer
/// between host and device.
fn plain_message(error: &RunError) -> String {
    if error.node().is_none() {
        return error.to_string();
    }

    let cause = innermost(error);
    format!("a transfer between host and device failed: {cause}")
}

/// The deepest cause of a run error, which names what went wrong without the node it happened on.
fn innermost(error: &RunError) -> String {
    let mut current: &dyn std::error::Error = error;
    while let Some(next) = current.source() {
        current = next;
    }

    current.to_string()
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::{StageError, StageKind};
    use av_toolbox_graph::{DiagnosticCode, NodeId, Span};
    use av_toolbox_pipeline::{SourceMap, StageRef};
    use av_toolbox_runtime::RunError;
    use codespan_reporting::term::termcolor::NoColor;

    use super::*;

    const TEXT: &str = "[filter.blur]\ntype = \"test/blur\"\n";

    fn file(text: &str) -> PipelineFiles {
        let mut sources = SourceMap::new();
        sources.push("pipeline.toml".into(), text.to_string(), Vec::new());

        PipelineFiles::new(sources)
    }

    fn rendered(text: &str, diagnostics: &[Diagnostic]) -> String {
        let mut writer = NoColor::new(Vec::new());

        super::diagnostics(&mut writer, &file(text), diagnostics);

        String::from_utf8(writer.into_inner()).expect("utf-8")
    }

    fn error(span: Span) -> Diagnostic {
        Diagnostic::error(
            DiagnosticCode::UnknownStage,
            "filter.blur.type",
            span,
            "no filter is registered",
        )
    }

    #[test]
    fn an_error_line_names_its_level() {
        let mut writer = NoColor::new(Vec::new());

        error_line(&mut writer, "cannot read pipeline.toml");
        let written = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert_eq!(written, "error: cannot read pipeline.toml\n");
    }

    #[test]
    fn a_primary_label_underlines_its_span() {
        let output = rendered(TEXT, &[error(Span::new(21, 32))]);

        assert!(output.contains("error[unknown-stage]: no filter is registered"));
        assert!(output.contains("pipeline.toml:2:8"));
        assert!(output.contains("^^^^^^^^^^^"));
    }

    #[test]
    fn a_note_with_a_span_is_a_secondary_label() {
        let diagnostic = error(Span::new(21, 32)).with_note("defined here", Span::new(0, 13));

        let output = rendered(TEXT, &[diagnostic]);

        assert!(output.contains("-------------"));
        assert!(output.contains("defined here"));
    }

    #[test]
    fn a_note_without_a_span_is_a_note_line() {
        let diagnostic = error(Span::new(21, 32)).with_note("try list-stages", Span::unknown());

        let output = rendered(TEXT, &[diagnostic]);

        assert!(output.contains("= try list-stages"));
    }

    #[test]
    fn an_unknown_span_names_the_path_instead() {
        let diagnostic = Diagnostic::error(
            DiagnosticCode::NoSinks,
            "",
            Span::unknown(),
            "the pipeline has no sink",
        );
        let with_path = error(Span::unknown());

        let output = rendered(TEXT, &[diagnostic, with_path]);

        assert!(output.contains("error[no-sinks]: the pipeline has no sink"));
        assert!(output.contains("= in filter.blur.type"));
        assert!(!output.contains("pipeline.toml:"));
    }

    #[test]
    fn a_label_after_multi_byte_text_lands_on_its_text() {
        let text = "# réglage\n[filter.blur]\n";
        let header = text.find("[filter.blur]").expect("present");

        let output = rendered(text, &[error(Span::new(header, header + 13))]);

        assert!(output.contains("pipeline.toml:2:1"));
        assert!(output.contains("^^^^^^^^^^^^^"));
    }

    #[test]
    fn errors_print_before_warnings() {
        let warning = Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            "filter.blur",
            Span::new(0, 13),
            "unused",
        );
        let failure = error(Span::new(21, 32));

        let output = rendered(TEXT, &[warning, failure]);
        let error_at = output.find("error[").expect("an error");
        let warning_at = output.find("warning[").expect("a warning");

        assert!(error_at < warning_at);
    }

    #[test]
    fn the_summary_counts_errors_and_warnings() {
        let warning = Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            "filter.blur",
            Span::new(0, 13),
            "unused",
        );
        let failure = error(Span::new(21, 32));

        let output = rendered(TEXT, &[warning, failure]);

        assert!(output.ends_with("error: pipeline has 1 error and 1 warning\n"));
    }

    #[test]
    fn warnings_alone_are_summarised_as_warnings() {
        let first = Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            "filter.a",
            Span::new(0, 13),
            "unused",
        );
        let second = Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            "filter.b",
            Span::new(0, 13),
            "unused",
        );

        let output = rendered(TEXT, &[first, second]);

        assert!(output.ends_with("warning: pipeline has 2 warnings\n"));
    }

    #[test]
    fn nothing_to_report_prints_nothing() {
        assert_eq!(rendered(TEXT, &[]), "");
    }

    #[test]
    fn a_diagnostic_in_a_called_file_shows_that_file_and_the_call() {
        let mut sources = SourceMap::new();
        let root_text = "[call.part]\npath = \"part.toml\"\n";
        let root = sources.push("main.toml".into(), root_text.to_string(), Vec::new());
        let call_header = Span::in_file(root, 0, 11);
        let part = sources.push("part.toml".into(), TEXT.to_string(), vec![call_header]);
        let files = PipelineFiles::new(sources);
        let failure = error(Span::in_file(part, 21, 32));
        let mut writer = NoColor::new(Vec::new());

        super::diagnostics(&mut writer, &files, &[failure]);
        let output = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert!(output.contains("part.toml:2:8"));
        assert!(output.contains("main.toml:1:1"));
        assert!(output.contains("called from here"));
    }

    #[test]
    fn a_run_error_points_at_its_stage() {
        let failure = RunError::Stage {
            node: NodeId::new(0),
            source: StageError::msg("the reader closed the stream"),
        };
        let stage = StageRef {
            kind: StageKind::Filter,
            name: "blur",
            span: Span::new(0, 13),
        };
        let mut writer = NoColor::new(Vec::new());

        run_error(&mut writer, &file(TEXT), &failure, Some(stage));
        let output = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert!(output.contains("filter blur failed: the reader closed the stream"));
        assert!(output.contains("^^^^^^^^^^^^^"));
        assert!(!output.contains("NodeId"));
    }

    #[test]
    fn a_run_error_in_a_called_file_notes_the_call() {
        let mut sources = SourceMap::new();
        let root_text = "[call.part]\npath = \"part.toml\"\n";
        let root = sources.push("main.toml".into(), root_text.to_string(), Vec::new());
        let call_header = Span::in_file(root, 0, 11);
        let part = sources.push("part.toml".into(), TEXT.to_string(), vec![call_header]);
        let files = PipelineFiles::new(sources);
        let failure = RunError::Stage {
            node: NodeId::new(0),
            source: StageError::msg("the reader closed the stream"),
        };
        let stage = StageRef {
            kind: StageKind::Filter,
            name: "part.blur",
            span: Span::in_file(part, 0, 13),
        };
        let mut writer = NoColor::new(Vec::new());

        run_error(&mut writer, &files, &failure, Some(stage));
        let output = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert!(output.contains("part.toml:1:1"));
        assert!(output.contains("main.toml:1:1"));
        assert!(output.contains("called from here"));
    }

    #[test]
    fn a_run_error_with_no_stage_is_a_plain_line() {
        let failure = RunError::DroppedItems { dropped: 3 };
        let mut writer = NoColor::new(Vec::new());

        run_error(&mut writer, &file(TEXT), &failure, None);
        let output = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert_eq!(
            output,
            "error: the run finished with 3 items that never reached a worker\n"
        );
    }

    #[test]
    fn a_run_error_at_a_compiler_inserted_node_hides_the_node_id() {
        let failure = RunError::Stage {
            node: NodeId::new(3),
            source: StageError::msg("the device was lost"),
        };
        let mut writer = NoColor::new(Vec::new());

        run_error(&mut writer, &file(TEXT), &failure, None);
        let output = String::from_utf8(writer.into_inner()).expect("utf-8");

        assert_eq!(
            output,
            "error: a transfer between host and device failed: the device was lost\n"
        );
        assert!(!output.contains("NodeId"));
    }
}
