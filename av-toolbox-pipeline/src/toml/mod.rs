mod args;
mod stage;
#[cfg(test)]
mod tests;

use std::borrow::Cow;
use std::ops::Range;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::DeTable;
use av_toolbox_core::StageKind;
use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, Span, StageSpec};

use self::stage::read_stage;

/// The tables a pipeline file may hold, one per stage kind.
const KINDS: [(&str, StageKind); 5] = [
    ("source", StageKind::Source),
    ("filter", StageKind::Filter),
    ("mask", StageKind::Mask),
    ("transform", StageKind::Transform),
    ("sink", StageKind::Sink),
];

/// Reads a pipeline file into stage definitions.
///
/// Every problem in the file is reported, not only the first. Stages come back in the order their
/// definitions start in the text.
pub fn parse(text: &str) -> Result<Vec<StageSpec>, Diagnostics> {
    let (document, errors) = DeTable::parse_recoverable(text);
    if !errors.is_empty() {
        let reported: Vec<Diagnostic> = errors.iter().map(syntax_error).collect();
        let diagnostics = Diagnostics::from(reported);
        return Err(diagnostics);
    }

    let mut diagnostics = Diagnostics::new();
    let mut specs = Vec::new();

    let table = document.get_ref();
    for (key, value) in table.iter() {
        let name = key.get_ref();
        let Some(kind) = kind_named(name) else {
            let diagnostic = unknown_kind(key);
            diagnostics.push(diagnostic);
            continue;
        };

        let inner = value.get_ref();
        let Some(stages) = inner.as_table() else {
            let range = value.span();
            let span = span_of(range);
            let message = format!("{kind} must be a table of stages, such as [{kind}.name]");
            let path = kind.to_string();
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);
            diagnostics.push(diagnostic);
            continue;
        };

        for (stage_name, stage) in stages.iter() {
            let read = read_stage(kind, stage_name, stage, &mut diagnostics);
            if let Some(spec) = read {
                specs.push(spec);
            }
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    specs.sort_by_key(|spec| spec.span.start);

    Ok(specs)
}

fn span_of(range: Range<usize>) -> Span {
    Span::new(range.start, range.end)
}

fn kind_named(name: &str) -> Option<StageKind> {
    let found = KINDS.iter().find(|(table, _)| *table == name);
    found.map(|(_, kind)| *kind)
}

fn syntax_error(error: &::toml::de::Error) -> Diagnostic {
    let unknown = Span::unknown();
    let span = error.span().map(span_of).unwrap_or(unknown);
    let message = error.message();

    Diagnostic::error(DiagnosticCode::InvalidSyntax, "", span, message)
}

fn unknown_kind(key: &TomlSpanned<Cow<'_, str>>) -> Diagnostic {
    let name = key.get_ref();
    let range = key.span();
    let span = span_of(range);
    let message = format!("{name} is not a stage kind, use source, filter, mask, transform or sink");
    let path = name.to_string();
    let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

    let Some(singular) = name.strip_suffix('s') else {
        return diagnostic;
    };

    if kind_named(singular).is_none() {
        return diagnostic;
    }

    let note = format!("did you mean `{singular}`?");
    let unknown = Span::unknown();
    diagnostic.with_note(note, unknown)
}
