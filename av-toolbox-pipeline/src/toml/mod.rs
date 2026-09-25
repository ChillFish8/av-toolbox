mod args;
mod call;
mod reader;
mod stage;
#[cfg(test)]
mod tests;
mod variable;

use std::borrow::Cow;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeTable, DeValue};
use av_toolbox_core::StageKind;
use av_toolbox_graph::{
    ArgSpans,
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    FileId,
    InputRef,
    Span,
    Spanned,
    StageSpec,
};
use serde_json::Value;

use self::call::{read_call, read_input, read_output};
use self::stage::read_stage;
use self::variable::read_variable;
use crate::variables::Declaration;

/// One pipeline file, with its templates left unfilled.
#[derive(Debug, Clone, Default)]
pub struct RawFile {
    pub variables: Vec<Declaration>,
    pub stages: Vec<RawStage>,
    pub calls: Vec<RawCall>,
    pub inputs: Vec<RawInput>,
    pub outputs: Vec<RawOutput>,
}

#[derive(Debug, Clone)]
pub struct RawStage {
    pub name: Spanned<String>,
    pub kind: StageKind,
    pub stage_type: Spanned<String>,
    pub inputs: Vec<Spanned<InputRef>>,
    pub args: Spanned<Value>,
    pub arg_spans: ArgSpans,
    pub workers: Option<Spanned<RawWorkers>>,
    pub span: Span,
}

/// How many workers a stage runs, written either as a count or a template.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum RawWorkers {
    Count(usize),
    Template(String),
}

#[derive(Debug, Clone)]
pub struct RawCall {
    pub name: Spanned<String>,
    pub path: Spanned<String>,
    pub inputs: Vec<Spanned<InputRef>>,
    /// The span of the `inputs` value, or `None` when the call has no `inputs`.
    pub inputs_span: Option<Span>,
    pub variables: Vec<RawCallVariable>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct RawCallVariable {
    /// The variable's name, spanning its key.
    pub name: Spanned<String>,
    pub value: Spanned<Value>,
}

#[derive(Debug, Clone)]
pub struct RawInput {
    pub name: Spanned<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RawOutput {
    pub name: Spanned<String>,
    /// The one stage or input this output reads, spanning its array element.
    pub source: Spanned<String>,
    pub description: Option<String>,
}

/// The tables a pipeline file may hold.
#[derive(Debug, Copy, Clone)]
enum Table {
    Stage(StageKind),
    Variable,
    Call,
    Input,
    Output,
}

const TABLES: [(&str, Table); 9] = [
    ("source", Table::Stage(StageKind::Source)),
    ("filter", Table::Stage(StageKind::Filter)),
    ("mask", Table::Stage(StageKind::Mask)),
    ("transform", Table::Stage(StageKind::Transform)),
    ("sink", Table::Stage(StageKind::Sink)),
    ("variable", Table::Variable),
    ("call", Table::Call),
    ("input", Table::Input),
    ("output", Table::Output),
];

/// Reads one pipeline file, with its templates left unfilled.
///
/// Every problem in the file is reported, not only the first. Each list keeps the order its
/// entries start in the text, and every span points into `file`.
pub fn parse(text: &str, file: FileId) -> Result<RawFile, Diagnostics> {
    let (document, errors) = DeTable::parse_recoverable(text);
    if !errors.is_empty() {
        let reported: Vec<Diagnostic> = errors.iter().map(|error| syntax_error(error, file)).collect();
        return Err(Diagnostics::from(reported));
    }

    let mut diagnostics = Diagnostics::new();
    let mut raw = RawFile::default();

    let table = document.get_ref();
    for (key, value) in table.iter() {
        let name = key.get_ref();
        let Some(kind) = table_named(name) else {
            let diagnostic = unknown_table(key, file);
            diagnostics.push(diagnostic);
            continue;
        };

        let inner = value.get_ref();
        let Some(entries) = inner.as_table() else {
            let range = value.span();
            let span = Span::in_file(file, range.start, range.end);
            let message = format!("{name} must be a table, such as [{name}.name]");
            let path = name.to_string();
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);
            diagnostics.push(diagnostic);
            continue;
        };

        for (entry_name, entry) in entries.iter() {
            read_entry(kind, entry_name, entry, file, &mut raw, &mut diagnostics);
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    raw.stages.sort_by_key(|stage| stage.span.start);
    raw.variables.sort_by_key(|variable| variable.name.span.start);
    raw.calls.sort_by_key(|call| call.span.start);
    raw.inputs.sort_by_key(|input| input.name.span.start);
    raw.outputs.sort_by_key(|output| output.name.span.start);

    Ok(raw)
}

fn read_entry(
    kind: Table,
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    raw: &mut RawFile,
    diagnostics: &mut Diagnostics,
) {
    match kind {
        Table::Stage(stage_kind) => {
            let read = read_stage(stage_kind, name, value, file, diagnostics);
            raw.stages.extend(read);
        },
        Table::Variable => {
            let read = read_variable(name, value, file, diagnostics);
            raw.variables.extend(read);
        },
        Table::Call => {
            let read = read_call(name, value, file, diagnostics);
            raw.calls.extend(read);
        },
        Table::Input => {
            let read = read_input(name, value, file, diagnostics);
            raw.inputs.extend(read);
        },
        Table::Output => {
            let read = read_output(name, value, file, diagnostics);
            raw.outputs.extend(read);
        },
    }
}

/// Reads a pipeline file that uses no variables, calls, inputs or outputs.
pub fn parse_specs(text: &str) -> Result<Vec<StageSpec>, Diagnostics> {
    let raw = parse(text, FileId::ROOT)?;
    let mut specs = Vec::with_capacity(raw.stages.len());

    for stage in raw.stages {
        let workers = match stage.workers.map(|workers| workers.value) {
            Some(RawWorkers::Count(count)) => Some(count),
            _ => None,
        };

        specs.push(StageSpec {
            name: stage.name,
            kind: stage.kind,
            stage_type: stage.stage_type,
            inputs: stage.inputs,
            args: stage.args,
            arg_spans: stage.arg_spans,
            workers,
            span: stage.span,
        });
    }

    Ok(specs)
}

fn table_named(name: &str) -> Option<Table> {
    let found = TABLES.iter().find(|(table, _)| *table == name);
    found.map(|(_, table)| *table)
}

fn syntax_error(error: &::toml::de::Error, file: FileId) -> Diagnostic {
    let unknown = Span::unknown();
    let span = error
        .span()
        .map(|range| Span::in_file(file, range.start, range.end))
        .unwrap_or(unknown);
    let message = error.message();

    Diagnostic::error(DiagnosticCode::InvalidSyntax, "", span, message)
}

fn unknown_table(key: &TomlSpanned<Cow<'_, str>>, file: FileId) -> Diagnostic {
    let name = key.get_ref();
    let range = key.span();
    let span = Span::in_file(file, range.start, range.end);
    let allowed = "use source, filter, mask, transform, sink, variable, call, input or output";
    let message = format!("{name} is not a table a pipeline holds, {allowed}");
    let path = name.to_string();
    let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

    let Some(singular) = name.strip_suffix('s') else {
        return diagnostic;
    };

    if table_named(singular).is_none() {
        return diagnostic;
    }

    let note = format!("did you mean `{singular}`?");
    let unknown = Span::unknown();
    diagnostic.with_note(note, unknown)
}
