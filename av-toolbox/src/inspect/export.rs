use std::collections::BTreeMap;

use av_toolbox_graph::{Diagnostic, InputRef, Note, Severity, Span, StageSpec};
use av_toolbox_pipeline::variables::ResolvedVariable;
use serde::Serialize;
use serde_json::Value;

use crate::profile::file::stage_key;
use crate::report::PipelineFiles;

/// The version of the export's shape and of the hash input.
pub(crate) const FORMAT_VERSION: u32 = 1;

/// Everything `inspect` prints about a pipeline.
#[derive(Debug, Serialize)]
pub(crate) struct Export {
    pub(crate) format_version: u32,
    pub(crate) valid: bool,
    pub(crate) pipeline: String,
    pub(crate) hash: Option<String>,
    pub(crate) variables: Option<BTreeMap<String, ExportedVariable>>,
    pub(crate) stages: Option<Vec<ExportedStage>>,
    pub(crate) diagnostics: Vec<ExportedDiagnostic>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedVariable {
    #[serde(rename = "type")]
    pub(crate) variable_type: &'static str,
    pub(crate) value: Value,
}

/// One stage after variables are filled in and calls are expanded.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedStage {
    /// The stage's `<kind>.<name>`, as profile files name it.
    pub(crate) key: String,
    pub(crate) name: String,
    pub(crate) kind: String,
    #[serde(rename = "type")]
    pub(crate) stage_type: String,
    pub(crate) inputs: Vec<ExportedInput>,
    pub(crate) args: Value,
    pub(crate) workers: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedInput {
    /// The input's name, or `None` when it was given positionally.
    pub(crate) name: Option<String>,
    pub(crate) stage: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedDiagnostic {
    pub(crate) severity: String,
    pub(crate) code: &'static str,
    pub(crate) message: String,
    #[serde(flatten)]
    pub(crate) location: ExportedLocation,
    pub(crate) notes: Vec<ExportedNote>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ExportedNote {
    pub(crate) message: String,
    #[serde(flatten)]
    pub(crate) location: ExportedLocation,
}

/// Where a diagnostic or note points, with every field `None` when its span is unknown.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub(crate) struct ExportedLocation {
    pub(crate) file: Option<String>,
    pub(crate) line: Option<usize>,
    pub(crate) column: Option<usize>,
}

pub(crate) fn variables(resolved: &[ResolvedVariable]) -> BTreeMap<String, ExportedVariable> {
    let mut exported = BTreeMap::new();

    for variable in resolved {
        let value = variable.value.to_json();
        let entry = ExportedVariable {
            variable_type: variable.variable_type.as_str(),
            value,
        };
        exported.insert(variable.name.clone(), entry);
    }

    exported
}

pub(crate) fn stages(specs: &[StageSpec]) -> Vec<ExportedStage> {
    specs.iter().map(stage).collect()
}

fn stage(spec: &StageSpec) -> ExportedStage {
    let inputs = spec
        .inputs
        .iter()
        .map(|reference| input(&reference.value))
        .collect();

    ExportedStage {
        key: stage_key(spec),
        name: spec.name.value.clone(),
        kind: spec.kind.to_string(),
        stage_type: spec.stage_type.value.clone(),
        inputs,
        args: spec.args.value.clone(),
        workers: spec.workers,
    }
}

fn input(reference: &InputRef) -> ExportedInput {
    ExportedInput {
        name: reference.name.clone(),
        stage: reference.target.clone(),
    }
}

/// Every diagnostic, errors first, with a note at each call that led to its file.
pub(crate) fn diagnostics(files: &PipelineFiles, found: &[Diagnostic]) -> Vec<ExportedDiagnostic> {
    let errors = found
        .iter()
        .filter(|diagnostic| diagnostic.severity() == Severity::Error);
    let warnings = found
        .iter()
        .filter(|diagnostic| diagnostic.severity() == Severity::Warning);

    let mut exported = Vec::new();
    for diagnostic in errors.chain(warnings) {
        let noted = files.with_call_notes(diagnostic.clone());
        let converted = exported_diagnostic(files, &noted);
        exported.push(converted);
    }

    exported
}

fn exported_diagnostic(files: &PipelineFiles, diagnostic: &Diagnostic) -> ExportedDiagnostic {
    let span = diagnostic.span();
    let location = location(files, span);
    let notes = diagnostic
        .notes()
        .iter()
        .map(|note| exported_note(files, note))
        .collect();

    ExportedDiagnostic {
        severity: diagnostic.severity().to_string(),
        code: diagnostic.code().as_str(),
        message: diagnostic.message().to_string(),
        location,
        notes,
    }
}

fn exported_note(files: &PipelineFiles, note: &Note) -> ExportedNote {
    let location = location(files, note.span);

    ExportedNote {
        message: note.message.clone(),
        location,
    }
}

fn location(files: &PipelineFiles, span: Span) -> ExportedLocation {
    let Some(located) = files.locate(span) else {
        return ExportedLocation::default();
    };

    ExportedLocation {
        file: Some(located.file),
        line: Some(located.line),
        column: Some(located.column),
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use av_toolbox_graph::{
        ArgSpans,
        Diagnostic,
        DiagnosticCode,
        FileId,
        InputRef,
        Span,
        Spanned,
        StageSpec,
    };
    use av_toolbox_pipeline::SourceMap;
    use av_toolbox_pipeline::variables::{ResolvedVariable, VariableType, VariableValue};
    use serde_json::json;

    use super::*;
    use crate::report::PipelineFiles;

    fn input(name: Option<&str>, target: &str) -> Spanned<InputRef> {
        let reference = InputRef {
            name: name.map(str::to_string),
            target: target.to_string(),
        };

        Spanned::unknown(reference)
    }

    fn spec(kind: StageKind, name: &str, inputs: Vec<Spanned<InputRef>>) -> StageSpec {
        StageSpec {
            name: Spanned::unknown(name.to_string()),
            kind,
            stage_type: Spanned::unknown("test/stage".to_string()),
            inputs,
            args: Spanned::unknown(json!({ "sigma": 1.5 })),
            arg_spans: ArgSpans::default(),
            workers: Some(2),
            span: Span::unknown(),
        }
    }

    #[test]
    fn a_stage_exports_its_key_type_args_and_workers() {
        let specs = [spec(StageKind::Filter, "scenes.denoise", Vec::new())];

        let exported = stages(&specs);

        assert_eq!(exported[0].key, "filter.scenes.denoise");
        assert_eq!(exported[0].name, "scenes.denoise");
        assert_eq!(exported[0].kind, "filter");
        assert_eq!(exported[0].stage_type, "test/stage");
        assert_eq!(exported[0].args, json!({ "sigma": 1.5 }));
        assert_eq!(exported[0].workers, Some(2));
    }

    #[test]
    fn inputs_keep_their_order_and_names() {
        let inputs = vec![input(None, "clip"), input(Some("mask"), "lines")];
        let specs = [spec(StageKind::Filter, "blend", inputs)];

        let exported = stages(&specs);

        let expected = vec![
            ExportedInput {
                name: None,
                stage: "clip".to_string(),
            },
            ExportedInput {
                name: Some("mask".to_string()),
                stage: "lines".to_string(),
            },
        ];
        assert_eq!(exported[0].inputs, expected);
    }

    #[test]
    fn a_stage_serialises_its_type_under_type() {
        let specs = [spec(StageKind::Sink, "out", Vec::new())];

        let exported = stages(&specs);
        let value = serde_json::to_value(&exported[0]).expect("plain JSON");

        assert_eq!(value["type"], json!("test/stage"));
        assert!(value.get("stage_type").is_none());
    }

    #[test]
    fn variables_export_their_type_and_value() {
        let resolved = [
            ResolvedVariable {
                name: "width".to_string(),
                variable_type: VariableType::Integer,
                value: VariableValue::Integer(1280),
            },
            ResolvedVariable {
                name: "input".to_string(),
                variable_type: VariableType::Path,
                value: VariableValue::Path("/clips/in.mkv".to_string()),
            },
        ];

        let exported = variables(&resolved);
        let value = serde_json::to_value(&exported).expect("plain JSON");

        let expected = json!({
            "input": { "type": "path", "value": "/clips/in.mkv" },
            "width": { "type": "integer", "value": 1280 },
        });
        assert_eq!(value, expected);
    }

    const ROOT_TEXT: &str = "[filter.blur]\ntype = \"test/blur\"\n";

    fn root_files() -> PipelineFiles {
        let mut sources = SourceMap::new();
        sources.push("pipeline.toml".into(), ROOT_TEXT.to_string(), Vec::new());

        PipelineFiles::new(sources)
    }

    #[test]
    fn errors_export_before_warnings_with_their_location() {
        let warning = Diagnostic::warning(DiagnosticCode::UnusedStage, "", Span::new(1, 7), "unused");
        let error = Diagnostic::error(DiagnosticCode::UnknownStage, "", Span::new(14, 18), "unknown");
        let found = [warning, error];

        let exported = diagnostics(&root_files(), &found);

        assert_eq!(exported[0].severity, "error");
        assert_eq!(exported[0].code, "unknown-stage");
        assert_eq!(exported[0].message, "unknown");
        assert_eq!(exported[0].location.file.as_deref(), Some("pipeline.toml"));
        assert_eq!(exported[0].location.line, Some(2));
        assert_eq!(exported[0].location.column, Some(1));
        assert_eq!(exported[1].severity, "warning");
        assert_eq!(exported[1].location.column, Some(2));
    }

    #[test]
    fn an_unknown_span_exports_null_locations() {
        let error = Diagnostic::error(DiagnosticCode::UnreadableFile, "", Span::unknown(), "cannot read");

        let exported = diagnostics(&root_files(), &[error]);
        let value = serde_json::to_value(&exported[0]).expect("plain JSON");

        assert_eq!(value["file"], json!(null));
        assert_eq!(value["line"], json!(null));
        assert_eq!(value["column"], json!(null));
    }

    #[test]
    fn a_diagnostic_in_a_called_file_notes_the_call_with_its_location() {
        let mut sources = SourceMap::new();
        let root = sources.push("main.toml".into(), ROOT_TEXT.to_string(), Vec::new());
        let call = Span::in_file(root, 14, 18);
        let part = sources.push("part.toml".into(), ROOT_TEXT.to_string(), vec![call]);
        let files = PipelineFiles::new(sources);
        let span = Span::in_file(part, 1, 7);
        let error = Diagnostic::error(DiagnosticCode::UnknownStage, "", span, "unknown");

        let exported = diagnostics(&files, &[error]);

        assert_eq!(exported[0].location.file.as_deref(), Some("part.toml"));
        assert_eq!(exported[0].notes[0].message, "called from here");
        assert_eq!(exported[0].notes[0].location.file.as_deref(), Some("main.toml"));
        assert_eq!(exported[0].notes[0].location.line, Some(2));
        assert_eq!(root, FileId::ROOT);
    }
}
