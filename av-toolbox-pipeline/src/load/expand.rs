use std::collections::HashSet;
use std::path::{Path, PathBuf};

use av_toolbox_graph::{
    ArgSpans,
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    FileId,
    Span,
    Spanned,
    StageSpec,
    suggest_name,
};
use serde_json::Value;

use super::files::{Files, SourceMap};
use super::resolve::{Given, resolve_variables};
use super::{LoadFailure, Loaded};
use crate::toml::{self, RawFile, RawStage, RawWorkers};
use crate::variables::{FillError, Supplied, Values, fill};

/// Reads a pipeline and every file it calls into one flat list of stages.
pub(super) struct Loader<'a> {
    files: &'a dyn Files,
    supplied: &'a Supplied,
    sources: SourceMap,
    diagnostics: Diagnostics,
    specs: Vec<StageSpec>,
}

/// One file being expanded, and where its names and paths are rooted.
struct Instance<'a> {
    file: FileId,
    directory: PathBuf,
    /// Put in front of every name in the file, such as `outer.inner.`.
    prefix: String,
    given: Given<'a>,
}

/// Everything needed to fill templates in one stage's args and workers.
struct Filling<'a> {
    arg_spans: &'a ArgSpans,
    /// The span reported when a template's own span cannot be found.
    fallback: Span,
    values: &'a Values,
    declared: &'a HashSet<&'a str>,
    path: &'a str,
}

impl<'a> Loader<'a> {
    pub(super) fn new(files: &'a dyn Files, supplied: &'a Supplied) -> Self {
        Self {
            files,
            supplied,
            sources: SourceMap::new(),
            diagnostics: Diagnostics::new(),
            specs: Vec::new(),
        }
    }

    pub(super) fn load_root(&mut self, root: &Path) {
        let read = self.files.read(root);
        let text = match read {
            Ok(text) => text,
            Err(error) => {
                let display = root.display();
                let message = format!("cannot read {display}: {error}");
                let diagnostic =
                    Diagnostic::error(DiagnosticCode::UnreadableFile, "", Span::unknown(), message);

                self.diagnostics.push(diagnostic);
                return;
            },
        };

        let directory = directory_of(root);
        let path = root.to_path_buf();
        let file = self.sources.push(path, text, Vec::new());
        let supplied = self.supplied;
        let instance = Instance {
            file,
            directory,
            prefix: String::new(),
            given: Given::Root(supplied),
        };

        self.expand(&instance);
    }

    pub(super) fn finish(self) -> Result<Loaded, LoadFailure> {
        let Self {
            sources,
            diagnostics,
            mut specs,
            ..
        } = self;

        if diagnostics.has_errors() {
            return Err(LoadFailure { diagnostics, sources });
        }

        specs.sort_by_key(|spec| (spec.span.file.index(), spec.span.start));

        Ok(Loaded { specs, sources })
    }

    /// Expands one file into specs.
    fn expand(&mut self, instance: &Instance<'_>) {
        let source = self.sources.get(instance.file).expect("the file was just added");
        let text = source.text.clone();
        let parsed = toml::parse(&text, instance.file);
        let raw = match parsed {
            Ok(raw) => raw,
            Err(found) => {
                self.diagnostics.extend(found);
                return;
            },
        };

        let values = resolve_variables(
            &raw.variables,
            &instance.given,
            &instance.directory,
            &mut self.diagnostics,
        );
        let declared: HashSet<&str> = raw
            .variables
            .iter()
            .map(|declaration| declaration.name.value.as_str())
            .collect();

        if let Given::Root(_) = instance.given {
            self.refuse_root_ports(&raw);
        }

        for stage in raw.stages {
            let converted = self.stage(instance, stage, &values, &declared);
            self.specs.extend(converted);
        }
    }

    fn refuse_root_ports(&mut self, raw: &RawFile) {
        for input in &raw.inputs {
            let path = format!("input.{}", input.name.value);
            let message = "only a called file may declare inputs";
            let diagnostic =
                Diagnostic::error(DiagnosticCode::InvalidDefinition, path, input.name.span, message);
            self.diagnostics.push(diagnostic);
        }

        for output in &raw.outputs {
            let path = format!("output.{}", output.name.value);
            let message = "only a called file may declare outputs";
            let diagnostic =
                Diagnostic::error(DiagnosticCode::InvalidDefinition, path, output.name.span, message);
            self.diagnostics.push(diagnostic);
        }
    }

    fn stage(
        &mut self,
        instance: &Instance<'_>,
        stage: RawStage,
        values: &Values,
        declared: &HashSet<&str>,
    ) -> Option<StageSpec> {
        let RawStage {
            name,
            kind,
            stage_type,
            mut inputs,
            args,
            arg_spans,
            workers,
            span,
        } = stage;

        let path = format!("{kind}.{}", name.value);

        for input in &mut inputs {
            let prefixed = format!("{}{}", instance.prefix, input.value.target);
            input.value.target = prefixed;
        }

        let args_filling = Filling {
            arg_spans: &arg_spans,
            fallback: args.span,
            values,
            declared,
            path: &path,
        };
        let filled_args = self.fill_value(&args.value, "", &args_filling);
        let workers = self.workers(workers, values, declared, &path);

        let (Some(filled_args), Ok(workers)) = (filled_args, workers) else {
            return None;
        };

        let prefixed_name = format!("{}{}", instance.prefix, name.value);
        Some(StageSpec {
            name: Spanned::new(prefixed_name, name.span),
            kind,
            stage_type,
            inputs,
            args: Spanned::new(filled_args, args.span),
            arg_spans,
            workers,
            span,
        })
    }

    /// Fills every template under `value`, or `None` after reporting each that cannot be.
    fn fill_value(&mut self, value: &Value, pointer: &str, filling: &Filling<'_>) -> Option<Value> {
        match value {
            Value::String(text) => {
                let span = filling.arg_spans.narrowest(pointer).unwrap_or(filling.fallback);
                self.fill_string(text, span, filling)
            },
            Value::Array(items) => {
                let mut filled = Vec::with_capacity(items.len());
                let mut failed = false;

                for (index, item) in items.iter().enumerate() {
                    let nested = format!("{pointer}/{index}");
                    let converted = self.fill_value(item, &nested, filling);
                    match converted {
                        Some(converted) => filled.push(converted),
                        None => failed = true,
                    }
                }

                (!failed).then_some(Value::Array(filled))
            },
            Value::Object(entries) => {
                let mut filled = serde_json::Map::new();
                let mut failed = false;

                for (key, item) in entries {
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    let nested = format!("{pointer}/{escaped}");
                    let converted = self.fill_value(item, &nested, filling);
                    match converted {
                        Some(converted) => {
                            filled.insert(key.clone(), converted);
                        },
                        None => failed = true,
                    }
                }

                (!failed).then_some(Value::Object(filled))
            },
            other => Some(other.clone()),
        }
    }

    /// Fills `text`'s templates, or `None` after reporting the problem.
    ///
    /// A template naming a variable that was declared but could not be resolved is left
    /// unreported here, since resolving it already reported the problem.
    fn fill_string(&mut self, text: &str, span: Span, filling: &Filling<'_>) -> Option<Value> {
        let filled = fill(text, filling.values);

        match filled {
            Ok(value) => Some(value),
            Err(FillError::Unknown(name)) => {
                if filling.declared.contains(name.as_str()) {
                    return None;
                }

                let message = format!("no variable named {name} is declared in this file");
                let mut diagnostic =
                    Diagnostic::error(DiagnosticCode::UnknownVariable, filling.path, span, message);

                let closest = suggest_name(&name, filling.declared.iter().copied());
                if let Some(closest) = closest {
                    let note = format!("did you mean `{closest}`?");
                    diagnostic = diagnostic.with_note(note, Span::unknown());
                }

                self.diagnostics.push(diagnostic);
                None
            },
            Err(FillError::Template(problem)) => {
                let message = problem.message();
                let diagnostic =
                    Diagnostic::error(DiagnosticCode::InvalidDefinition, filling.path, span, message);

                self.diagnostics.push(diagnostic);
                None
            },
        }
    }

    fn workers(
        &mut self,
        workers: Option<Spanned<RawWorkers>>,
        values: &Values,
        declared: &HashSet<&str>,
        path: &str,
    ) -> Result<Option<usize>, ()> {
        let Some(workers) = workers else {
            return Ok(None);
        };

        let template = match workers.value {
            RawWorkers::Count(count) => return Ok(Some(count)),
            RawWorkers::Template(template) => template,
        };

        let workers_path = format!("{path}.workers");
        let no_spans = ArgSpans::new();
        let workers_filling = Filling {
            arg_spans: &no_spans,
            fallback: workers.span,
            values,
            declared,
            path: &workers_path,
        };
        let filled = self.fill_string(&template, workers.span, &workers_filling);
        let Some(filled) = filled else {
            return Err(());
        };

        let count = filled.as_u64().and_then(|count| usize::try_from(count).ok());
        match count {
            Some(count) if count > 0 => Ok(Some(count)),
            _ => {
                let message =
                    format!("workers must be a whole number of at least 1, the template gave {filled}");
                let diagnostic = Diagnostic::error(
                    DiagnosticCode::InvalidDefinition,
                    workers_path,
                    workers.span,
                    message,
                );

                self.diagnostics.push(diagnostic);
                Err(())
            },
        }
    }
}

/// The directory relative paths in `path` resolve against.
///
/// A bare file name has an empty parent, which resolves against the working directory.
fn directory_of(path: &Path) -> PathBuf {
    let parent = path.parent();
    parent.map(Path::to_path_buf).unwrap_or_default()
}
