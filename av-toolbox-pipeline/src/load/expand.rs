use std::collections::{HashMap, HashSet};
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
use serde_json::{Map, Value};

use super::aliases::{drop_unread, follow};
use super::calls::{CallSite, Link};
use super::files::{Files, SourceMap};
use super::fill::{FillFailure, fill_template, narrowed};
use super::resolve::{Given, resolve_variables};
use super::{LoadFailure, Loaded};
use crate::toml::{self, RawFile, RawStage, RawWorkers};
use crate::variables::{Declaration, ResolvedVariable, Supplied, Values};

/// Reads a pipeline and every file it calls into one flat list of stages.
pub(super) struct Loader<'a> {
    pub(super) files: &'a dyn Files,
    supplied: &'a Supplied,
    pub(super) sources: SourceMap,
    pub(super) diagnostics: Diagnostics,
    specs: Vec<StageSpec>,
    /// The root file's variables, once it has been expanded.
    variables: Vec<ResolvedVariable>,
    /// Input, output and call names mapped to the name they stand for.
    pub(super) aliases: HashMap<String, String>,
    /// The aliases which are outputs of a called file.
    pub(super) outputs: HashSet<String>,
}

/// One file being expanded, and where its names and paths are rooted.
pub(super) struct Instance<'a> {
    pub(super) file: FileId,
    pub(super) directory: PathBuf,
    /// Put in front of every name in the file, such as `outer.inner.`.
    pub(super) prefix: String,
    pub(super) given: Given<'a>,
    /// This file and every file that called it on the way here.
    pub(super) chain: Vec<Link>,
    /// The call that named this file, or `None` for the root file.
    pub(super) call: Option<CallSite>,
}

/// What filling templates needs, and the path problems are reported under.
pub(super) struct Filling<'a> {
    pub(super) values: &'a Values,
    /// The file's declared variable names, in the order they are declared.
    pub(super) declared: &'a [&'a str],
    pub(super) path: &'a str,
}

impl<'a> Loader<'a> {
    pub(super) fn new(files: &'a dyn Files, supplied: &'a Supplied) -> Self {
        Self {
            files,
            supplied,
            sources: SourceMap::new(),
            diagnostics: Diagnostics::new(),
            specs: Vec::new(),
            variables: Vec::new(),
            aliases: HashMap::new(),
            outputs: HashSet::new(),
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
        let found = self.files.canonicalize(root);
        let canonical = found.unwrap_or_else(|_| path.clone());
        let file = self.sources.push(path, text, Vec::new());
        let supplied = self.supplied;
        let root_link = Link {
            canonical,
            entered_by: None,
        };
        let instance = Instance {
            file,
            directory,
            prefix: String::new(),
            given: Given::Root(supplied),
            chain: vec![root_link],
            call: None,
        };

        self.expand(&instance);
    }

    pub(super) fn finish(self) -> Result<Loaded, LoadFailure> {
        let Self {
            sources,
            diagnostics,
            mut specs,
            variables,
            aliases,
            outputs,
            ..
        } = self;

        if diagnostics.has_errors() {
            return Err(LoadFailure { diagnostics, sources });
        }

        let mut read_outputs: HashSet<String> = HashSet::new();
        let mut resolve_diagnostics = Diagnostics::new();

        for spec in &mut specs {
            for input in &mut spec.inputs {
                let resolved = follow(&aliases, &outputs, &input.value.target, &mut read_outputs);
                match resolved {
                    Some(target) => input.value.target = target,
                    None => {
                        let message = format!("{} loops back to itself through calls", input.value.target);
                        let diagnostic = Diagnostic::error(DiagnosticCode::Cycle, "", input.span, message);
                        resolve_diagnostics.push(diagnostic);
                    },
                }
            }
        }

        if !resolve_diagnostics.is_empty() {
            return Err(LoadFailure {
                diagnostics: resolve_diagnostics,
                sources,
            });
        }

        drop_unread(&mut specs, &aliases, &outputs, &read_outputs);

        specs.sort_by_key(|spec| (spec.span.file.index(), spec.span.start));

        Ok(Loaded {
            specs,
            sources,
            variables,
        })
    }

    /// Expands one file into specs, returning its output names.
    ///
    /// Returns `None` when the file could not be parsed.
    pub(super) fn expand(&mut self, instance: &Instance<'_>) -> Option<Vec<String>> {
        let source = self.sources.get(instance.file).expect("the file was just added");
        let text = source.text.clone();
        let parsed = toml::parse(&text, instance.file);
        let raw = match parsed {
            Ok(raw) => raw,
            Err(found) => {
                self.diagnostics.extend(found);
                return None;
            },
        };

        let values = resolve_variables(
            &raw.variables,
            &instance.given,
            &instance.directory,
            &mut self.diagnostics,
        );
        let declared: Vec<&str> = raw
            .variables
            .iter()
            .map(|declaration| declaration.name.value.as_str())
            .collect();

        if instance.call.is_none() {
            self.variables = resolved_variables(&raw.variables, &values);
        }

        match &instance.call {
            None => self.refuse_root_ports(&raw),
            Some(site) => self.bind_inputs(instance, site, &raw),
        }

        self.check_names(&raw);

        let mut outputs_of: HashMap<String, Option<Vec<String>>> = HashMap::new();
        for call in &raw.calls {
            let expanded = self.call(instance, call, &values, &declared);
            outputs_of.insert(call.name.value.clone(), expanded);
        }

        self.check_references(&raw, &outputs_of);
        self.check_targets(&raw, &outputs_of);

        for stage in raw.stages {
            let converted = self.stage(instance, stage, &values, &declared);
            self.specs.extend(converted);
        }

        let mut output_names = Vec::with_capacity(raw.outputs.len());
        for output in raw.outputs {
            let alias = format!("{}{}", instance.prefix, output.name.value);
            let target = format!("{}{}", instance.prefix, output.source.value);

            self.aliases.insert(alias.clone(), target);
            self.outputs.insert(alias);
            output_names.push(output.name.value);
        }

        Some(output_names)
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
        declared: &[&str],
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
            values,
            declared,
            path: &path,
        };
        let filled_args = self.fill_value(&args.value, "", &arg_spans, args.span, &args_filling);
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
    ///
    /// A template is reported at its span in `arg_spans`, or at `fallback` when it has none.
    pub(super) fn fill_value(
        &mut self,
        value: &Value,
        pointer: &str,
        arg_spans: &ArgSpans,
        fallback: Span,
        filling: &Filling<'_>,
    ) -> Option<Value> {
        match value {
            Value::String(text) => {
                let span = arg_spans.narrowest(pointer).unwrap_or(fallback);
                self.fill_string(text, span, filling)
            },
            Value::Array(items) => {
                let mut filled = Vec::with_capacity(items.len());
                let mut failed = false;

                for (index, item) in items.iter().enumerate() {
                    let nested = format!("{pointer}/{index}");
                    let converted = self.fill_value(item, &nested, arg_spans, fallback, filling);
                    match converted {
                        Some(converted) => filled.push(converted),
                        None => failed = true,
                    }
                }

                (!failed).then_some(Value::Array(filled))
            },
            Value::Object(entries) => {
                let mut filled = Map::new();
                let mut failed = false;

                for (key, item) in entries {
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    let nested = format!("{pointer}/{escaped}");
                    let converted = self.fill_value(item, &nested, arg_spans, fallback, filling);
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
    pub(super) fn fill_string(&mut self, text: &str, span: Span, filling: &Filling<'_>) -> Option<Value> {
        let filled = fill_template(text, filling.values);

        match filled {
            Ok(value) => Some(value),
            Err(FillFailure::Unknown { name }) => {
                if filling.declared.contains(&name.as_str()) {
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
            Err(FillFailure::Invalid { message, range }) => {
                let narrow = narrowed(&self.sources, text, span, &range);
                let diagnostic =
                    Diagnostic::error(DiagnosticCode::InvalidDefinition, filling.path, narrow, message);

                self.diagnostics.push(diagnostic);
                None
            },
        }
    }

    fn workers(
        &mut self,
        workers: Option<Spanned<RawWorkers>>,
        values: &Values,
        declared: &[&str],
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
        let workers_filling = Filling {
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
pub(super) fn directory_of(path: &Path) -> PathBuf {
    let parent = path.parent();
    parent.map(Path::to_path_buf).unwrap_or_default()
}

/// Pairs each declaration with the value it resolved to, skipping those that did not resolve.
fn resolved_variables(declarations: &[Declaration], values: &Values) -> Vec<ResolvedVariable> {
    let mut resolved = Vec::new();

    for declaration in declarations {
        let name = &declaration.name.value;
        let Some(value) = values.get(name) else {
            continue;
        };

        let variable = ResolvedVariable {
            name: name.clone(),
            variable_type: declaration.variable_type,
            value: value.clone(),
        };
        resolved.push(variable);
    }

    resolved
}
