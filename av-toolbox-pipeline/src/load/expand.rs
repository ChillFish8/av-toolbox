use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, FileId, Span, Spanned, StageSpec};

use super::aliases::{drop_unread, follow};
use super::calls::{CallSite, Link};
use super::deferred::follow_references;
use super::files::{Files, SourceMap};
use super::fill::Filling;
use super::resolve::{Given, resolve_variables};
use super::{LoadFailure, Loaded};
use crate::toml::{self, RawFile, RawStage, RawWorkers};
use crate::variables::{Bound, Declaration, ResolvedVariable, Supplied, Values};

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

            follow_references(
                spec,
                &aliases,
                &outputs,
                &mut read_outputs,
                &mut resolve_diagnostics,
            );
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
            prefix: &instance.prefix,
            path: &path,
        };
        let mut deferred = Vec::new();
        let filled_args = self.fill_value(
            &args.value,
            "",
            &arg_spans,
            args.span,
            &args_filling,
            &mut deferred,
        );
        let workers = self.workers(workers, &args_filling);

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
            deferred,
            span,
        })
    }

    /// The stage's worker count, filling a template with `filling` under the `workers` path.
    fn workers(
        &mut self,
        workers: Option<Spanned<RawWorkers>>,
        filling: &Filling<'_>,
    ) -> Result<Option<usize>, ()> {
        let Some(workers) = workers else {
            return Ok(None);
        };

        let template = match workers.value {
            RawWorkers::Count(count) => return Ok(Some(count)),
            RawWorkers::Template(template) => template,
        };

        let workers_path = format!("{}.workers", filling.path);
        let workers_filling = Filling {
            path: &workers_path,
            ..*filling
        };
        let filled = self.fill_now(&template, workers.span, &workers_filling);
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
        let Some(Bound::Ready(value)) = values.get(name) else {
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
