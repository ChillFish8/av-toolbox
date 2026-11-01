use std::collections::{HashMap, HashSet};
use std::io::Error;
use std::path::{Path, PathBuf};

use av_toolbox_graph::{
    ArgSpans,
    Diagnostic,
    DiagnosticCode,
    Expr,
    InputRef,
    Piece,
    Span,
    Spanned,
    Template,
};
use serde_json::Value;

use super::expand::{Instance, Loader, directory_of};
use super::fill::{Filled, Filling};
use super::resolve::{Given, GivenValue, Passed};
use crate::toml::{RawCall, RawFile};
use crate::variables::Values;

/// A file being expanded because a call named it.
pub(super) struct CallSite {
    /// The call's header in the calling file.
    header: Span,
    /// The path as the call wrote it, for messages.
    written_path: String,
    /// The prefix of the calling file.
    caller_prefix: String,
    inputs: Vec<Spanned<InputRef>>,
    inputs_span: Option<Span>,
}

/// One file on the chain of calls, and the call header that entered it.
#[derive(Clone)]
pub(super) struct Link {
    pub(super) canonical: PathBuf,
    pub(super) entered_by: Option<Span>,
}

impl Loader<'_> {
    /// Expands one call, returning the called file's output names.
    ///
    /// Returns `None` when the path template cannot be filled, the call loops back to a file
    /// already being called, or the called file cannot be read or parsed.
    pub(super) fn call(
        &mut self,
        instance: &Instance<'_>,
        call: &RawCall,
        values: &Values,
        declared: &[&str],
    ) -> Option<Vec<String>> {
        let call_path = format!("call.{}", call.name.value);
        let path_filling = Filling {
            values,
            declared,
            prefix: &instance.prefix,
            path: &call_path,
            directory: &instance.directory,
        };
        let filled = self.fill_now(&call.path.value, call.path.span, &path_filling)?;
        let Value::String(written) = filled else {
            let message = "path must be text, such as \"parts/denoise.toml\"";
            let diagnostic = Diagnostic::error(
                DiagnosticCode::InvalidDefinition,
                call_path,
                call.path.span,
                message,
            );
            self.diagnostics.push(diagnostic);
            return None;
        };

        let path = instance.directory.join(&written);
        let canonical = self.canonical(&path, call.path.span)?;

        let looped = instance.chain.iter().position(|link| link.canonical == canonical);
        if let Some(start) = looped {
            self.report_cycle(&instance.chain[start..], call, &written);
            return None;
        }

        let text = self.read(&path, call.path.span)?;

        let caller = self
            .sources
            .get(instance.file)
            .expect("the calling file is known");
        let mut called_from = vec![call.span];
        called_from.extend(caller.called_from.iter().copied());

        let file = self.sources.push(path.clone(), text, called_from);
        let given_values = self.given_values(instance, call, values, declared);

        let mut chain = instance.chain.clone();
        chain.push(Link {
            canonical,
            entered_by: Some(call.span),
        });

        let site = CallSite {
            header: call.span,
            written_path: written,
            caller_prefix: instance.prefix.clone(),
            inputs: call.inputs.clone(),
            inputs_span: call.inputs_span,
        };
        let child = Instance {
            file,
            directory: directory_of(&path),
            prefix: format!("{}{}.", instance.prefix, call.name.value),
            given: Given::Call(&given_values),
            chain,
            call: Some(site),
        };

        let outputs = self.expand(&child)?;

        if let [only] = outputs.as_slice() {
            let alias = format!("{}{}", instance.prefix, call.name.value);
            let target = format!("{}{}.{only}", instance.prefix, call.name.value);
            self.aliases.insert(alias, target);
        }

        Some(outputs)
    }

    fn canonical(&mut self, path: &Path, span: Span) -> Option<PathBuf> {
        let found = self.files.canonicalize(path);
        match found {
            Ok(canonical) => Some(canonical),
            Err(error) => {
                self.unreadable(path, span, &error);
                None
            },
        }
    }

    fn read(&mut self, path: &Path, span: Span) -> Option<String> {
        let read = self.files.read(path);
        match read {
            Ok(text) => Some(text),
            Err(error) => {
                self.unreadable(path, span, &error);
                None
            },
        }
    }

    fn unreadable(&mut self, path: &Path, span: Span, error: &Error) {
        let display = path.display();
        let message = format!("cannot read {display}: {error}");
        let diagnostic = Diagnostic::error(DiagnosticCode::UnreadableFile, "", span, message);

        self.diagnostics.push(diagnostic);
    }

    /// Reports a call closing a loop, with a note at each other call in the loop.
    ///
    /// `looped` starts at the file being called again. The call that first entered it sits
    /// outside the loop, so it gets no note.
    fn report_cycle(&mut self, looped: &[Link], call: &RawCall, written: &str) {
        let message = format!("calling {written} here loops back to a file already being called");
        let mut diagnostic = Diagnostic::error(DiagnosticCode::CallCycle, "", call.span, message);

        let entries = looped.iter().skip(1).filter_map(|link| link.entered_by);
        for entered_by in entries {
            diagnostic = diagnostic.with_note("part of the loop", entered_by);
        }

        self.diagnostics.push(diagnostic);
    }

    /// The call's `variables`, with the caller's templates filled in.
    fn given_values(
        &mut self,
        instance: &Instance<'_>,
        call: &RawCall,
        values: &Values,
        declared: &[&str],
    ) -> Vec<GivenValue> {
        let mut given = Vec::with_capacity(call.variables.len());
        let path = format!("call.{}.variables", call.name.value);
        let filling = Filling {
            values,
            declared,
            prefix: &instance.prefix,
            path: &path,
            directory: &instance.directory,
        };

        for variable in &call.variables {
            let span = variable.value.span;
            let passed = self.passed(&variable.value.value, span, &filling);

            given.push(GivenValue {
                name: variable.name.value.clone(),
                name_span: variable.name.span,
                value: passed,
                value_span: span,
                directory: instance.directory.clone(),
            });
        }

        given
    }

    /// One `variables` entry with the caller's templates filled, or `None` after reporting.
    ///
    /// A string reading the caller's stages is passed on as its expression.
    fn passed(&mut self, value: &Value, span: Span, filling: &Filling<'_>) -> Option<Passed> {
        let Value::String(text) = value else {
            let no_spans = ArgSpans::new();
            let mut ignored = Vec::new();
            let filled = self.fill_value(value, "", &no_spans, span, filling, &mut ignored)?;

            return Some(Passed::Value(filled));
        };

        let filled = self.fill_string(text, span, filling)?;
        match filled {
            Filled::Ready(ready) => Some(Passed::Value(ready)),
            Filled::Deferred(template) => {
                let expr = self.whole_expression(&template, span, filling.path)?;
                Some(Passed::Reference(expr))
            },
        }
    }

    /// The expression `template` is made of, or `None` after reporting text around it.
    fn whole_expression(&mut self, template: &Template, span: Span, path: &str) -> Option<Expr> {
        if let [Piece::Expr(only)] = template.pieces() {
            return Some(only.expr.clone());
        }

        let message =
            "a variable given a stage reference must hold only the template, such as \"{{ src.width }}\"";
        let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

        self.diagnostics.push(diagnostic);
        None
    }

    /// Makes each of the called file's inputs an alias of the stage its call passed.
    pub(super) fn bind_inputs(&mut self, instance: &Instance<'_>, site: &CallSite, raw: &RawFile) {
        let slots: Vec<&str> = raw.inputs.iter().map(|input| input.name.value.as_str()).collect();
        let fallback = site.inputs_span.unwrap_or(site.header);
        let mut bound: HashSet<String> = HashSet::new();
        let listed = slots.join(", ");

        let is_list = site.inputs.iter().any(|input| input.value.name.is_none());
        if is_list && site.inputs.len() > 1 {
            let message = "a list of inputs may hold only one stage, name each input instead";
            self.invalid_inputs(fallback, message.to_string());
            return;
        }

        if is_list && slots.len() != 1 {
            let count = slots.len();
            let message = format!(
                "a list of inputs only works when {} takes exactly one input, it takes {count}",
                site.written_path
            );
            self.invalid_inputs(fallback, message);
            return;
        }

        for input in &site.inputs {
            let slot = match (&input.value.name, slots.as_slice()) {
                (Some(name), _) => name.clone(),
                (None, [only]) => only.to_string(),
                (None, _) => unreachable!("a list is checked to fit exactly one input above"),
            };

            if !slots.contains(&slot.as_str()) {
                let message = format!(
                    "{} has no input named {slot}, it takes {listed}",
                    site.written_path
                );
                self.invalid_inputs(input.span, message);
                continue;
            }

            let alias = format!("{}{slot}", instance.prefix);
            let target = format!("{}{}", site.caller_prefix, input.value.target);
            self.aliases.insert(alias, target);
            bound.insert(slot);
        }

        for name in slots {
            if bound.contains(name) {
                continue;
            }

            let message = format!("the input {name} is not given, {} needs it", site.written_path);
            self.invalid_inputs(fallback, message);
        }
    }

    fn invalid_inputs(&mut self, span: Span, message: String) {
        let diagnostic = Diagnostic::error(DiagnosticCode::InvalidInputs, "", span, message);
        self.diagnostics.push(diagnostic);
    }

    /// Reports an input, output or call sharing a name with anything else in the file.
    ///
    /// Two stages sharing a name are not reported here.
    pub(super) fn check_names(&mut self, raw: &RawFile) {
        let stages = raw.stages.iter().map(|stage| (&stage.name, true));
        let inputs = raw.inputs.iter().map(|input| (&input.name, false));
        let outputs = raw.outputs.iter().map(|output| (&output.name, false));
        let calls = raw.calls.iter().map(|call| (&call.name, false));
        let mut named: Vec<(&Spanned<String>, bool)> =
            stages.chain(inputs).chain(outputs).chain(calls).collect();
        named.sort_by_key(|(name, _)| name.span.start);

        let mut seen: HashMap<&str, (Span, bool)> = HashMap::new();
        for (name, is_stage) in named {
            let Some((first_span, first_is_stage)) = seen.get(name.value.as_str()).copied() else {
                seen.insert(&name.value, (name.span, is_stage));
                continue;
            };

            if first_is_stage && is_stage {
                continue;
            }

            let message = format!("the name {} is already used in this file", name.value);
            let path = name.value.clone();
            let diagnostic = Diagnostic::error(DiagnosticCode::DuplicateStageName, path, name.span, message)
                .with_note("first used here", first_span);
            self.diagnostics.push(diagnostic);
        }
    }

    /// Reports each reference to a call's output that names no output.
    pub(super) fn check_references(
        &mut self,
        raw: &RawFile,
        outputs_of: &HashMap<String, Option<Vec<String>>>,
    ) {
        let stage_inputs = raw.stages.iter().flat_map(|stage| stage.inputs.iter());
        let call_inputs = raw.calls.iter().flat_map(|call| call.inputs.iter());
        let references = stage_inputs
            .chain(call_inputs)
            .map(|input| (input.value.target.as_str(), input.span));
        let output_sources = raw
            .outputs
            .iter()
            .map(|output| (output.source.value.as_str(), output.source.span));

        for (target, span) in references.chain(output_sources) {
            self.check_reference(target, span, outputs_of);
        }
    }

    fn check_reference(
        &mut self,
        target: &str,
        span: Span,
        outputs_of: &HashMap<String, Option<Vec<String>>>,
    ) {
        if let Some(Some(outputs)) = outputs_of.get(target) {
            if outputs.len() == 1 {
                return;
            }

            let count = outputs.len();
            let message = match outputs.first() {
                Some(first) => format!("{target} gives {count} outputs, pick one such as {target}.{first}"),
                None => format!("{target} gives no outputs"),
            };
            self.unknown_output(span, message, outputs);
            return;
        }

        let Some((call_name, output)) = target.split_once('.') else {
            return;
        };

        let Some(Some(outputs)) = outputs_of.get(call_name) else {
            return;
        };

        if outputs.iter().any(|known| known == output) {
            return;
        }

        let message = format!("{call_name} has no output named {output}");
        self.unknown_output(span, message, outputs);
    }

    fn unknown_output(&mut self, span: Span, message: String, outputs: &[String]) {
        let mut diagnostic = Diagnostic::error(DiagnosticCode::UnknownCallOutput, "", span, message);

        if !outputs.is_empty() {
            let listed = outputs.join(", ");
            let note = format!("the outputs are {listed}");
            diagnostic = diagnostic.with_note(note, Span::unknown());
        }

        self.diagnostics.push(diagnostic);
    }
}
