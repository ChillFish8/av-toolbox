use std::ops::Range;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeArray, DeTable, DeValue};
use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, FileId, InputRef, Span, Spanned};

use crate::variables::has_template;

/// One table being read, remembering whether anything in it was wrong.
pub(super) struct TableReader<'a> {
    /// The table's own path, such as `filter.denoise`.
    pub(super) path: String,
    pub(super) header: Span,
    pub(super) file: FileId,
    pub(super) diagnostics: &'a mut Diagnostics,
    pub(super) failed: bool,
}

impl TableReader<'_> {
    pub(super) fn span(&self, range: Range<usize>) -> Span {
        Span::in_file(self.file, range.start, range.end)
    }

    pub(super) fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
        self.failed = true;
    }

    pub(super) fn error(&mut self, key: &str, span: Span, message: impl Into<String>) {
        let path = self.key_path(key);
        let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

        self.report(diagnostic);
    }

    /// Reports the stage name `written_name` as invalid, suggesting the name with `_` for each `-` when that is valid.
    pub(super) fn invalid_name(&mut self, header: Span, written_name: &str, message: String) {
        let path = self.key_path("");
        let mut diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, header, message);

        let suggestion = written_name.replace('-', "_");
        if is_stage_name(&suggestion) {
            let note = format!("did you mean `{suggestion}`?");
            diagnostic = diagnostic.with_note(note, Span::unknown());
        }

        self.report(diagnostic);
    }

    pub(super) fn key_path(&self, key: &str) -> String {
        match key {
            "" => self.path.clone(),
            key => format!("{}.{key}", self.path),
        }
    }

    /// Reports every key of `table` not in `allowed`.
    ///
    /// Each report calls the key a `noun` and adds `note`, which lists the keys that are allowed.
    pub(super) fn check_keys(&mut self, table: &DeTable<'_>, allowed: &[&str], noun: &str, note: &str) {
        for key in table.keys() {
            let name = key.get_ref();
            let is_known = allowed.contains(&name.as_ref());
            if is_known {
                continue;
            }

            let path = self.key_path(name);
            let range = key.span();
            let span = self.span(range);
            let message = format!("{name} is not a {noun}");
            let unknown = Span::unknown();
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message)
                .with_note(note, unknown);

            self.report(diagnostic);
        }
    }

    /// Reads `key` as a string, reporting a value of any other shape.
    pub(super) fn string(&mut self, table: &DeTable<'_>, key: &str) -> Option<Spanned<String>> {
        let value = entry(table, key)?;
        let range = value.span();
        let span = self.span(range);
        let inner = value.get_ref();
        let Some(text) = inner.as_str() else {
            self.error(key, span, format!("{key} must be a string"));
            return None;
        };

        let owned = text.to_string();
        Some(Spanned::new(owned, span))
    }

    /// Refuses a template in a string that must be taken as written.
    pub(super) fn refuse_template(&mut self, key: &str, text: &Spanned<String>) {
        if !has_template(&text.value) {
            return;
        }

        let message = format!("templates cannot be used in {key}");
        self.error(key, text.span, message);
    }

    /// Reads an `inputs` value, a list of names or a table of slot names to names.
    pub(super) fn inputs(&mut self, value: &TomlSpanned<DeValue<'_>>) -> Vec<Spanned<InputRef>> {
        match value.get_ref() {
            DeValue::Array(items) => self.positional(items),
            DeValue::Table(entries) => self.named(entries),
            _ => {
                let range = value.span();
                let span = self.span(range);
                self.error(
                    "inputs",
                    span,
                    "inputs must be a list of stage names, or a table of input names to stage names",
                );

                Vec::new()
            },
        }
    }

    fn positional(&mut self, items: &DeArray<'_>) -> Vec<Spanned<InputRef>> {
        let mut inputs = Vec::with_capacity(items.len());

        for item in items.iter() {
            let range = item.span();
            let span = self.span(range);
            let inner = item.get_ref();
            let Some(target) = inner.as_str() else {
                self.error("inputs", span, "each input must be the name of a stage");
                continue;
            };

            let written = Spanned::new(target.to_string(), span);
            self.refuse_template("inputs", &written);

            let input = InputRef {
                name: None,
                target: written.value,
            };
            inputs.push(Spanned::new(input, span));
        }

        inputs
    }

    fn named(&mut self, entries: &DeTable<'_>) -> Vec<Spanned<InputRef>> {
        let mut inputs = Vec::with_capacity(entries.len());

        for (slot, item) in entries.iter() {
            let slot_range = slot.span();
            let item_range = item.span();
            let span = Span::in_file(self.file, slot_range.start, item_range.end);

            if has_template(slot.get_ref()) {
                let slot_span = self.span(slot_range);
                self.error("inputs", slot_span, "templates cannot be used in input names");
                continue;
            }

            let inner = item.get_ref();
            let Some(target) = inner.as_str() else {
                self.error("inputs", span, "each input must be the name of a stage");
                continue;
            };

            let written = Spanned::new(target.to_string(), span);
            self.refuse_template("inputs", &written);

            let slot_name = slot.get_ref().to_string();
            let input = InputRef {
                name: Some(slot_name),
                target: written.value,
            };
            inputs.push(Spanned::new(input, span));
        }

        inputs
    }
}

pub(super) fn entry<'t, 'i>(table: &'t DeTable<'i>, key: &str) -> Option<&'t TomlSpanned<DeValue<'i>>> {
    let found = table.iter().find(|(name, _)| name.get_ref() == key);
    found.map(|(_, value)| value)
}

/// Whether `name` may name a stage, call, input or output.
pub(super) fn is_stage_name(name: &str) -> bool {
    let allowed = |character: char| character.is_ascii_alphanumeric() || character == '_';

    !name.is_empty() && name.chars().all(allowed)
}
