use std::borrow::Cow;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeArray, DeTable, DeValue};
use av_toolbox_core::StageKind;
use av_toolbox_graph::{
    ArgSpans,
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    InputRef,
    Span,
    Spanned,
    StageSpec,
};
use serde_json::{Map, Value};

use super::args::convert_args;
use super::span_of;

const KEYS: [&str; 4] = ["type", "inputs", "workers", "args"];

/// Reads one `[kind.name]` table into a spec, reporting everything wrong with it.
pub(super) fn read_stage(
    kind: StageKind,
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    diagnostics: &mut Diagnostics,
) -> Option<StageSpec> {
    let range = value.span();
    let header = span_of(range);
    let mut reader = StageReader {
        kind,
        name: name.get_ref(),
        header,
        diagnostics,
        failed: false,
    };

    reader.check_name();

    let inner = value.get_ref();
    let Some(table) = inner.as_table() else {
        reader.error(
            "",
            header,
            "a stage must be a table of type, inputs, workers and args",
        );
        return None;
    };

    reader.check_keys(table);

    let stage_type = reader.stage_type(table);
    let inputs = reader.inputs(table);
    let workers = reader.workers(table);
    let (args, arg_spans) = reader.args(table);

    let (Some(stage_type), false) = (stage_type, reader.failed) else {
        return None;
    };

    let name = name.get_ref().to_string();
    let name = Spanned::new(name, header);
    let spec = StageSpec {
        name,
        kind,
        stage_type,
        inputs,
        args,
        arg_spans,
        workers,
        span: header,
    };
    Some(spec)
}

/// One stage table being read, remembering whether anything in it was wrong.
struct StageReader<'a> {
    kind: StageKind,
    name: &'a str,
    header: Span,
    diagnostics: &'a mut Diagnostics,
    failed: bool,
}

impl StageReader<'_> {
    fn report(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
        self.failed = true;
    }

    fn error(&mut self, key: &str, span: Span, message: impl Into<String>) {
        let path = self.path(key);
        let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

        self.report(diagnostic);
    }

    fn path(&self, key: &str) -> String {
        match key {
            "" => format!("{}.{}", self.kind, self.name),
            key => format!("{}.{}.{key}", self.kind, self.name),
        }
    }

    fn check_name(&mut self) {
        let allowed =
            |character: char| character.is_ascii_alphanumeric() || character == '_' || character == '-';
        let valid = !self.name.is_empty() && self.name.chars().all(allowed);
        if valid {
            return;
        }

        let message = format!(
            "the stage name {:?} may only use letters, digits, _ and -",
            self.name
        );
        self.error("", self.header, message);
    }

    fn check_keys(&mut self, table: &DeTable<'_>) {
        for key in table.keys() {
            let name = key.get_ref();
            let name_ref = name.as_ref();
            let is_known = KEYS.contains(&name_ref);
            if is_known {
                continue;
            }

            let path = self.path(name);
            let range = key.span();
            let span = span_of(range);
            let message = format!("{name} is not a stage setting");
            let unknown = Span::unknown();
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message)
                .with_note("a stage takes type, inputs, workers and args", unknown);

            self.report(diagnostic);
        }
    }

    fn stage_type(&mut self, table: &DeTable<'_>) -> Option<Spanned<String>> {
        let Some(value) = entry(table, "type") else {
            self.error(
                "type",
                self.header,
                "type is missing, it names the stage to run, such as y4m/pipe",
            );
            return None;
        };

        let range = value.span();
        let span = span_of(range);
        let inner = value.get_ref();
        let Some(text) = inner.as_str() else {
            self.error("type", span, "type must be a string, such as \"y4m/pipe\"");
            return None;
        };

        let text = text.to_string();
        let spanned = Spanned::new(text, span);
        Some(spanned)
    }

    fn inputs(&mut self, table: &DeTable<'_>) -> Vec<Spanned<InputRef>> {
        let Some(value) = entry(table, "inputs") else {
            return Vec::new();
        };

        match value.get_ref() {
            DeValue::Array(items) => self.positional(items),
            DeValue::Table(entries) => self.named(entries),
            _ => {
                let range = value.span();
                let span = span_of(range);
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
            let span = span_of(range);
            let inner = item.get_ref();
            let Some(target) = inner.as_str() else {
                self.error("inputs", span, "each input must be the name of a stage");
                continue;
            };

            let input = InputRef {
                name: None,
                target: target.to_string(),
            };
            let spanned = Spanned::new(input, span);
            inputs.push(spanned);
        }

        inputs
    }

    fn named(&mut self, entries: &DeTable<'_>) -> Vec<Spanned<InputRef>> {
        let mut inputs = Vec::with_capacity(entries.len());

        for (slot, item) in entries.iter() {
            let slot_span = slot.span();
            let item_span = item.span();
            let span = Span::new(slot_span.start, item_span.end);
            let inner = item.get_ref();
            let Some(target) = inner.as_str() else {
                self.error("inputs", span, "each input must be the name of a stage");
                continue;
            };

            let slot_name = slot.get_ref().to_string();
            let input = InputRef {
                name: Some(slot_name),
                target: target.to_string(),
            };
            let spanned = Spanned::new(input, span);
            inputs.push(spanned);
        }

        inputs
    }

    fn workers(&mut self, table: &DeTable<'_>) -> Option<usize> {
        let value = entry(table, "workers")?;
        let range = value.span();
        let span = span_of(range);
        let inner = value.get_ref();
        let integer = inner.as_integer();
        let parsed = integer.and_then(|integer| {
            let text = integer.as_str();
            let radix = integer.radix();
            usize::from_str_radix(text, radix).ok()
        });

        match parsed {
            Some(workers) if workers > 0 => Some(workers),
            _ => {
                self.error("workers", span, "workers must be a whole number of at least 1");
                None
            },
        }
    }

    fn args(&mut self, table: &DeTable<'_>) -> (Spanned<Value>, ArgSpans) {
        let empty_map = Map::new();
        let empty_value = Value::Object(empty_map);
        let empty = Spanned::new(empty_value, self.header);
        let Some(value) = entry(table, "args") else {
            return (empty, ArgSpans::default());
        };

        let range = value.span();
        let span = span_of(range);
        let inner = value.get_ref();
        let Some(arguments) = inner.as_table() else {
            self.error("args", span, "args must be a table");
            return (empty, ArgSpans::default());
        };

        let path = self.path("args");
        let converted = convert_args(arguments, &path, self.diagnostics);
        let Some((json, arg_spans)) = converted else {
            self.failed = true;
            return (empty, ArgSpans::default());
        };

        let spanned = Spanned::new(json, span);
        (spanned, arg_spans)
    }
}

fn entry<'t, 'i>(table: &'t DeTable<'i>, key: &str) -> Option<&'t TomlSpanned<DeValue<'i>>> {
    let found = table.iter().find(|(name, _)| name.get_ref() == key);
    found.map(|(_, value)| value)
}
