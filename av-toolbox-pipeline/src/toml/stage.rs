use std::borrow::Cow;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeTable, DeValue};
use av_toolbox_core::StageKind;
use av_toolbox_graph::{ArgSpans, Diagnostics, FileId, Span, Spanned};
use serde_json::{Map, Value};

use super::args::convert_args;
use super::reader::{TableReader, entry, is_stage_name};
use super::{RawStage, RawWorkers};
use crate::variables::{has_template, pieces};

const KEYS: [&str; 4] = ["type", "inputs", "workers", "args"];

/// Reads one `[kind.name]` table, reporting everything wrong with it.
pub(super) fn read_stage(
    kind: StageKind,
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    diagnostics: &mut Diagnostics,
) -> Option<RawStage> {
    let range = value.span();
    let header = Span::in_file(file, range.start, range.end);
    let written_name = name.get_ref();
    let mut reader = TableReader {
        path: format!("{kind}.{written_name}"),
        header,
        file,
        diagnostics,
        failed: false,
    };

    if !is_stage_name(written_name) {
        let message = format!("the stage name {written_name:?} may only use letters, digits and _");
        reader.invalid_name(header, written_name, message);
    }

    let inner = value.get_ref();
    let Some(table) = inner.as_table() else {
        reader.error(
            "",
            header,
            "a stage must be a table of type, inputs, workers and args",
        );
        return None;
    };

    reader.check_keys(
        table,
        &KEYS,
        "stage setting",
        "a stage takes type, inputs, workers and args",
    );

    let stage_type = stage_type(&mut reader, table);

    let inputs = match entry(table, "inputs") {
        Some(value) => reader.inputs(value),
        None => Vec::new(),
    };

    let workers = workers(&mut reader, table);
    let (args, arg_spans) = args(&mut reader, table);

    let (Some(stage_type), false) = (stage_type, reader.failed) else {
        return None;
    };

    let owned_name = written_name.to_string();
    Some(RawStage {
        name: Spanned::new(owned_name, header),
        kind,
        stage_type,
        inputs,
        args,
        arg_spans,
        workers,
        span: header,
    })
}

fn stage_type(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> Option<Spanned<String>> {
    let Some(written) = entry(table, "type") else {
        let header = reader.header;
        reader.error(
            "type",
            header,
            "type is missing, it names the stage to run, such as y4m/pipe",
        );
        return None;
    };

    let range = written.span();
    let span = reader.span(range);
    let inner = written.get_ref();
    let Some(text) = inner.as_str() else {
        reader.error("type", span, "type must be a string, such as \"y4m/pipe\"");
        return None;
    };

    let stage_type = Spanned::new(text.to_string(), span);
    reader.refuse_template("type", &stage_type);

    Some(stage_type)
}

fn workers(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> Option<Spanned<RawWorkers>> {
    let value = entry(table, "workers")?;
    let range = value.span();
    let span = reader.span(range);
    let inner = value.get_ref();

    let template = inner.as_str().filter(|text| has_template(text));
    if let Some(text) = template {
        let checked = pieces(text);
        if let Err(problem) = checked {
            reader.error("workers", span, problem.message());
            return None;
        }

        let template = RawWorkers::Template(text.to_string());
        return Some(Spanned::new(template, span));
    }

    let integer = inner.as_integer();
    let parsed = integer.and_then(|integer| {
        let text = integer.as_str();
        let radix = integer.radix();
        usize::from_str_radix(text, radix).ok()
    });

    match parsed {
        Some(count) if count > 0 => {
            let workers = RawWorkers::Count(count);
            Some(Spanned::new(workers, span))
        },
        _ => {
            reader.error("workers", span, "workers must be a whole number of at least 1");
            None
        },
    }
}

fn args(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> (Spanned<Value>, ArgSpans) {
    let empty_map = Map::new();
    let empty_value = Value::Object(empty_map);
    let empty = Spanned::new(empty_value, reader.header);
    let Some(value) = entry(table, "args") else {
        return (empty, ArgSpans::default());
    };

    let range = value.span();
    let span = reader.span(range);
    let inner = value.get_ref();
    let Some(arguments) = inner.as_table() else {
        reader.error("args", span, "args must be a table");
        return (empty, ArgSpans::default());
    };

    let path = reader.key_path("args");
    let converted = convert_args(arguments, &path, reader.file, reader.diagnostics);
    let Some((json, arg_spans)) = converted else {
        reader.failed = true;
        return (empty, ArgSpans::default());
    };

    (Spanned::new(json, span), arg_spans)
}
