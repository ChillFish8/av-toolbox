use std::borrow::Cow;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeTable, DeValue};
use av_toolbox_graph::{Diagnostics, FileId, Span, Spanned};

use super::args::convert_value;
use super::reader::{TableReader, entry, is_stage_name};
use super::{RawCall, RawCallVariable, RawInput, RawOutput};
use crate::variables::{has_template, pieces};

const CALL_KEYS: [&str; 3] = ["path", "inputs", "variables"];
const INPUT_KEYS: [&str; 1] = ["description"];
const OUTPUT_KEYS: [&str; 2] = ["inputs", "description"];

/// Starts reading a `[<table>.<name>]` table, checking its name and that it is a table.
fn open<'a, 'v, 'i>(
    table: &str,
    name: &TomlSpanned<Cow<'_, str>>,
    value: &'v TomlSpanned<DeValue<'i>>,
    file: FileId,
    diagnostics: &'a mut Diagnostics,
) -> (TableReader<'a>, Option<&'v DeTable<'i>>) {
    let range = value.span();
    let header = Span::in_file(file, range.start, range.end);
    let written_name = name.get_ref();
    let mut reader = TableReader {
        path: format!("{table}.{written_name}"),
        header,
        file,
        diagnostics,
        failed: false,
    };

    if !is_stage_name(written_name) {
        let message = format!("the name {written_name:?} may only use letters, digits and _");
        reader.invalid_name(header, written_name, message);
    }

    let inner = value.get_ref();
    let found = inner.as_table();
    if found.is_none() {
        let message = format!("{table} entries must be tables, such as [{table}.name]");
        reader.error("", header, message);
    }

    (reader, found)
}

pub(super) fn read_call(
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    diagnostics: &mut Diagnostics,
) -> Option<RawCall> {
    let (mut reader, table) = open("call", name, value, file, diagnostics);
    let table = table?;

    reader.check_keys(
        table,
        &CALL_KEYS,
        "call setting",
        "a call takes path, inputs and variables",
    );

    let path = call_path(&mut reader, table);

    let inputs_value = entry(table, "inputs");
    let inputs = match inputs_value {
        Some(value) => reader.inputs(value),
        None => Vec::new(),
    };

    let inputs_span = inputs_value.map(|value| {
        let range = value.span();
        reader.span(range)
    });

    let variables = call_variables(&mut reader, table);

    let (Some(path), false) = (path, reader.failed) else {
        return None;
    };

    let header = reader.header;
    let written_name = name.get_ref().to_string();
    Some(RawCall {
        name: Spanned::new(written_name, header),
        path,
        inputs,
        inputs_span,
        variables,
        span: header,
    })
}

fn call_path(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> Option<Spanned<String>> {
    if entry(table, "path").is_none() {
        let header = reader.header;
        reader.error(
            "path",
            header,
            "path is missing, it names the pipeline file to call",
        );
        return None;
    }

    let path = reader.string(table, "path")?;
    let checked = pieces(&path.value);
    if let Err(problem) = checked {
        reader.error("path", path.span, problem.message());
        return None;
    }

    Some(path)
}

fn call_variables(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> Vec<RawCallVariable> {
    let Some(value) = entry(table, "variables") else {
        return Vec::new();
    };

    let inner = value.get_ref();
    let Some(entries) = inner.as_table() else {
        let range = value.span();
        let span = reader.span(range);
        reader.error(
            "variables",
            span,
            "variables must be a table of variable names to values",
        );
        return Vec::new();
    };

    let mut variables = Vec::with_capacity(entries.len());
    for (key, item) in entries.iter() {
        let key_range = key.span();
        let key_span = reader.span(key_range);
        let item_range = item.span();
        let item_span = reader.span(item_range);
        let written_name = key.get_ref().to_string();

        if has_template(&written_name) {
            reader.error(
                "variables",
                key_span,
                "templates cannot be used in variable names",
            );
            continue;
        }

        let path = reader.key_path("variables");
        let converted = convert_value(item, &path, reader.file, reader.diagnostics);
        let Some(json) = converted else {
            reader.failed = true;
            continue;
        };

        variables.push(RawCallVariable {
            name: Spanned::new(written_name, key_span),
            value: Spanned::new(json, item_span),
        });
    }

    variables
}

pub(super) fn read_input(
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    diagnostics: &mut Diagnostics,
) -> Option<RawInput> {
    let (mut reader, table) = open("input", name, value, file, diagnostics);
    let table = table?;

    reader.check_keys(
        table,
        &INPUT_KEYS,
        "input setting",
        "an input takes only a description",
    );
    let description = reader.string(table, "description");

    if reader.failed {
        return None;
    }

    let header = reader.header;
    let written_name = name.get_ref().to_string();
    Some(RawInput {
        name: Spanned::new(written_name, header),
        description: description.map(|text| text.value),
    })
}

pub(super) fn read_output(
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    diagnostics: &mut Diagnostics,
) -> Option<RawOutput> {
    let (mut reader, table) = open("output", name, value, file, diagnostics);
    let table = table?;

    reader.check_keys(
        table,
        &OUTPUT_KEYS,
        "output setting",
        "an output takes inputs and a description",
    );
    let description = reader.string(table, "description");

    let Some(inputs_value) = entry(table, "inputs") else {
        let header = reader.header;
        reader.error(
            "inputs",
            header,
            "inputs is missing, it names the one stage this output gives back",
        );
        return None;
    };

    let failed_before = reader.failed;
    let inputs = reader.inputs(inputs_value);
    if reader.failed && !failed_before {
        return None;
    }

    let range = inputs_value.span();
    let inputs_span = reader.span(range);
    let single = match inputs.as_slice() {
        [only] if only.value.name.is_none() => Some(only),
        _ => None,
    };

    let Some(source) = single else {
        reader.error(
            "inputs",
            inputs_span,
            "an output reads exactly one stage, such as inputs = [\"denoise\"]",
        );
        return None;
    };

    let source = Spanned::new(source.value.target.clone(), source.span);

    if reader.failed {
        return None;
    }

    let header = reader.header;
    let written_name = name.get_ref().to_string();
    Some(RawOutput {
        name: Spanned::new(written_name, header),
        source,
        description: description.map(|text| text.value),
    })
}
