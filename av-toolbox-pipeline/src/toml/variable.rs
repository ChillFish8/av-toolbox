use std::borrow::Cow;

use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeTable, DeValue};
use av_toolbox_graph::{Diagnostics, FileId, Span, Spanned};
use serde_json::Value;

use super::args::convert_value;
use super::reader::{TableReader, entry};
use crate::variables::{Declaration, VariableType, is_variable_name};

const KEYS: [&str; 6] = ["type", "description", "default", "min", "max", "choices"];

/// Reads one `[variable.name]` table, reporting everything wrong with it.
pub(super) fn read_variable(
    name: &TomlSpanned<Cow<'_, str>>,
    value: &TomlSpanned<DeValue<'_>>,
    file: FileId,
    diagnostics: &mut Diagnostics,
) -> Option<Declaration> {
    let range = value.span();
    let header = Span::in_file(file, range.start, range.end);
    let written_name = name.get_ref();
    let mut reader = TableReader {
        path: format!("variable.{written_name}"),
        header,
        file,
        diagnostics,
        failed: false,
    };

    if !is_variable_name(written_name) {
        let message =
            format!("the variable name {written_name:?} may only use lower case letters, digits and _");
        reader.error("", header, message);
    }

    let inner = value.get_ref();
    let Some(table) = inner.as_table() else {
        reader.error("", header, "a variable must be a table, such as [variable.name]");
        return None;
    };

    reader.check_keys(
        table,
        &KEYS,
        "variable setting",
        "a variable takes type, description, default, min, max and choices",
    );

    let variable_type = variable_type(&mut reader, table);
    let description = reader.string(table, "description");
    let default = written_value(&mut reader, table, "default");
    let min = written_value(&mut reader, table, "min");
    let max = written_value(&mut reader, table, "max");
    let choices = written_value(&mut reader, table, "choices");

    let variable_type = variable_type?;
    check_bound(&mut reader, variable_type, "min", min.as_ref());
    check_bound(&mut reader, variable_type, "max", max.as_ref());
    let choices = check_choices(&mut reader, variable_type, choices);

    if reader.failed {
        return None;
    }

    let owned_name = written_name.to_string();
    Some(Declaration {
        name: Spanned::new(owned_name, header),
        variable_type,
        description: description.map(|text| text.value),
        default: default.map(|written| written.value),
        min: min.map(|written| written.value),
        max: max.map(|written| written.value),
        choices,
    })
}

/// A value under a variable's key, with where the key itself was written.
struct Written {
    key: Span,
    value: Spanned<Value>,
}

fn variable_type(reader: &mut TableReader<'_>, table: &DeTable<'_>) -> Option<VariableType> {
    if entry(table, "type").is_none() {
        let header = reader.header;
        reader.error(
            "type",
            header,
            "type is missing, use string, path, integer, float or bool",
        );
        return None;
    }

    let written = reader.string(table, "type")?;
    let named = VariableType::named(&written.value);
    if named.is_none() {
        reader.error(
            "type",
            written.span,
            "type must be one of string, path, integer, float or bool",
        );
    }

    named
}

fn written_value(reader: &mut TableReader<'_>, table: &DeTable<'_>, key: &str) -> Option<Written> {
    let (written_key, value) = table.iter().find(|(name, _)| name.get_ref() == key)?;
    let key_range = written_key.span();
    let key_span = reader.span(key_range);
    let value_range = value.span();
    let value_span = reader.span(value_range);
    let path = reader.key_path(key);
    let converted = convert_value(value, &path, reader.file, reader.diagnostics);
    let Some(json) = converted else {
        reader.failed = true;
        return None;
    };

    Some(Written {
        key: key_span,
        value: Spanned::new(json, value_span),
    })
}

/// Checks `min` or `max` is allowed on the type and holds a number of that type.
fn check_bound(
    reader: &mut TableReader<'_>,
    variable_type: VariableType,
    key: &str,
    bound: Option<&Written>,
) {
    let Some(bound) = bound else {
        return;
    };

    if !variable_type.allows_bounds() {
        let message = format!("{key} only applies to integer and float variables");
        reader.error(key, bound.key, message);
        return;
    }

    let fits = match variable_type {
        VariableType::Integer => bound.value.value.is_i64(),
        _ => bound.value.value.is_number(),
    };

    if !fits {
        let message = format!("{key} must be a number of type {variable_type}");
        reader.error(key, bound.value.span, message);
    }
}

fn check_choices(
    reader: &mut TableReader<'_>,
    variable_type: VariableType,
    choices: Option<Written>,
) -> Option<Spanned<Vec<Value>>> {
    let choices = choices?;

    if !variable_type.allows_choices() {
        reader.error(
            "choices",
            choices.key,
            "choices only apply to string and integer variables",
        );
        return None;
    }

    let span = choices.value.span;
    let Value::Array(listed) = choices.value.value else {
        reader.error("choices", span, "choices must be a list");
        return None;
    };

    if listed.is_empty() {
        reader.error("choices", span, "choices must list at least one value");
        return None;
    }

    let fits = |choice: &Value| match variable_type {
        VariableType::Integer => choice.is_i64(),
        _ => choice.is_string(),
    };

    if !listed.iter().all(fits) {
        let message = format!("every choice must be of type {variable_type}");
        reader.error("choices", span, message);
        return None;
    }

    Some(Spanned::new(listed, span))
}
