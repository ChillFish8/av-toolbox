use ::toml::Spanned as TomlSpanned;
use ::toml::de::{DeArray, DeFloat, DeInteger, DeTable, DeValue};
use av_toolbox_graph::{ArgSpans, Diagnostic, DiagnosticCode, Diagnostics, Span};
use serde_json::{Map, Number, Value};

use super::span_of;

/// Converts a stage's `args` table to JSON, recording where every value was written.
///
/// Returns `None` after reporting each value that has no JSON form.
pub(super) fn convert_args(
    table: &DeTable<'_>,
    path: &str,
    diagnostics: &mut Diagnostics,
) -> Option<(Value, ArgSpans)> {
    let mut conversion = Conversion {
        path,
        spans: ArgSpans::new(),
        diagnostics,
        failed: false,
    };

    let converted = conversion.table(table, "");
    if conversion.failed {
        return None;
    }

    Some((converted, conversion.spans))
}

struct Conversion<'a> {
    path: &'a str,
    spans: ArgSpans,
    diagnostics: &'a mut Diagnostics,
    failed: bool,
}

impl Conversion<'_> {
    fn table(&mut self, table: &DeTable<'_>, pointer: &str) -> Value {
        let mut object = Map::new();

        for (key, value) in table.iter() {
            let name = key.get_ref();
            let escaped = escape(name);
            let nested = format!("{pointer}/{escaped}");
            let converted = self.value(value, &nested);
            let owned_name = name.to_string();

            object.insert(owned_name, converted);
        }

        Value::Object(object)
    }

    fn value(&mut self, value: &TomlSpanned<DeValue<'_>>, pointer: &str) -> Value {
        let range = value.span();
        let span = span_of(range);
        self.spans.insert(pointer, span);

        let inner = value.get_ref();
        match inner {
            DeValue::String(text) => {
                let owned = text.to_string();
                Value::String(owned)
            },
            DeValue::Integer(integer) => self.integer(integer, pointer, span),
            DeValue::Float(float) => self.float(float, pointer, span),
            DeValue::Boolean(flag) => Value::Bool(*flag),
            DeValue::Datetime(_) => self.refuse(pointer, span, "dates and times cannot be passed to a stage"),
            DeValue::Array(items) => self.array(items, pointer),
            DeValue::Table(table) => self.table(table, pointer),
        }
    }

    fn integer(&mut self, integer: &DeInteger<'_>, pointer: &str, span: Span) -> Value {
        let text = integer.as_str();
        let radix = integer.radix();
        let parsed = i64::from_str_radix(text, radix);

        match parsed {
            Ok(number) => Value::from(number),
            Err(_) => self.refuse(pointer, span, "the number does not fit in a 64 bit integer"),
        }
    }

    fn float(&mut self, float: &DeFloat<'_>, pointer: &str, span: Span) -> Value {
        let text = float.as_str();
        let parsed = text.parse::<f64>();
        let number = parsed.ok().and_then(Number::from_f64);

        match number {
            Some(number) => Value::Number(number),
            None => self.refuse(
                pointer,
                span,
                "NaN and infinite numbers cannot be passed to a stage",
            ),
        }
    }

    fn array(&mut self, items: &DeArray<'_>, pointer: &str) -> Value {
        let mut converted = Vec::with_capacity(items.len());

        for (index, item) in items.iter().enumerate() {
            let nested = format!("{pointer}/{index}");
            let value = self.value(item, &nested);

            converted.push(value);
        }

        Value::Array(converted)
    }

    fn refuse(&mut self, pointer: &str, span: Span, message: &str) -> Value {
        let dotted = pointer.replace('/', ".");
        let path = format!("{}{dotted}", self.path);
        let diagnostic = Diagnostic::error(DiagnosticCode::InvalidDefinition, path, span, message);

        self.diagnostics.push(diagnostic);
        self.failed = true;

        Value::Null
    }
}

/// Escapes a key for a JSON pointer, as the schema validator writes them.
fn escape(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}
