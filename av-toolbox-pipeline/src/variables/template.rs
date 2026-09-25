use serde_json::Value;

use super::Values;

/// One piece of a string that may hold templates.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
pub(crate) enum Piece<'a> {
    Text(&'a str),
    Variable(&'a str),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
pub(crate) enum TemplateError {
    Unclosed,
    EmptyName,
}

impl TemplateError {
    #[expect(dead_code, reason = "used by the loader")]
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::Unclosed => "a template opened with {{ is never closed with }}",
            Self::EmptyName => "a template must name a variable, such as {{ clip_input }}",
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
pub(crate) enum FillError {
    Template(TemplateError),
    Unknown(String),
}

#[expect(dead_code, reason = "used by the loader")]
pub(crate) fn has_template(text: &str) -> bool {
    text.contains("{{")
}

/// Splits `text` into plain text and the variables its templates name.
#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
pub(crate) fn pieces(text: &str) -> Result<Vec<Piece<'_>>, TemplateError> {
    let mut found = Vec::new();
    let mut rest = text;

    while let Some(open) = rest.find("{{") {
        let before = &rest[..open];
        if !before.is_empty() {
            found.push(Piece::Text(before));
        }

        let after_open = &rest[open + 2..];
        let Some(close) = after_open.find("}}") else {
            return Err(TemplateError::Unclosed);
        };

        let name = after_open[..close].trim();
        if name.is_empty() {
            return Err(TemplateError::EmptyName);
        }

        found.push(Piece::Variable(name));
        rest = &after_open[close + 2..];
    }

    if !rest.is_empty() {
        found.push(Piece::Text(rest));
    }

    Ok(found)
}

/// The value `text` becomes once its templates are filled from `values`.
///
/// A string that is exactly one template becomes that variable's typed value. Any other string
/// stays a string, with each template replaced by its value's text.
#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
pub(crate) fn fill(text: &str, values: &Values) -> Result<Value, FillError> {
    let found = pieces(text).map_err(FillError::Template)?;

    if let [Piece::Variable(name)] = found.as_slice() {
        let value = lookup(values, name)?;
        return Ok(value.to_json());
    }

    let mut filled = String::with_capacity(text.len());
    for piece in &found {
        match piece {
            Piece::Text(plain) => filled.push_str(plain),
            Piece::Variable(name) => {
                let value = lookup(values, name)?;
                let rendered = value.to_text();

                filled.push_str(&rendered);
            },
        }
    }

    Ok(Value::String(filled))
}

#[cfg_attr(not(test), expect(dead_code, reason = "used by the loader"))]
fn lookup<'v>(values: &'v Values, name: &str) -> Result<&'v super::VariableValue, FillError> {
    let found = values.get(name);
    found.ok_or_else(|| FillError::Unknown(name.to_string()))
}
