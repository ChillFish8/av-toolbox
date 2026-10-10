mod supplied;
mod template;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use av_toolbox_graph::Spanned;
use serde_json::{Number, Value};

pub use self::supplied::Supplied;
pub(crate) use self::supplied::{SuppliedValue, environment_name};
pub(crate) use self::template::{FillError, fill, has_template, pieces};
#[cfg(test)]
pub(crate) use self::template::{Piece, TemplateError};

/// Every variable of one file, by name, once resolved.
pub(crate) type Values = HashMap<String, VariableValue>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum VariableType {
    String,
    Path,
    Integer,
    Float,
    Bool,
}

impl VariableType {
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "string" => Some(Self::String),
            "path" => Some(Self::Path),
            "integer" => Some(Self::Integer),
            "float" => Some(Self::Float),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Path => "path",
            Self::Integer => "integer",
            Self::Float => "float",
            Self::Bool => "bool",
        }
    }

    pub fn allows_bounds(self) -> bool {
        matches!(self, Self::Integer | Self::Float)
    }

    pub fn allows_choices(self) -> bool {
        matches!(self, Self::String | Self::Integer)
    }
}

impl fmt::Display for VariableType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A variable's value once it has been read and checked.
///
/// A path is held as the absolute path it resolved to.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableValue {
    String(String),
    Path(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
}

impl VariableValue {
    pub fn to_json(&self) -> Value {
        match self {
            Self::String(text) | Self::Path(text) => Value::String(text.clone()),
            Self::Integer(number) => Value::from(*number),
            Self::Float(number) => {
                let converted = Number::from_f64(*number);
                converted.map(Value::Number).unwrap_or(Value::Null)
            },
            Self::Bool(flag) => Value::Bool(*flag),
        }
    }

    pub fn to_text(&self) -> String {
        match self {
            Self::String(text) | Self::Path(text) => text.clone(),
            Self::Integer(number) => number.to_string(),
            Self::Float(number) => number.to_string(),
            Self::Bool(flag) => flag.to_string(),
        }
    }
}

/// A root file variable and the value it resolved to.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedVariable {
    pub name: String,
    pub variable_type: VariableType,
    pub value: VariableValue,
}

/// A `[variable.<name>]` table as it was written.
#[derive(Debug, Clone)]
pub struct Declaration {
    /// The name, spanning the table's header.
    pub name: Spanned<String>,
    pub variable_type: VariableType,
    pub description: Option<String>,
    pub default: Option<Spanned<Value>>,
    pub min: Option<Spanned<Value>>,
    pub max: Option<Spanned<Value>>,
    pub choices: Option<Spanned<Vec<Value>>>,
}

pub(crate) fn is_variable_name(name: &str) -> bool {
    let allowed =
        |character: char| character.is_ascii_lowercase() || character.is_ascii_digit() || character == '_';

    !name.is_empty() && name.chars().all(allowed)
}

/// Reads text from the command line or environment as a value of `variable_type`.
pub(crate) fn parse_text(
    variable_type: VariableType,
    text: &str,
    directory: &Path,
) -> Result<VariableValue, String> {
    match variable_type {
        VariableType::String => Ok(VariableValue::String(text.to_string())),
        VariableType::Path => {
            let resolved = resolve_path(directory, text);
            Ok(VariableValue::Path(resolved))
        },
        VariableType::Integer => {
            let parsed = text.parse::<i64>();
            parsed
                .map(VariableValue::Integer)
                .map_err(|_| format!("{text:?} is not a whole number"))
        },
        VariableType::Float => {
            let parsed = text.parse::<f64>();
            match parsed {
                Ok(number) if number.is_finite() => Ok(VariableValue::Float(number)),
                _ => Err(format!("{text:?} is not a finite number")),
            }
        },
        VariableType::Bool => match text {
            "true" => Ok(VariableValue::Bool(true)),
            "false" => Ok(VariableValue::Bool(false)),
            _ => Err(format!("{text:?} is not true or false")),
        },
    }
}

/// Reads a value written in a pipeline file as a value of `variable_type`.
pub(crate) fn from_json(
    variable_type: VariableType,
    value: &Value,
    directory: &Path,
) -> Result<VariableValue, String> {
    let converted = match variable_type {
        VariableType::String => value.as_str().map(|text| VariableValue::String(text.to_string())),
        VariableType::Path => value.as_str().map(|text| {
            let resolved = resolve_path(directory, text);
            VariableValue::Path(resolved)
        }),
        VariableType::Integer => value.as_i64().map(VariableValue::Integer),
        VariableType::Float => value.as_f64().map(VariableValue::Float),
        VariableType::Bool => value.as_bool().map(VariableValue::Bool),
    };

    converted.ok_or_else(|| format!("expected a value of type {variable_type}, found {value}"))
}

/// Checks `value` against the declaration's `min`, `max` and `choices`.
pub(crate) fn check_rules(declaration: &Declaration, value: &VariableValue) -> Result<(), String> {
    let number = match value {
        VariableValue::Integer(number) => Some(*number as f64),
        VariableValue::Float(number) => Some(*number),
        _ => None,
    };

    if let (Some(number), Some(min)) = (number, &declaration.min) {
        let bound = min.value.as_f64().unwrap_or(f64::MIN);
        if number < bound {
            return Err(format!("{number} is below the minimum of {}", min.value));
        }
    }

    if let (Some(number), Some(max)) = (number, &declaration.max) {
        let bound = max.value.as_f64().unwrap_or(f64::MAX);
        if number > bound {
            return Err(format!("{number} is above the maximum of {}", max.value));
        }
    }

    let Some(choices) = &declaration.choices else {
        return Ok(());
    };

    let written = value.to_json();
    if choices.value.contains(&written) {
        return Ok(());
    }

    let listed: Vec<String> = choices.value.iter().map(render_choice).collect();
    let joined = listed.join(", ");
    let shown = value.to_text();

    Err(format!("{shown} is not one of {joined}"))
}

fn render_choice(choice: &Value) -> String {
    match choice {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// Resolves `text` against `directory` into an absolute path, with `.` and `..` folded away.
pub(crate) fn resolve_path(directory: &Path, text: &str) -> String {
    let joined = directory.join(text);
    let normalised = normalise_path(&joined);

    normalised.to_string_lossy().into_owned()
}

/// Makes `path` absolute and folds away `.` and `..` without touching the disk.
pub(crate) fn normalise_path(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut folded = PathBuf::new();

    for component in absolute.components() {
        match component {
            Component::CurDir => {},
            Component::ParentDir => {
                folded.pop();
            },
            other => folded.push(other),
        }
    }

    folded
}
