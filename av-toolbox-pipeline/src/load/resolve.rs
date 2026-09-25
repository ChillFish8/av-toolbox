use std::path::Path;

use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, Span, suggest_name};

use crate::variables::{
    Declaration,
    Supplied,
    SuppliedValue,
    Values,
    VariableValue,
    check_rules,
    environment_name,
    from_json,
    parse_text,
};

/// Where a file's variable values come from.
pub(super) enum Given<'a> {
    /// The root file, filled from flags and the environment.
    Root(&'a Supplied),
    /// A called file, filled from its call's `variables` table.
    Call(&'a [GivenValue]),
}

/// One entry of a call's `variables` table, with its templates already filled.
pub(super) struct GivenValue {
    pub(super) name: String,
    pub(super) name_span: Span,
    pub(super) value: serde_json::Value,
    pub(super) value_span: Span,
    /// The directory of the calling file, which a path value is relative to.
    pub(super) directory: std::path::PathBuf,
}

/// Resolves every declared variable to a value, reporting each one that cannot be.
///
/// `directory` is the directory of the file declaring the variables.
pub(super) fn resolve_variables(
    declarations: &[Declaration],
    given: &Given<'_>,
    directory: &Path,
    diagnostics: &mut Diagnostics,
) -> Values {
    let mut values = Values::new();

    for declaration in declarations {
        let default = default_value(declaration, directory, diagnostics);
        let supplied = supplied_value(declaration, given, diagnostics);

        let chosen = match (supplied, default) {
            (Some(value), _) => Some(value),
            (None, Some(value)) => Some(value),
            (None, None) => None,
        };

        let Some(value) = chosen else {
            let has_default = declaration.default.is_some();
            let was_given = is_given(declaration, given);
            if !has_default && !was_given {
                let missing = missing(declaration, given);
                diagnostics.push(missing);
            }

            continue;
        };

        let name = declaration.name.value.clone();
        values.insert(name, value);
    }

    report_unknown(declarations, given, diagnostics);

    values
}

/// The declared default, or `None` when there is none or it is invalid.
fn default_value(
    declaration: &Declaration,
    directory: &Path,
    diagnostics: &mut Diagnostics,
) -> Option<VariableValue> {
    let default = declaration.default.as_ref()?;
    let converted = from_json(declaration.variable_type, &default.value, directory);
    let checked = converted.and_then(|value| check_rules(declaration, &value).map(|()| value));

    match checked {
        Ok(value) => Some(value),
        Err(problem) => {
            let name = &declaration.name.value;
            let message = format!("the default of {name} is invalid, {problem}");
            let path = format!("variable.{name}.default");
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidVariable, path, default.span, message);

            diagnostics.push(diagnostic);
            None
        },
    }
}

/// The value given from outside the file, or `None` when none was given or it is invalid.
fn supplied_value(
    declaration: &Declaration,
    given: &Given<'_>,
    diagnostics: &mut Diagnostics,
) -> Option<VariableValue> {
    let name = &declaration.name.value;
    let variable_type = declaration.variable_type;
    let working_directory = Path::new("");

    let (read, origin, span) = match given {
        Given::Root(supplied) => match supplied.get(name)? {
            SuppliedValue::Flag(text) => {
                let read = parse_text(variable_type, text, working_directory);
                let origin = format!("--var {name}");
                (read, origin, Span::unknown())
            },
            SuppliedValue::Environment(text) => {
                let read = parse_text(variable_type, text, working_directory);
                let origin = environment_name(name);
                (read, origin, Span::unknown())
            },
        },
        Given::Call(values) => {
            let entry = values.iter().find(|entry| entry.name == *name)?;
            let read = from_json(variable_type, &entry.value, &entry.directory);
            let origin = String::from("the call");
            (read, origin, entry.value_span)
        },
    };

    let checked = read.and_then(|value| check_rules(declaration, &value).map(|()| value));
    match checked {
        Ok(value) => Some(value),
        Err(problem) => {
            let message = format!("the value of {name} from {origin} is invalid, {problem}");
            let path = format!("variable.{name}");
            let header = declaration.name.span;
            let diagnostic = Diagnostic::error(DiagnosticCode::InvalidVariable, path, span, message)
                .with_note("declared here", header);

            diagnostics.push(diagnostic);
            None
        },
    }
}

/// Whether a value was given from outside the file, valid or not.
fn is_given(declaration: &Declaration, given: &Given<'_>) -> bool {
    let name = &declaration.name.value;

    match given {
        Given::Root(supplied) => supplied.get(name).is_some(),
        Given::Call(values) => values.iter().any(|entry| entry.name == *name),
    }
}

fn missing(declaration: &Declaration, given: &Given<'_>) -> Diagnostic {
    let name = &declaration.name.value;
    let message = format!("{name} has no value and no default");
    let path = format!("variable.{name}");
    let mut diagnostic = Diagnostic::error(
        DiagnosticCode::MissingVariable,
        path,
        declaration.name.span,
        message,
    );

    if let Some(description) = &declaration.description {
        diagnostic = diagnostic.with_note(description.clone(), Span::unknown());
    }

    let how = match given {
        Given::Root(_) => {
            let environment = environment_name(name);
            format!("set it with --var {name}=... or {environment}=...")
        },
        Given::Call(_) => format!("set it with variables.{name} on the call"),
    };

    diagnostic.with_note(how, Span::unknown())
}

fn report_unknown(declarations: &[Declaration], given: &Given<'_>, diagnostics: &mut Diagnostics) {
    let declared = || {
        declarations
            .iter()
            .map(|declaration| declaration.name.value.as_str())
    };
    let is_declared = |name: &str| declared().any(|declared_name| declared_name == name);

    match given {
        Given::Root(supplied) => {
            let names: std::collections::BTreeSet<&str> = supplied.flag_names().collect();

            for name in names {
                if is_declared(name) {
                    continue;
                }

                let message = format!("--var {name} names no variable in this pipeline");
                let mut diagnostic =
                    Diagnostic::error(DiagnosticCode::UnknownVariable, "", Span::unknown(), message);

                let closest = suggest_name(name, declared());
                if let Some(closest) = closest {
                    let note = format!("did you mean `{closest}`?");
                    diagnostic = diagnostic.with_note(note, Span::unknown());
                }

                diagnostics.push(diagnostic);
            }
        },
        Given::Call(values) => {
            for entry in values.iter() {
                if is_declared(&entry.name) {
                    continue;
                }

                let message = format!("{} is not a variable of the called file", entry.name);
                let diagnostic =
                    Diagnostic::error(DiagnosticCode::UnknownVariable, "", entry.name_span, message);

                diagnostics.push(diagnostic);
            }
        },
    }
}
