use av_toolbox_graph::DiagnosticCode;
use serde_json::json;

use super::{locate, parsed, reported};
use crate::variables::VariableType;

#[test]
fn a_variable_reads_every_key() {
    let text = r#"
[variable.sigma]
type = "float"
description = "How strong"
default = 0.8
min = 0
max = 4.5
"#;

    let file = parsed(text);
    let sigma = &file.variables[0];

    assert_eq!(sigma.name.value, "sigma");
    assert_eq!(sigma.name.span, locate(text, "[variable.sigma]"));
    assert_eq!(sigma.variable_type, VariableType::Float);
    assert_eq!(sigma.description.as_deref(), Some("How strong"));
    assert_eq!(
        sigma.default.as_ref().map(|default| &default.value),
        Some(&json!(0.8))
    );
    assert_eq!(sigma.min.as_ref().map(|min| &min.value), Some(&json!(0)));
    assert_eq!(sigma.max.as_ref().map(|max| &max.value), Some(&json!(4.5)));
}

#[test]
fn choices_are_read_as_a_list() {
    let file = parsed("[variable.mode]\ntype = \"string\"\nchoices = [\"a\", \"b\"]\n");
    let choices = file.variables[0]
        .choices
        .as_ref()
        .map(|choices| choices.value.clone());

    assert_eq!(choices, Some(vec![json!("a"), json!("b")]));
}

#[test]
fn a_missing_type_is_refused_on_the_header() {
    let text = "[variable.a]\ndefault = 1\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(diagnostics[0].span(), locate(text, "[variable.a]"));
}

#[test]
fn an_unknown_type_is_refused() {
    let diagnostics = reported("[variable.a]\ntype = \"number\"\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert!(
        diagnostics[0]
            .message()
            .contains("string, path, integer, float or bool")
    );
}

#[test]
fn an_upper_case_name_is_refused() {
    let diagnostics = reported("[variable.Clip]\ntype = \"path\"\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn a_bound_on_a_string_is_refused_on_its_key() {
    let text = "[variable.a]\ntype = \"string\"\nmin = 1\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(diagnostics[0].span(), locate(text, "min"));
}

#[test]
fn choices_on_a_float_are_refused_on_the_key() {
    let text = "[variable.a]\ntype = \"float\"\nchoices = [1.0]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics[0].span(), locate(text, "choices"));
}

#[test]
fn a_float_bound_on_an_integer_is_refused() {
    let diagnostics = reported("[variable.a]\ntype = \"integer\"\nmax = 1.5\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn empty_choices_are_refused() {
    let diagnostics = reported("[variable.a]\ntype = \"string\"\nchoices = []\n");

    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
}

#[test]
fn an_unknown_variable_key_lists_the_allowed_keys() {
    let diagnostics = reported("[variable.a]\ntype = \"string\"\nfoo = 1\n");

    assert!(
        diagnostics[0].notes()[0]
            .message
            .contains("type, description, default")
    );
}

#[test]
fn a_datetime_default_is_refused_without_naming_stages() {
    let diagnostics = reported("[variable.a]\ntype = \"string\"\ndefault = 1979-05-27\n");

    assert_eq!(diagnostics[0].message(), "dates and times are not supported here");
}

#[test]
fn a_template_in_a_default_is_refused_on_its_value() {
    let text = "[variable.a]\ntype = \"string\"\ndefault = \"{{ b }}\"\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(
        diagnostics[0].message(),
        "templates cannot be used in a variable's default"
    );
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ b }}\""));
}

#[test]
fn a_template_in_choices_is_refused_on_its_choice() {
    let text = "[variable.a]\ntype = \"string\"\nchoices = [\"x\", \"{{ b\"]\n";

    let diagnostics = reported(text);

    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].code(), DiagnosticCode::InvalidDefinition);
    assert_eq!(
        diagnostics[0].message(),
        "templates cannot be used in a variable's choices"
    );
    assert_eq!(diagnostics[0].span(), locate(text, "\"{{ b\""));
}
