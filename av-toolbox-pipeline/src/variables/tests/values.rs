use std::path::Path;

use av_toolbox_graph::{Span, Spanned};
use serde_json::json;

use super::{declaration, spanned};
use crate::variables::{
    VariableType,
    VariableValue,
    check_rules,
    from_json,
    is_variable_name,
    parse_text,
    resolve_path,
};

#[test]
fn each_type_reads_its_text_form() {
    let directory = Path::new("/work");

    assert_eq!(
        parse_text(VariableType::String, "hello", directory),
        Ok(VariableValue::String("hello".into()))
    );
    assert_eq!(
        parse_text(VariableType::Integer, "-4", directory),
        Ok(VariableValue::Integer(-4))
    );
    assert_eq!(
        parse_text(VariableType::Float, "0.5", directory),
        Ok(VariableValue::Float(0.5))
    );
    assert_eq!(
        parse_text(VariableType::Bool, "true", directory),
        Ok(VariableValue::Bool(true))
    );
    assert_eq!(
        parse_text(VariableType::Path, "in.mkv", directory),
        Ok(VariableValue::Path("/work/in.mkv".into()))
    );
}

#[test]
fn bad_text_is_refused() {
    let directory = Path::new("/work");

    assert!(parse_text(VariableType::Integer, "1.5", directory).is_err());
    assert!(parse_text(VariableType::Float, "inf", directory).is_err());
    assert!(parse_text(VariableType::Float, "abc", directory).is_err());
    assert!(parse_text(VariableType::Bool, "yes", directory).is_err());
}

#[test]
fn an_absolute_path_is_kept() {
    let directory = Path::new("/work");

    assert_eq!(resolve_path(directory, "/data/in.mkv"), "/data/in.mkv");
}

#[test]
fn a_float_accepts_a_whole_number() {
    let directory = Path::new("/work");

    assert_eq!(
        from_json(VariableType::Float, &json!(1), directory),
        Ok(VariableValue::Float(1.0))
    );
}

#[test]
fn a_value_of_the_wrong_json_type_is_refused() {
    let directory = Path::new("/work");

    let refused = from_json(VariableType::Integer, &json!("four"), directory);

    assert!(
        refused
            .expect_err("a string is not an integer")
            .contains("integer")
    );
}

#[test]
fn bounds_accept_values_inside_and_refuse_values_outside() {
    let mut float = declaration(VariableType::Float);
    float.min = spanned(json!(0.0));
    float.max = spanned(json!(4));

    assert!(check_rules(&float, &VariableValue::Float(4.0)).is_ok());
    assert!(check_rules(&float, &VariableValue::Float(-0.1)).is_err());
    assert!(check_rules(&float, &VariableValue::Float(4.1)).is_err());
}

#[test]
fn choices_accept_listed_values_only() {
    let mut mode = declaration(VariableType::String);
    let listed = vec![json!("natural"), json!("fast")];
    mode.choices = Some(Spanned::new(listed, Span::new(1, 2)));

    assert!(check_rules(&mode, &VariableValue::String("fast".into())).is_ok());

    let refused = check_rules(&mode, &VariableValue::String("slow".into()));

    assert!(
        refused
            .expect_err("slow is not a choice")
            .contains("natural, fast")
    );
}

#[test]
fn values_render_as_json_and_text() {
    assert_eq!(VariableValue::Float(0.8).to_json(), json!(0.8));
    assert_eq!(VariableValue::Integer(3).to_text(), "3");
    assert_eq!(VariableValue::Bool(false).to_text(), "false");
    assert_eq!(VariableValue::Path("/a/b".into()).to_json(), json!("/a/b"));
}

#[test]
fn variable_names_are_lower_case_digits_and_underscores() {
    assert!(is_variable_name("clip_input2"));
    assert!(!is_variable_name("Clip"));
    assert!(!is_variable_name("clip-input"));
    assert!(!is_variable_name(""));
}
