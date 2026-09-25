use serde_json::json;

use crate::variables::{FillError, Piece, TemplateError, Values, VariableValue, fill, pieces};

fn values() -> Values {
    let mut values = Values::new();
    values.insert("sigma".into(), VariableValue::Float(0.8));
    values.insert("dir".into(), VariableValue::Path("/out".into()));

    values
}

#[test]
fn text_without_templates_is_one_piece() {
    assert_eq!(pieces("plain"), Ok(vec![Piece::Text("plain")]));
}

#[test]
fn spaces_inside_the_braces_are_optional() {
    assert_eq!(pieces("{{sigma}}"), Ok(vec![Piece::Variable("sigma")]));
    assert_eq!(pieces("{{  sigma }}"), Ok(vec![Piece::Variable("sigma")]));
}

#[test]
fn an_unclosed_template_is_refused() {
    assert_eq!(pieces("{{ sigma"), Err(TemplateError::Unclosed));
}

#[test]
fn an_empty_template_is_refused() {
    assert_eq!(pieces("a {{ }} b"), Err(TemplateError::EmptyName));
}

#[test]
fn a_lone_template_keeps_its_type() {
    assert_eq!(fill("{{ sigma }}", &values()), Ok(json!(0.8)));
}

#[test]
fn a_template_inside_text_gives_a_string() {
    assert_eq!(fill("{{ dir }}/out.y4m", &values()), Ok(json!("/out/out.y4m")));
    assert_eq!(fill("s={{ sigma }}", &values()), Ok(json!("s=0.8")));
}

#[test]
fn an_unknown_name_is_reported() {
    assert_eq!(
        fill("{{ missing }}", &values()),
        Err(FillError::Unknown("missing".into()))
    );
}
