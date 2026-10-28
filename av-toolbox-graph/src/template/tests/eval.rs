use serde_json::{Value, json};

use crate::Span;
use crate::template::{Expr, Literal, LiteralKind, Piece, Scope, Template, VariableRules};

struct TestScope;

impl Scope for TestScope {
    fn variable(&self, name: &str) -> Option<Literal> {
        match name {
            "depth" => Some(Literal::Integer(8)),
            _ => None,
        }
    }

    fn field(&self, stage: &str, field: &str) -> Result<Literal, String> {
        match (stage, field) {
            ("input", "width") => Ok(Literal::Integer(64)),
            _ => Err(format!("no field {stage}.{field}")),
        }
    }
}

/// Turns one-segment paths into variables and two-segment paths into fields.
pub(super) fn resolve_paths(expr: &mut Expr) -> Result<(), String> {
    let Expr::Path(segments) = expr else {
        return Ok(());
    };

    let resolved = match segments.as_slice() {
        [name] => Expr::Variable(name.clone()),
        [stage, field] => Expr::Field {
            stage: stage.clone(),
            field: field.clone(),
        },
        _ => return Err(format!("cannot resolve {segments:?}")),
    };

    *expr = resolved;
    Ok(())
}

fn parse_resolved(text: &str) -> Template {
    let mut template = Template::parse(text).expect("the template parses");
    for piece in template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        found
            .expr
            .rewrite(&mut resolve_paths)
            .expect("every path resolves");
    }

    template
}

fn fill(text: &str) -> Value {
    let template = parse_resolved(text);
    template.fill(&TestScope).expect("the template fills")
}

fn fill_error(text: &str) -> String {
    let template = parse_resolved(text);
    let error = template.fill(&TestScope).expect_err("the template fails to fill");

    error.message
}

#[test]
fn integer_arithmetic_stays_integer() {
    assert_eq!(fill("{{ 7 + 2 * 3 }}"), json!(13));
}

#[test]
fn division_gives_a_float() {
    assert_eq!(fill("{{ 6 / 2 }}"), json!(3.0));
}

#[test]
fn floor_gives_an_integer() {
    assert_eq!(fill("{{ floor(input.width / 3) }}"), json!(21));
}

#[test]
fn round_out_of_range_is_an_error() {
    fill_error("{{ round(1e300) }}");
}

#[test]
fn max_reads_a_field() {
    assert_eq!(fill("{{ max(10, input.width) }}"), json!(64));
}

#[test]
fn mixed_numbers_give_a_float() {
    assert_eq!(fill("{{ 1 + 0.5 }}"), json!(1.5));
}

#[test]
fn if_picks_a_branch() {
    let filled = fill(r#"{{ if depth > 8 then "deep" else "shallow" }}"#);

    assert_eq!(filled, json!("shallow"));
}

#[test]
fn text_with_templates_stays_a_string() {
    assert_eq!(fill("{{ input.width }}px"), json!("64px"));
}

#[test]
fn integer_overflow_is_an_error() {
    fill_error("{{ 9223372036854775807 + 1 }}");
}

#[test]
fn remainder_by_zero_is_an_error() {
    fill_error("{{ 5 % 0 }}");
    fill_error("{{ 5 / 0 }}");
}

#[test]
fn adding_text_to_a_number_is_an_error() {
    let message = fill_error(r#"{{ "a" + 1 }}"#);

    assert!(message.contains("string"), "{message}");
    assert!(message.contains("integer"), "{message}");
}

#[test]
fn an_unrewritten_path_is_an_error() {
    let path = Expr::Path(vec!["input".to_string(), "width".to_string()]);
    let evaluated = path.evaluate(&TestScope);

    assert!(evaluated.is_err());
}

#[test]
fn a_checked_value_outside_its_bounds_names_the_variable() {
    let rules = VariableRules {
        name: "depth".to_string(),
        expected: LiteralKind::Integer,
        min: None,
        max: Some(16.0),
        choices: None,
        span: Span::new(3, 9),
    };
    let checked = Expr::Checked {
        inner: Box::new(Expr::Literal(Literal::Integer(20))),
        rules,
    };

    let error = checked.evaluate(&TestScope).expect_err("20 is above the maximum");

    assert!(error.message.contains("depth"), "{}", error.message);
    assert_eq!(error.checked, Some(Span::new(3, 9)));
}

#[test]
fn a_checked_integer_is_accepted_as_a_float() {
    let rules = VariableRules {
        name: "strength".to_string(),
        expected: LiteralKind::Float,
        min: Some(0.0),
        max: None,
        choices: None,
        span: Span::new(0, 1),
    };
    let checked = Expr::Checked {
        inner: Box::new(Expr::Literal(Literal::Integer(2))),
        rules,
    };

    let evaluated = checked.evaluate(&TestScope).expect("an integer is a valid float");

    assert_eq!(evaluated, Literal::Float(2.0));
}

#[test]
fn a_checked_value_outside_its_choices_is_an_error() {
    let rules = VariableRules {
        name: "mode".to_string(),
        expected: LiteralKind::String,
        min: None,
        max: None,
        choices: Some(vec![Literal::String("fast".to_string())]),
        span: Span::new(0, 1),
    };
    let checked = Expr::Checked {
        inner: Box::new(Expr::Literal(Literal::String("slow".to_string()))),
        rules,
    };

    let error = checked.evaluate(&TestScope).expect_err("slow is not a choice");

    assert!(error.message.contains("fast"), "{}", error.message);
}

/// Rules that accept any integer, for the call variable written at `span`.
fn any_integer(span: Span) -> VariableRules {
    VariableRules {
        name: "width".to_string(),
        expected: LiteralKind::Integer,
        min: None,
        max: None,
        choices: None,
        span,
    }
}

#[test]
fn a_failure_inside_a_checked_value_belongs_to_its_variable() {
    let unknown = Expr::Field {
        stage: "input".to_string(),
        field: "widht".to_string(),
    };
    let checked = Expr::Checked {
        inner: Box::new(unknown),
        rules: any_integer(Span::new(3, 9)),
    };

    let error = checked.evaluate(&TestScope).expect_err("widht is no field");

    assert_eq!(error.variable, Some(Span::new(3, 9)));
    assert_eq!(error.checked, None);
}

#[test]
fn a_failure_belongs_to_the_innermost_variable() {
    let unknown = Expr::Field {
        stage: "input".to_string(),
        field: "widht".to_string(),
    };
    let inner = Expr::Checked {
        inner: Box::new(unknown),
        rules: any_integer(Span::new(3, 9)),
    };
    let outer = Expr::Checked {
        inner: Box::new(inner),
        rules: any_integer(Span::new(20, 30)),
    };

    let error = outer.evaluate(&TestScope).expect_err("widht is no field");

    assert_eq!(error.variable, Some(Span::new(3, 9)));
}

#[test]
fn a_failed_fill_points_at_its_template() {
    let template = parse_resolved("ab{{ unknown }}");
    let error = template.fill(&TestScope).expect_err("the variable is unknown");

    assert_eq!(error.range, 2..15);
    assert_eq!(error.checked, None);
}

#[test]
fn logic_short_circuits() {
    assert_eq!(fill("{{ false and unknown }}"), json!(false));
    assert_eq!(fill("{{ not (depth == 8) or true }}"), json!(true));
    assert_eq!(fill("{{ depth == 8 or unknown }}"), json!(true));
}
