use crate::template::{BinaryOp, Expr, Literal, Piece, Template};

fn only_expr(text: &str) -> Expr {
    let template = Template::parse(text).expect("the template parses");

    match template.pieces() {
        [Piece::Expr(found)] => found.expr.clone(),
        other => panic!("expected one expression, found {other:?}"),
    }
}

fn integer(value: i64) -> Box<Expr> {
    Box::new(Expr::Literal(Literal::Integer(value)))
}

#[test]
fn a_lone_name_parses_to_a_path() {
    let expr = only_expr("{{ input }}");

    assert_eq!(expr, Expr::Path(vec!["input".to_string()]));
}

#[test]
fn text_around_templates_is_kept() {
    let template = Template::parse("a{{ x }}b").expect("the template parses");

    let [Piece::Text(before), Piece::Expr(middle), Piece::Text(after)] = template.pieces() else {
        panic!(
            "expected text, an expression and text, found {:?}",
            template.pieces()
        );
    };

    assert_eq!(before, "a");
    assert_eq!(middle.expr, Expr::Path(vec!["x".to_string()]));
    assert_eq!(middle.range, 1..8);
    assert_eq!(after, "b");
}

#[test]
fn multiplication_binds_tighter_than_addition() {
    let expr = only_expr("{{ 1 + 2 * 3 }}");

    let product = Expr::Binary {
        op: BinaryOp::Multiply,
        left: integer(2),
        right: integer(3),
    };
    let expected = Expr::Binary {
        op: BinaryOp::Add,
        left: integer(1),
        right: Box::new(product),
    };

    assert_eq!(expr, expected);
}

#[test]
fn a_dotted_name_keeps_every_segment() {
    let expr = only_expr("{{ denoise.clean.width }}");

    let segments = vec!["denoise".to_string(), "clean".to_string(), "width".to_string()];
    assert_eq!(expr, Expr::Path(segments));
}

#[test]
fn an_unclosed_template_is_an_error() {
    let error = Template::parse("{{ x").expect_err("the template is unclosed");

    assert_eq!(error.message, "a template opened with {{ is never closed with }}");
}

#[test]
fn an_empty_template_is_an_error() {
    let parsed = Template::parse("{{ }}");

    assert!(parsed.is_err());
}

#[test]
fn an_unknown_function_lists_the_known_ones() {
    let error = Template::parse("{{ clamp(1) }}").expect_err("clamp is not a function");

    assert!(
        error.message.contains("min, max, round, floor, ceil, abs"),
        "{}",
        error.message
    );
}

#[test]
fn an_error_points_inside_the_template() {
    let error = Template::parse("ab{{ 1 + }}").expect_err("the addition has no right side");

    assert!(error.range.start >= 2, "{:?}", error.range);
    assert_eq!(error.checked, None);
}

#[test]
fn comparisons_do_not_chain() {
    let parsed = Template::parse("{{ 1 < 2 < 3 }}");

    assert!(parsed.is_err());
}

#[test]
fn if_binds_loosest() {
    let expr = only_expr("{{ if a or b then 1 + 2 else -3 }}");

    let Expr::If { condition, .. } = expr else {
        panic!("expected an if, found {expr:?}");
    };
    assert!(matches!(*condition, Expr::Binary { op: BinaryOp::Or, .. }));
}

#[test]
fn exponent_floats_are_read() {
    let expr = only_expr("{{ 2.5e-3 }}");

    assert_eq!(expr, Expr::Literal(Literal::Float(2.5e-3)));
}

#[test]
fn an_integer_too_large_is_an_error() {
    let parsed = Template::parse("{{ 9223372036854775808 }}");

    assert!(parsed.is_err());
}

#[test]
fn closing_braces_inside_a_string_do_not_close_the_template() {
    let expr = only_expr(r#"{{ "}}" }}"#);

    assert_eq!(expr, Expr::Literal(Literal::String("}}".to_string())));
}

#[test]
fn stages_lists_every_field_in_order() {
    let mut template =
        Template::parse("{{ b.width }}x{{ a.height + b.height }}").expect("the template parses");
    for piece in template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        found
            .expr
            .rewrite(&mut super::eval::resolve_paths)
            .expect("every path resolves");
    }

    assert_eq!(template.stages(), vec!["b", "a", "b"]);
}

#[test]
fn a_non_ascii_character_is_an_error() {
    let error = Template::parse("{{ é }}").expect_err("é is not part of the language");

    assert!(
        error.message.contains("unexpected character"),
        "{}",
        error.message
    );
}

#[test]
fn none_parses_as_a_literal() {
    let expr = only_expr("{{ none }}");

    assert_eq!(expr, Expr::Literal(Literal::None { variable: None }));
}

#[test]
fn is_none_and_is_not_none_parse() {
    let operand = Box::new(Expr::Path(vec!["x".to_string()]));

    let is_none = only_expr("{{ x is none }}");
    let expected = Expr::IsNone {
        operand: operand.clone(),
        negated: false,
    };
    assert_eq!(is_none, expected);

    let is_not_none = only_expr("{{ x is not none }}");
    let expected = Expr::IsNone {
        operand,
        negated: true,
    };
    assert_eq!(is_not_none, expected);
}

#[test]
fn comparing_with_the_none_keyword_suggests_is() {
    for text in ["{{ x == none }}", "{{ none != x }}", "{{ x < none }}"] {
        let error = Template::parse(text).expect_err("comparing with none is refused");

        assert!(error.message.contains("is none"), "{text}: {}", error.message);
    }
}

#[test]
fn is_only_tests_for_none() {
    let error = Template::parse("{{ x is 3 }}").expect_err("is needs none");

    assert!(error.message.contains("is none"), "{}", error.message);
}
