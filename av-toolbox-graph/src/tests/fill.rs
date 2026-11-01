use av_toolbox_core::StageKind;
use serde_json::{Value, json};

use crate::graph::CompiledGraph;
use crate::graph::tests::node_named;
use crate::template::{Expr, LiteralKind, Piece, VariableRules};
use crate::{
    CommonStageElimination,
    DiagnosticCode,
    Diagnostics,
    OptimisationPass,
    Span,
    Spanned,
    StageSpec,
    build,
    fill_references,
    residency,
    testing,
};

fn sized(name: &str, width: u32, height: u32) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/sized");
    spec.args = Spanned::unknown(json!({ "width": width, "height": height }));

    spec
}

/// A resize reading `from` whose width is the template `text`.
fn resize(name: &str, from: &str, text: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/resize");
    spec.inputs = vec![testing::positional(from)];
    spec.args = Spanned::unknown(json!({ "width": text }));
    spec.deferred = vec![testing::deferred("/width", text)];

    spec
}

fn fixed_resize(name: &str, from: &str, width: u32) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/resize");
    spec.inputs = vec![testing::positional(from)];
    spec.args = Spanned::unknown(json!({ "width": width }));

    spec
}

fn unproducing(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/unproducing");
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn sink(name: &str, from: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(from)];

    spec
}

fn built(specs: &[StageSpec]) -> CompiledGraph {
    build(&testing::registry(), specs)
        .expect("should build")
        .into_graph()
}

fn filled(specs: &[StageSpec]) -> CompiledGraph {
    let mut graph = built(specs);
    let capabilities = residency::tests::capabilities();
    fill_references(&mut graph, &testing::registry(), &capabilities).expect("should fill");

    graph
}

fn fill_failure(specs: &[StageSpec]) -> Diagnostics {
    let mut graph = built(specs);
    let capabilities = residency::tests::capabilities();

    fill_references(&mut graph, &testing::registry(), &capabilities).expect_err("should fail")
}

fn width_of(graph: &CompiledGraph, name: &str) -> Value {
    let id = node_named(graph, name);

    graph.node(id).args()["width"].clone()
}

#[test]
fn a_reference_fills_from_the_natural_format() {
    let specs = vec![
        sized("input", 64, 48),
        resize("scaled", "input", "{{ input.width }}"),
        sink("out", "scaled"),
    ];

    let graph = filled(&specs);
    let scaled = node_named(&graph, "scaled");

    assert_eq!(width_of(&graph, "scaled"), json!(64));
    assert!(!graph.node(scaled).is_deferred());
}

#[test]
fn filling_returns_the_nodes_filled() {
    let specs = vec![
        sized("input", 64, 48),
        resize("scaled", "input", "{{ input.width }}"),
        sink("out", "scaled"),
    ];
    let mut graph = built(&specs);
    let capabilities = residency::tests::capabilities();

    let filled = fill_references(&mut graph, &testing::registry(), &capabilities).expect("should fill");

    assert_eq!(filled, vec![node_named(&graph, "scaled")]);
}

#[test]
fn mixed_text_fills_to_a_string() {
    let specs = vec![
        sized("input", 64, 48),
        resize("scaled", "input", "{{ input.width }}x{{ input.height }}"),
        sink("out", "scaled"),
    ];

    let graph = filled(&specs);

    assert_eq!(width_of(&graph, "scaled"), json!("64x48"));
}

#[test]
fn a_chain_of_references_fills_in_order() {
    let specs = vec![
        sized("input", 64, 48),
        fixed_resize("a", "input", 100),
        resize("b", "a", "{{ a.width * 2 }}"),
        resize("c", "b", "{{ b.width + 1 }}"),
        sink("out", "c"),
    ];

    let graph = filled(&specs);

    assert_eq!(width_of(&graph, "b"), json!(200));
    assert_eq!(width_of(&graph, "c"), json!(201));
}

#[test]
fn a_failing_natural_format_names_both_stages() {
    let mut reading = resize("reading", "input", "{{ broken.width }}");
    reading.deferred[0].span = Span::new(40, 60);

    let specs = vec![
        sized("input", 64, 48),
        unproducing("broken", "input"),
        reading,
        sink("out", "reading"),
        sink("other", "broken"),
    ];

    let diagnostics = fill_failure(&specs);
    let failed = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::StageBuildFailed)
        .expect("a build failure");

    assert!(failed.path().contains("broken"), "{}", failed.path());
    assert!(
        failed.message().contains("cannot say what it produces"),
        "{}",
        failed.message()
    );
    assert!(
        failed
            .notes()
            .iter()
            .any(|note| note.message.contains("reading") && note.span == Span::new(40, 60)),
        "{:?}",
        failed.notes(),
    );
}

#[test]
fn an_unknown_field_is_an_invalid_expression() {
    let mut scaled = resize("scaled", "input", "{{ input.widht }}");
    scaled.deferred[0].span = Span::new(10, 30);

    let specs = vec![sized("input", 64, 48), scaled, sink("out", "scaled")];

    let diagnostics = fill_failure(&specs);
    let invalid = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::InvalidExpression)
        .expect("an invalid expression");

    assert!(invalid.message().contains("width"), "{}", invalid.message());
    assert_eq!(invalid.span(), Span::new(10, 30));
}

#[test]
fn every_deferred_node_is_attempted() {
    let specs = vec![
        sized("input", 64, 48),
        resize("first", "input", "{{ input.widht }}"),
        resize("second", "first", "{{ input.heigth }}"),
        sink("out", "second"),
    ];

    let diagnostics = fill_failure(&specs);
    let invalid = diagnostics
        .iter()
        .filter(|entry| entry.code() == DiagnosticCode::InvalidExpression)
        .count();

    assert_eq!(invalid, 2);
}

#[test]
fn a_late_variable_outside_its_bounds_points_at_the_call() {
    let rules_span = Span::new(5, 9);
    let mut scaled = resize("scaled", "input", "{{ input.width }}");
    for piece in scaled.deferred[0].template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        let inner = found.expr.clone();
        found.expr = Expr::Checked {
            inner: Box::new(inner),
            rules: VariableRules {
                name: String::from("width"),
                expected: LiteralKind::Integer,
                min: None,
                max: Some(16.0),
                choices: None,
                optional: false,
                span: rules_span,
            },
        };
    }

    let specs = vec![sized("input", 64, 48), scaled, sink("out", "scaled")];

    let diagnostics = fill_failure(&specs);
    let invalid = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::InvalidVariable)
        .expect("an invalid variable");

    assert_eq!(invalid.span(), rules_span);
}

#[test]
fn a_stage_reading_an_unfilled_stage_is_not_reported_again() {
    let specs = vec![
        sized("input", 64, 48),
        resize("first", "input", "{{ input.widht }}"),
        resize("second", "first", "{{ first.width }}"),
        sink("out", "second"),
    ];

    let diagnostics = fill_failure(&specs);

    assert_eq!(diagnostics.iter().count(), 1);
}

#[test]
fn a_reference_to_a_merged_stage_fills_from_the_kept_stage() {
    let specs = vec![
        sized("a", 64, 48),
        sized("b", 64, 48),
        resize("scaled", "a", "{{ b.width }}"),
        sink("out", "scaled"),
    ];
    let mut graph = built(&specs);
    CommonStageElimination.run(&mut graph);

    let kept = graph
        .nodes()
        .iter()
        .find(|node| node.name() == "a" || node.name() == "b")
        .expect("one source survives")
        .name()
        .to_string();
    let scaled = node_named(&graph, "scaled");
    let named = graph.node(scaled).deferred()[0].template.stages();
    assert_eq!(named, vec![kept.as_str()]);

    let capabilities = residency::tests::capabilities();
    fill_references(&mut graph, &testing::registry(), &capabilities).expect("should fill");

    assert_eq!(width_of(&graph, "scaled"), json!(64));
}

/// Fills a resize whose width template `text` came through a call variable written at `variable`.
fn fill_through_variable(text: &str, variable: Span) -> Diagnostics {
    let mut scaled = resize("scaled", "input", text);
    scaled.deferred[0].span = Span::new(10, 30);
    testing::through_variable(&mut scaled.deferred[0], variable);

    let specs = vec![sized("input", 64, 48), scaled, sink("out", "scaled")];

    fill_failure(&specs)
}

fn assert_at_the_variable(diagnostics: &Diagnostics, variable: Span) {
    let invalid = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::InvalidExpression)
        .expect("an invalid expression");
    let used: Vec<Span> = invalid
        .notes()
        .iter()
        .filter(|note| note.message == "used here")
        .map(|note| note.span)
        .collect();

    assert_eq!(invalid.span(), variable);
    assert_eq!(used, vec![Span::new(10, 30)]);
}

#[test]
fn an_unknown_field_in_a_call_variable_points_at_the_variable() {
    let variable = Span::new(5, 9);

    let diagnostics = fill_through_variable("{{ input.widht }}", variable);

    assert_at_the_variable(&diagnostics, variable);
}

#[test]
fn a_division_by_zero_in_a_call_variable_points_at_the_variable() {
    let variable = Span::new(5, 9);

    let diagnostics = fill_through_variable("{{ input.width % 0 }}", variable);

    assert!(
        diagnostics
            .iter()
            .any(|entry| entry.message().contains("divides by zero")),
        "{diagnostics:?}"
    );
    assert_at_the_variable(&diagnostics, variable);
}

#[test]
fn a_call_variable_failing_in_two_stages_is_reported_once() {
    let variable = Span::new(5, 9);
    let mut first = resize("first", "input", "{{ input.widht }}");
    testing::through_variable(&mut first.deferred[0], variable);
    let mut second = resize("second", "input", "{{ input.widht }}");
    testing::through_variable(&mut second.deferred[0], variable);

    let specs = vec![
        sized("input", 64, 48),
        first,
        second,
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let diagnostics = fill_failure(&specs);

    assert_eq!(diagnostics.iter().count(), 1);
}

#[test]
fn a_filled_stage_failing_to_produce_notes_its_template() {
    let mut broken = unproducing("broken", "input");
    broken.args = Spanned::unknown(json!({ "note": "{{ input.width }}" }));
    broken.deferred = vec![testing::deferred("/note", "{{ input.width }}")];
    broken.deferred[0].span = Span::new(12, 20);

    let specs = vec![
        sized("input", 64, 48),
        broken,
        resize("reading", "input", "{{ broken.width }}"),
        sink("out", "reading"),
        sink("other", "broken"),
    ];

    let diagnostics = fill_failure(&specs);
    let failed = diagnostics
        .iter()
        .find(|entry| entry.code() == DiagnosticCode::StageBuildFailed)
        .expect("a build failure");

    assert!(
        failed
            .notes()
            .iter()
            .any(|note| note.message == "filled from this template" && note.span == Span::new(12, 20)),
        "{:?}",
        failed.notes(),
    );
}
