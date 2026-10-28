use av_toolbox_core::StageKind;
use serde_json::json;

use crate::{
    ArgSpans,
    DeferredArg,
    Expr,
    InputRef,
    LiteralKind,
    Piece,
    Span,
    Spanned,
    StageSpec,
    Template,
    VariableRules,
};

/// Builds a spec with no inputs, empty arguments and unknown spans.
///
/// Every field is public, so a test that cares about a span or an argument sets it directly.
pub(crate) fn spec(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        arg_spans: ArgSpans::default(),
        workers: None,
        deferred: Vec::new(),
        span: Span::unknown(),
    }
}

pub(crate) fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

pub(crate) fn named(name: &str, target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: Some(name.to_string()),
        target: target.to_string(),
    })
}

/// Builds a deferred arg at `pointer` from template `text`, reading each `stage.field` path as a field.
pub(crate) fn deferred(pointer: &str, text: &str) -> DeferredArg {
    let mut template = Template::parse(text).expect("the template parses");
    for piece in template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        found.expr.rewrite(&mut path_to_field).expect("the paths rewrite");
    }

    DeferredArg {
        pointer: pointer.to_string(),
        template,
        span: Span::unknown(),
    }
}

/// Wraps every expression in `deferred` as if a call passed it in a float variable written at `span`.
pub(crate) fn through_variable(deferred: &mut DeferredArg, span: Span) {
    for piece in deferred.template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        let inner = found.expr.clone();
        found.expr = Expr::Checked {
            inner: Box::new(inner),
            rules: VariableRules {
                name: String::from("passed"),
                expected: LiteralKind::Float,
                min: None,
                max: None,
                choices: None,
                span,
            },
        };
    }
}

fn path_to_field(expr: &mut Expr) -> Result<(), String> {
    let Expr::Path(segments) = expr else {
        return Ok(());
    };

    let [stage, field] = segments.as_slice() else {
        return Ok(());
    };

    *expr = Expr::Field {
        stage: stage.clone(),
        field: field.clone(),
    };

    Ok(())
}
