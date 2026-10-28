use std::ops::Range;

use av_toolbox_graph::{Expr, Literal, Piece, Scope, Span, Template, TemplateError};
use serde_json::Value;

use super::files::SourceMap;
use crate::variables::Values;

const STAGE_REFERENCE_MESSAGE: &str = "stage references are not supported here";

/// Why a string's templates could not be filled, with where inside the string to point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FillFailure {
    Unknown { name: String },
    Invalid { message: String, range: Range<usize> },
}

/// Reads variables from the values a file resolved.
struct ValueScope<'a> {
    values: &'a Values,
}

impl Scope for ValueScope<'_> {
    fn variable(&self, name: &str) -> Option<Literal> {
        let value = self.values.get(name)?;
        Some(value.to_literal())
    }

    fn field(&self, _stage: &str, _field: &str) -> Result<Literal, String> {
        Err(STAGE_REFERENCE_MESSAGE.to_string())
    }
}

/// The value `text` becomes once its templates are filled from `values`.
pub(super) fn fill_template(text: &str, values: &Values) -> Result<Value, FillFailure> {
    let parsed = Template::parse(text);
    let mut template = parsed.map_err(invalid_template)?;

    resolve_paths(&mut template, values)?;

    let scope = ValueScope { values };
    let filled = template.fill(&scope);

    filled.map_err(invalid_template)
}

/// Turns every path naming a variable into a variable read.
fn resolve_paths(template: &mut Template, values: &Values) -> Result<(), FillFailure> {
    for piece in template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        let mut unknown = None;
        let mut visit = |expr: &mut Expr| resolve_path(expr, values, &mut unknown);
        let rewritten = found.expr.rewrite(&mut visit);
        if let Err(message) = rewritten {
            return Err(FillFailure::Invalid {
                message,
                range: found.range.clone(),
            });
        }

        if let Some(name) = unknown {
            return Err(FillFailure::Unknown { name });
        }
    }

    Ok(())
}

/// Rewrites a one-segment path into a variable read, remembering the first name `values` lacks.
///
/// A path with several segments would name a stage field, which cannot be read here.
fn resolve_path(expr: &mut Expr, values: &Values, unknown: &mut Option<String>) -> Result<(), String> {
    let Expr::Path(segments) = expr else {
        return Ok(());
    };

    let [name] = segments.as_slice() else {
        return Err(STAGE_REFERENCE_MESSAGE.to_string());
    };

    if !values.contains_key(name) && unknown.is_none() {
        *unknown = Some(name.clone());
    }

    *expr = Expr::Variable(name.clone());

    Ok(())
}

fn invalid_template(error: TemplateError) -> FillFailure {
    FillFailure::Invalid {
        message: error.message,
        range: error.range,
    }
}

/// Narrows `span`, which covers the string `text`, to `range` inside it.
///
/// This only works for a TOML basic string without escapes up to the end of `range`, since only
/// then does a position in `text` match one in the file. Otherwise the whole span is kept.
pub(super) fn narrowed(sources: &SourceMap, text: &str, span: Span, range: &Range<usize>) -> Span {
    let source = sources.get(span.file);
    let written = source.and_then(|source| source.text.get(span.start..span.end));
    let covered = text.get(..range.end);

    let (Some(written), Some(covered)) = (written, covered) else {
        return span;
    };

    let inside = written
        .strip_prefix('"')
        .filter(|inside| !inside.starts_with("\"\""));
    let Some(inside) = inside else {
        return span;
    };

    if !inside.starts_with(covered) {
        return span;
    }

    let start = span.start + 1 + range.start;
    let end = span.start + 1 + range.end;

    Span::in_file(span.file, start, end)
}
