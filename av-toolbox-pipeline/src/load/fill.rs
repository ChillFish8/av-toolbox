use std::ops::Range;

use av_toolbox_graph::{
    ArgSpans,
    DeferredArg,
    Diagnostic,
    DiagnosticCode,
    Expr,
    Literal,
    Piece,
    Scope,
    Span,
    Template,
    TemplateError,
    suggest_name,
};
use serde_json::{Map, Value};

use super::expand::Loader;
use super::files::SourceMap;
use crate::variables::{Bound, Values};

const REFERENCE_OUTSIDE_ARGS_MESSAGE: &str = "stage references can only be used in args";

/// What filling templates needs, and the path problems are reported under.
pub(super) struct Filling<'a> {
    pub(super) values: &'a Values,
    /// The file's declared variable names, in the order they are declared.
    pub(super) declared: &'a [&'a str],
    /// Put in front of every stage name a template reads, such as `outer.inner.`.
    pub(super) prefix: &'a str,
    pub(super) path: &'a str,
}

/// A string once its templates are filled as far as loading can.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Filled {
    Ready(Value),
    /// The template reads other stages, so it is filled once their formats are known.
    ///
    /// Its variables are already inlined and its stage names carry the file's prefix.
    Deferred(Template),
}

/// Why a string's templates could not be filled, with where inside the string to point.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum FillFailure {
    Unknown { name: String },
    Invalid { message: String, range: Range<usize> },
}

/// The scope of a template whose variables are already inlined and which reads no stage.
struct InlinedScope;

impl Scope for InlinedScope {
    fn variable(&self, _name: &str) -> Option<Literal> {
        None
    }

    fn field(&self, stage: &str, field: &str) -> Result<Literal, String> {
        let message = format!("{stage}.{field} cannot be read while loading");
        Err(message)
    }
}

impl Loader<'_> {
    /// Fills every template under `value`, or `None` after reporting each that cannot be.
    ///
    /// A template reading other stages keeps its text and is pushed onto `deferred` instead.
    /// A template is reported at its span in `arg_spans`, or at `fallback` when it has none.
    pub(super) fn fill_value(
        &mut self,
        value: &Value,
        pointer: &str,
        arg_spans: &ArgSpans,
        fallback: Span,
        filling: &Filling<'_>,
        deferred: &mut Vec<DeferredArg>,
    ) -> Option<Value> {
        match value {
            Value::String(text) => {
                let span = arg_spans.narrowest(pointer).unwrap_or(fallback);
                let filled = self.fill_string(text, span, filling)?;

                match filled {
                    Filled::Ready(ready) => Some(ready),
                    Filled::Deferred(template) => {
                        deferred.push(DeferredArg {
                            pointer: pointer.to_string(),
                            template,
                            span,
                        });

                        Some(value.clone())
                    },
                }
            },
            Value::Array(items) => {
                let mut filled = Vec::with_capacity(items.len());
                let mut failed = false;

                for (index, item) in items.iter().enumerate() {
                    let nested = format!("{pointer}/{index}");
                    let converted = self.fill_value(item, &nested, arg_spans, fallback, filling, deferred);
                    match converted {
                        Some(converted) => filled.push(converted),
                        None => failed = true,
                    }
                }

                (!failed).then_some(Value::Array(filled))
            },
            Value::Object(entries) => {
                let mut filled = Map::new();
                let mut failed = false;

                for (key, item) in entries {
                    let escaped = key.replace('~', "~0").replace('/', "~1");
                    let nested = format!("{pointer}/{escaped}");
                    let converted = self.fill_value(item, &nested, arg_spans, fallback, filling, deferred);
                    match converted {
                        Some(converted) => {
                            filled.insert(key.clone(), converted);
                        },
                        None => failed = true,
                    }
                }

                (!failed).then_some(Value::Object(filled))
            },
            other => Some(other.clone()),
        }
    }

    /// Fills `text`'s templates, refusing one that reads other stages, or `None` after reporting.
    pub(super) fn fill_now(&mut self, text: &str, span: Span, filling: &Filling<'_>) -> Option<Value> {
        let filled = self.fill_string(text, span, filling)?;

        match filled {
            Filled::Ready(value) => Some(value),
            Filled::Deferred(_) => {
                let diagnostic = Diagnostic::error(
                    DiagnosticCode::InvalidDefinition,
                    filling.path,
                    span,
                    REFERENCE_OUTSIDE_ARGS_MESSAGE,
                );

                self.diagnostics.push(diagnostic);
                None
            },
        }
    }

    /// Fills `text`'s templates, or `None` after reporting the problem.
    ///
    /// A template naming a variable that was declared but could not be resolved is left
    /// unreported here, since resolving it already reported the problem.
    pub(super) fn fill_string(&mut self, text: &str, span: Span, filling: &Filling<'_>) -> Option<Filled> {
        let filled = fill_template(text, filling.values, filling.prefix);

        match filled {
            Ok(filled) => Some(filled),
            Err(FillFailure::Unknown { name }) => {
                if filling.declared.contains(&name.as_str()) {
                    return None;
                }

                let message = format!("no variable named {name} is declared in this file");
                let mut diagnostic =
                    Diagnostic::error(DiagnosticCode::UnknownVariable, filling.path, span, message);

                let closest = suggest_name(&name, filling.declared.iter().copied());
                if let Some(closest) = closest {
                    let note = format!("did you mean `{closest}`?");
                    diagnostic = diagnostic.with_note(note, Span::unknown());
                }

                self.diagnostics.push(diagnostic);
                None
            },
            Err(FillFailure::Invalid { message, range }) => {
                let narrow = narrowed(&self.sources, text, span, &range);
                let diagnostic =
                    Diagnostic::error(DiagnosticCode::InvalidDefinition, filling.path, narrow, message);

                self.diagnostics.push(diagnostic);
                None
            },
        }
    }
}

/// What `text` becomes once its variables from `values` are filled in.
///
/// Stage names the template reads get `prefix` in front.
pub(super) fn fill_template(text: &str, values: &Values, prefix: &str) -> Result<Filled, FillFailure> {
    let parsed = Template::parse(text);
    let mut template = parsed.map_err(invalid_template)?;

    resolve_paths(&mut template, values, prefix)?;

    let reads_stages = !template.stages().is_empty();
    if reads_stages {
        return Ok(Filled::Deferred(template));
    }

    let filled = template.fill(&InlinedScope);
    let value = filled.map_err(invalid_template)?;

    Ok(Filled::Ready(value))
}

/// Rewrites every path into the variable or stage field it names.
fn resolve_paths(template: &mut Template, values: &Values, prefix: &str) -> Result<(), FillFailure> {
    for piece in template.pieces_mut() {
        let Piece::Expr(found) = piece else {
            continue;
        };

        let mut unknown = None;
        let mut visit = |expr: &mut Expr| resolve_path(expr, values, prefix, &mut unknown);
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

/// Rewrites a path into what it names, remembering the first variable `values` lacks.
///
/// One segment is a variable, inlined as its value. `stage.field` reads a stage, and
/// `call.output.field` reads a call's output. Stage names get `prefix` in front.
fn resolve_path(
    expr: &mut Expr,
    values: &Values,
    prefix: &str,
    unknown: &mut Option<String>,
) -> Result<(), String> {
    let Expr::Path(segments) = expr else {
        return Ok(());
    };

    let resolved = match segments.as_slice() {
        [name] => inline_variable(name, values, unknown),
        [stage, field] => {
            let stage = format!("{prefix}{stage}");
            let field = field.clone();
            Expr::Field { stage, field }
        },
        [call, output, field] => {
            let stage = format!("{prefix}{call}.{output}");
            let field = field.clone();
            Expr::Field { stage, field }
        },
        _ => {
            let written = segments.join(".");
            let message = format!(
                "`{written}` has too many parts, a stage field is written stage.field or call.output.field"
            );
            return Err(message);
        },
    };

    *expr = resolved;

    Ok(())
}

fn inline_variable(name: &str, values: &Values, unknown: &mut Option<String>) -> Expr {
    match values.get(name) {
        Some(Bound::Ready(value)) => {
            let literal = value.to_literal();
            Expr::Literal(literal)
        },
        Some(Bound::Deferred(expr)) => expr.clone(),
        Some(Bound::Unset) => {
            let variable = Some(name.to_string());
            Expr::Literal(Literal::None { variable })
        },
        None => {
            if unknown.is_none() {
                *unknown = Some(name.to_string());
            }

            Expr::Variable(name.to_string())
        },
    }
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
