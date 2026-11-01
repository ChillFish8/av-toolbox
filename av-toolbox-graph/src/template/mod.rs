mod eval;
mod expr;
mod fields;
mod lexer;
mod parse;
#[cfg(test)]
mod tests;

use std::ops::Range;

use serde_json::Value;

pub use self::eval::{EvalError, Scope};
pub use self::expr::{BinaryOp, Expr, Function, Literal, LiteralKind, StageRead, UnaryOp, VariableRules};
pub use self::fields::field_of;
use self::parse::parse_pieces;
use crate::Span;

/// A string split into plain text and the expressions between its braces.
#[derive(Debug, Clone, PartialEq)]
pub struct Template {
    pieces: Vec<Piece>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Piece {
    Text(String),
    Expr(TemplateExpr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TemplateExpr {
    pub expr: Expr,
    /// Byte range inside the template text.
    pub range: Range<usize>,
}

/// Why a template could not be parsed or filled.
///
/// `variable` holds the span of the call variable a failure came from. `checked` holds the span
/// of that variable's rules when a filled value broke them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateError {
    pub message: String,
    /// Byte range inside the template text.
    pub range: Range<usize>,
    pub checked: Option<Span>,
    pub variable: Option<Span>,
}

impl Template {
    pub fn parse(text: &str) -> Result<Template, TemplateError> {
        let pieces = parse_pieces(text)?;
        Ok(Template { pieces })
    }

    pub fn has_template(text: &str) -> bool {
        text.contains("{{")
    }

    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    pub fn pieces_mut(&mut self) -> &mut [Piece] {
        &mut self.pieces
    }

    /// Every stage a `Field` names, in the order written.
    pub fn stages(&self) -> Vec<&str> {
        let reads = self.stage_reads();

        reads.into_iter().map(|read| read.stage).collect()
    }

    /// Every stage a `Field` names with the call variable it was written in, in the order written.
    pub fn stage_reads(&self) -> Vec<StageRead<'_>> {
        let mut reads = Vec::new();

        for piece in &self.pieces {
            if let Piece::Expr(found) = piece {
                found.expr.collect_stage_reads(None, &mut reads);
            }
        }

        reads
    }

    /// Whether any expression in the template calls `function`.
    pub fn calls(&self, function: Function) -> bool {
        self.pieces.iter().any(|piece| match piece {
            Piece::Expr(found) => found.expr.calls(function),
            Piece::Text(_) => false,
        })
    }

    /// Fills the template. A template that is one expression and nothing else keeps its type.
    pub fn fill(&self, scope: &dyn Scope) -> Result<Value, TemplateError> {
        if let [Piece::Expr(only)] = self.pieces.as_slice() {
            let value = evaluate(only, scope)?;
            return Ok(value.to_json());
        }

        let mut filled = String::new();
        for piece in &self.pieces {
            match piece {
                Piece::Text(plain) => filled.push_str(plain),
                Piece::Expr(found) => {
                    let value = evaluate(found, scope)?;
                    if let Literal::None { variable } = &value {
                        return Err(none_in_text(found, variable.as_deref()));
                    }

                    let rendered = value.to_text();

                    filled.push_str(&rendered);
                },
            }
        }

        Ok(Value::String(filled))
    }
}

fn evaluate(found: &TemplateExpr, scope: &dyn Scope) -> Result<Literal, TemplateError> {
    let evaluated = found.expr.evaluate(scope);
    match evaluated {
        Ok(value) => Ok(value),
        Err(error) => Err(located(error, &found.range)),
    }
}

fn none_in_text(found: &TemplateExpr, variable: Option<&str>) -> TemplateError {
    let source = match variable {
        Some(name) => format!("variable `{name}`"),
        None => "the expression".to_string(),
    };
    let message = format!("{source} is none, which cannot be part of a longer string");

    TemplateError {
        message,
        range: found.range.clone(),
        checked: None,
        variable: None,
    }
}

fn located(error: EvalError, range: &Range<usize>) -> TemplateError {
    TemplateError {
        message: error.message,
        range: range.clone(),
        checked: error.checked,
        variable: error.variable,
    }
}
