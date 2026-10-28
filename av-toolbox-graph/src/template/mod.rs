mod eval;
mod expr;
mod lexer;
mod parse;
#[cfg(test)]
mod tests;

use std::ops::Range;

use serde_json::Value;

pub use self::eval::{EvalError, Scope};
pub use self::expr::{BinaryOp, Expr, Function, Literal, LiteralKind, UnaryOp, VariableRules};
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
/// `checked` holds the span of a variable's rules when a filled value broke them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateError {
    pub message: String,
    /// Byte range inside the template text.
    pub range: Range<usize>,
    pub checked: Option<Span>,
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
        let mut stages = Vec::new();

        for piece in &self.pieces {
            if let Piece::Expr(found) = piece {
                found.expr.collect_stages(&mut stages);
            }
        }

        stages
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
    evaluated.map_err(|error| TemplateError {
        message: error.message,
        range: found.range.clone(),
        checked: error.checked,
    })
}
