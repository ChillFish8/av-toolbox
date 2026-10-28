use std::ops::Range;

use super::expr::{BinaryOp, Expr, Function, Literal, UnaryOp};
use super::lexer::{Token, TokenKind, tokenize};
use super::{Piece, TemplateError, TemplateExpr};

const OPEN: &str = "{{";
const CLOSE: &str = "}}";

/// Splits `text` into plain text and parsed expressions.
pub(super) fn parse_pieces(text: &str) -> Result<Vec<Piece>, TemplateError> {
    let mut pieces = Vec::new();
    let mut cursor = 0;

    while let Some(found) = text[cursor..].find(OPEN) {
        let open = cursor + found;
        if open > cursor {
            let before = text[cursor..open].to_string();
            pieces.push(Piece::Text(before));
        }

        let inner_start = open + OPEN.len();
        let Some(close) = find_close(text, inner_start) else {
            return Err(TemplateError {
                message: "a template opened with {{ is never closed with }}".to_string(),
                range: open..text.len(),
                checked: None,
            });
        };

        let end = close + CLOSE.len();
        let range = open..end;
        let inner = &text[inner_start..close];
        let expr = parse_expression(inner, inner_start, range.clone())?;

        pieces.push(Piece::Expr(TemplateExpr { expr, range }));
        cursor = end;
    }

    if cursor < text.len() {
        let after = text[cursor..].to_string();
        pieces.push(Piece::Text(after));
    }

    Ok(pieces)
}

/// Finds the `}}` closing a template, skipping any inside a string.
fn find_close(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut in_string = false;
    let mut position = start;

    while position < bytes.len() {
        if bytes[position] == b'"' {
            in_string = !in_string;
        } else if !in_string && bytes[position..].starts_with(CLOSE.as_bytes()) {
            return Some(position);
        }

        position += 1;
    }

    None
}

/// Parses the expression between a template's braces.
///
/// `offset` is where `text` starts inside the template and `template` is the whole `{{ ... }}`.
fn parse_expression(text: &str, offset: usize, template: Range<usize>) -> Result<Expr, TemplateError> {
    let tokens = tokenize(text, offset)?;
    if tokens.is_empty() {
        return Err(TemplateError {
            message: "a template must hold an expression, such as {{ clip_input }}".to_string(),
            range: template,
            checked: None,
        });
    }

    let mut parser = Parser {
        tokens,
        index: 0,
        end: offset + text.len(),
    };
    let expr = parser.expression()?;

    if let Some(token) = parser.peek() {
        let found = describe(&token.kind);
        let message = format!("unexpected {found}");
        return Err(parser.error_at(message, token.range.clone()));
    }

    Ok(expr)
}

struct Parser {
    tokens: Vec<Token>,
    index: usize,
    /// Where the expression ends inside the template.
    end: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn peek_kind(&self) -> Option<&TokenKind> {
        let token = self.peek();
        token.map(|token| &token.kind)
    }

    fn next(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.index).cloned();
        self.index += 1;
        token
    }

    /// Consumes the next token when it is `kind`.
    fn eat(&mut self, kind: &TokenKind) -> bool {
        if self.peek_kind() != Some(kind) {
            return false;
        }

        self.index += 1;
        true
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Token, TemplateError> {
        let Some(token) = self.next() else {
            let wanted = describe(kind);
            let message = format!("expected {wanted}, but the template ends");
            return Err(self.error_at_end(message));
        };

        if &token.kind != kind {
            let wanted = describe(kind);
            let found = describe(&token.kind);
            let message = format!("expected {wanted}, found {found}");
            return Err(self.error_at(message, token.range));
        }

        Ok(token)
    }

    fn error_at(&self, message: String, range: Range<usize>) -> TemplateError {
        TemplateError {
            message,
            range,
            checked: None,
        }
    }

    fn error_at_end(&self, message: String) -> TemplateError {
        self.error_at(message, self.end..self.end)
    }

    fn expression(&mut self) -> Result<Expr, TemplateError> {
        if !self.eat(&TokenKind::If) {
            return self.or();
        }

        let condition = self.expression()?;
        self.expect(&TokenKind::Then)?;

        let then = self.expression()?;
        self.expect(&TokenKind::Else)?;

        let otherwise = self.expression()?;

        Ok(Expr::If {
            condition: Box::new(condition),
            then: Box::new(then),
            otherwise: Box::new(otherwise),
        })
    }

    fn or(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.and()?;

        while self.eat(&TokenKind::Or) {
            let right = self.and()?;
            left = binary(BinaryOp::Or, left, right);
        }

        Ok(left)
    }

    fn and(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.not()?;

        while self.eat(&TokenKind::And) {
            let right = self.not()?;
            left = binary(BinaryOp::And, left, right);
        }

        Ok(left)
    }

    fn not(&mut self) -> Result<Expr, TemplateError> {
        if !self.eat(&TokenKind::Not) {
            return self.comparison();
        }

        let operand = self.not()?;

        Ok(Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(operand),
        })
    }

    fn comparison(&mut self) -> Result<Expr, TemplateError> {
        let left = self.additive()?;

        let Some(op) = self.peek_kind().and_then(comparison_op) else {
            return Ok(left);
        };

        self.index += 1;
        let right = self.additive()?;

        if let Some(token) = self.peek()
            && comparison_op(&token.kind).is_some()
        {
            let message = "comparisons do not chain, join them with `and` instead".to_string();
            return Err(self.error_at(message, token.range.clone()));
        }

        Ok(binary(op, left, right))
    }

    fn additive(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.multiplicative()?;

        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Plus) => BinaryOp::Add,
                Some(TokenKind::Minus) => BinaryOp::Subtract,
                _ => return Ok(left),
            };

            self.index += 1;
            let right = self.multiplicative()?;
            left = binary(op, left, right);
        }
    }

    fn multiplicative(&mut self) -> Result<Expr, TemplateError> {
        let mut left = self.unary()?;

        loop {
            let op = match self.peek_kind() {
                Some(TokenKind::Star) => BinaryOp::Multiply,
                Some(TokenKind::Slash) => BinaryOp::Divide,
                Some(TokenKind::Percent) => BinaryOp::Remainder,
                _ => return Ok(left),
            };

            self.index += 1;
            let right = self.unary()?;
            left = binary(op, left, right);
        }
    }

    fn unary(&mut self) -> Result<Expr, TemplateError> {
        if !self.eat(&TokenKind::Minus) {
            return self.primary();
        }

        let operand = self.unary()?;

        Ok(Expr::Unary {
            op: UnaryOp::Negate,
            operand: Box::new(operand),
        })
    }

    fn primary(&mut self) -> Result<Expr, TemplateError> {
        let Some(token) = self.next() else {
            return Err(self.error_at_end("expected a value, but the template ends".to_string()));
        };

        let literal = match token.kind {
            TokenKind::Integer(number) => Literal::Integer(number),
            TokenKind::Float(number) => Literal::Float(number),
            TokenKind::String(text) => Literal::String(text),
            TokenKind::True => Literal::Bool(true),
            TokenKind::False => Literal::Bool(false),
            TokenKind::OpenParen => {
                let inner = self.expression()?;
                self.expect(&TokenKind::CloseParen)?;
                return Ok(inner);
            },
            TokenKind::Name(name) => return self.name(name, token.range),
            other => {
                let found = describe(&other);
                let message = format!("expected a value, found {found}");
                return Err(self.error_at(message, token.range));
            },
        };

        Ok(Expr::Literal(literal))
    }

    /// Reads a dotted name, or a call when the name is followed by `(`.
    fn name(&mut self, first: String, range: Range<usize>) -> Result<Expr, TemplateError> {
        let mut segments = vec![first];

        while self.eat(&TokenKind::Dot) {
            let Some(token) = self.next() else {
                return Err(self.error_at_end("expected a name after `.`".to_string()));
            };

            let TokenKind::Name(segment) = token.kind else {
                let found = describe(&token.kind);
                let message = format!("expected a name after `.`, found {found}");
                return Err(self.error_at(message, token.range));
            };

            segments.push(segment);
        }

        if self.peek_kind() != Some(&TokenKind::OpenParen) {
            return Ok(Expr::Path(segments));
        }

        let written = segments.join(".");
        let Some(function) = Function::named(&written) else {
            let known: Vec<&str> = Function::ALL.iter().map(|function| function.name()).collect();
            let listed = known.join(", ");
            let message = format!("unknown function `{written}`, expected one of {listed}");
            return Err(self.error_at(message, range));
        };

        self.call(function, range.start)
    }

    fn call(&mut self, function: Function, start: usize) -> Result<Expr, TemplateError> {
        self.expect(&TokenKind::OpenParen)?;

        let mut arguments = Vec::new();
        if !self.eat(&TokenKind::CloseParen) {
            loop {
                let argument = self.expression()?;
                arguments.push(argument);

                if self.eat(&TokenKind::Comma) {
                    continue;
                }

                self.expect(&TokenKind::CloseParen)?;
                break;
            }
        }

        let end = self.tokens[self.index - 1].range.end;
        let count = arguments.len();
        let arity = check_arity(function, count);
        arity.map_err(|message| self.error_at(message, start..end))?;

        Ok(Expr::Call { function, arguments })
    }
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn comparison_op(kind: &TokenKind) -> Option<BinaryOp> {
    match kind {
        TokenKind::EqualEqual => Some(BinaryOp::Equal),
        TokenKind::BangEqual => Some(BinaryOp::NotEqual),
        TokenKind::Less => Some(BinaryOp::Less),
        TokenKind::LessEqual => Some(BinaryOp::LessEqual),
        TokenKind::Greater => Some(BinaryOp::Greater),
        TokenKind::GreaterEqual => Some(BinaryOp::GreaterEqual),
        _ => None,
    }
}

/// `min` and `max` take two or more arguments, every other function exactly one.
fn check_arity(function: Function, count: usize) -> Result<(), String> {
    let name = function.name();
    match function {
        Function::Min | Function::Max if count < 2 => {
            Err(format!("{name} takes two or more arguments, found {count}"))
        },
        Function::Min | Function::Max => Ok(()),
        _ if count != 1 => Err(format!("{name} takes one argument, found {count}")),
        _ => Ok(()),
    }
}

fn describe(kind: &TokenKind) -> String {
    let symbol = match kind {
        TokenKind::Integer(number) => return format!("the number {number}"),
        TokenKind::Float(number) => return format!("the number {number}"),
        TokenKind::String(text) => return format!("the string {text:?}"),
        TokenKind::Name(name) => return format!("the name `{name}`"),
        TokenKind::Dot => ".",
        TokenKind::Comma => ",",
        TokenKind::OpenParen => "(",
        TokenKind::CloseParen => ")",
        TokenKind::Plus => "+",
        TokenKind::Minus => "-",
        TokenKind::Star => "*",
        TokenKind::Slash => "/",
        TokenKind::Percent => "%",
        TokenKind::EqualEqual => "==",
        TokenKind::BangEqual => "!=",
        TokenKind::Less => "<",
        TokenKind::LessEqual => "<=",
        TokenKind::Greater => ">",
        TokenKind::GreaterEqual => ">=",
        TokenKind::If => "if",
        TokenKind::Then => "then",
        TokenKind::Else => "else",
        TokenKind::And => "and",
        TokenKind::Or => "or",
        TokenKind::Not => "not",
        TokenKind::True => "true",
        TokenKind::False => "false",
    };

    format!("`{symbol}`")
}
