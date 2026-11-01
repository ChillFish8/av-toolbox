use std::cmp::Ordering;

use super::expr::{BinaryOp, Expr, Function, Literal, LiteralKind, UnaryOp, VariableRules};
use crate::Span;

/// The smallest magnitude too large for an `i64`.
const INTEGER_LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// Where an expression reads its variables and stage fields from.
pub trait Scope {
    fn variable(&self, name: &str) -> Option<Literal>;

    /// The field of a stage's output, or a message saying why it cannot be read.
    fn field(&self, stage: &str, field: &str) -> Result<Literal, String>;

    /// The text of the file at `path`, or a message saying why it cannot be read.
    fn read_file(&self, _path: &str) -> Result<String, String> {
        Err("`read` cannot be used here".to_string())
    }
}

/// Why an expression could not be evaluated.
///
/// `variable` holds the span of the call variable a failure came from. `checked` holds the span
/// of that variable's rules when the value broke them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalError {
    pub message: String,
    pub checked: Option<Span>,
    pub variable: Option<Span>,
}

impl Expr {
    pub fn evaluate(&self, scope: &dyn Scope) -> Result<Literal, EvalError> {
        match self {
            Self::Literal(literal) => Ok(literal.clone()),
            Self::Path(segments) => {
                let written = segments.join(".");
                let message = format!("`{written}` names neither a variable nor a stage field");
                Err(failure(message))
            },
            Self::Variable(name) => {
                let Some(found) = scope.variable(name) else {
                    let message = format!("unknown variable `{name}`");
                    return Err(failure(message));
                };

                Ok(found)
            },
            Self::Field { stage, field } => scope.field(stage, field).map_err(failure),
            Self::Unary { op, operand } => {
                let value = operand.evaluate(scope)?;
                unary(*op, value)
            },
            Self::Binary { op, left, right } => binary(*op, left, right, scope),
            Self::IsNone { operand, negated } => {
                let value = operand.evaluate(scope)?;
                let is_none = matches!(value, Literal::None { .. });
                Ok(Literal::Bool(is_none != *negated))
            },
            Self::If {
                condition,
                then,
                otherwise,
            } => {
                let chosen = condition.evaluate(scope)?;
                if expect_bool(chosen, "an if condition")? {
                    then.evaluate(scope)
                } else {
                    otherwise.evaluate(scope)
                }
            },
            Self::Call { function, arguments } => {
                let values = arguments
                    .iter()
                    .map(|argument| argument.evaluate(scope))
                    .collect::<Result<Vec<_>, _>>()?;
                call(*function, values, scope)
            },
            Self::Checked { inner, rules } => {
                let evaluated = inner.evaluate(scope);
                let value = match evaluated {
                    Ok(value) => value,
                    Err(error) => return Err(within_variable(error, rules)),
                };

                let checked = check_rules(rules, value);
                match checked {
                    Ok(value) => Ok(value),
                    Err(message) => Err(broke_rules(message, rules)),
                }
            },
        }
    }
}

fn failure(message: String) -> EvalError {
    EvalError {
        message,
        checked: None,
        variable: None,
    }
}

/// Marks a failure as coming from the call variable `rules` belongs to, unless a variable nested
/// inside it already claimed it.
fn within_variable(error: EvalError, rules: &VariableRules) -> EvalError {
    if error.variable.is_some() {
        return error;
    }

    EvalError {
        variable: Some(rules.span),
        ..error
    }
}

fn broke_rules(message: String, rules: &VariableRules) -> EvalError {
    EvalError {
        message,
        checked: Some(rules.span),
        variable: Some(rules.span),
    }
}

/// The kind with its article, such as "an integer".
fn described(kind: LiteralKind) -> String {
    match kind {
        LiteralKind::Integer => "an integer".to_string(),
        other => format!("a {other}"),
    }
}

fn expect_bool(value: Literal, context: &str) -> Result<bool, EvalError> {
    match value {
        Literal::Bool(flag) => Ok(flag),
        none @ Literal::None { .. } => Err(none_misused(&none, context)),
        other => {
            let found = described(other.kind());
            let message = format!("{context} must be a bool, found {found}");
            Err(failure(message))
        },
    }
}

/// The failure for a `none` used where a value is needed, naming its variable when it has one.
fn none_misused(value: &Literal, context: &str) -> EvalError {
    let message = match value {
        Literal::None { variable: Some(name) } => {
            format!("variable `{name}` is none, so it cannot be used in {context}")
        },
        _ => format!("none cannot be used in {context}"),
    };

    failure(message)
}

fn as_float(value: &Literal) -> Option<f64> {
    match value {
        Literal::Integer(number) => Some(*number as f64),
        Literal::Float(number) => Some(*number),
        _ => None,
    }
}

/// A float result, refused when it is not finite.
fn finite(number: f64, op: &str) -> Result<Literal, EvalError> {
    if number.is_finite() {
        return Ok(Literal::Float(number));
    }

    let message = format!("`{op}` gave {number}, which is not a finite number");
    Err(failure(message))
}

fn overflow(op: &str) -> EvalError {
    let message = format!("`{op}` overflowed a 64-bit integer");
    failure(message)
}

fn mismatch(op: BinaryOp, left: &Literal, right: &Literal) -> EvalError {
    let symbol = op.symbol();
    let context = format!("`{symbol}`");

    for value in [left, right] {
        if matches!(value, Literal::None { .. }) {
            return none_misused(value, &context);
        }
    }

    let left_kind = described(left.kind());
    let right_kind = described(right.kind());

    let message = format!("cannot apply `{symbol}` to {left_kind} and {right_kind}");
    failure(message)
}

fn unary(op: UnaryOp, value: Literal) -> Result<Literal, EvalError> {
    match (op, value) {
        (UnaryOp::Negate, Literal::Integer(number)) => {
            let negated = number.checked_neg();
            negated.map(Literal::Integer).ok_or_else(|| overflow("-"))
        },
        (UnaryOp::Negate, Literal::Float(number)) => Ok(Literal::Float(-number)),
        (UnaryOp::Negate, none @ Literal::None { .. }) => Err(none_misused(&none, "a negation")),
        (UnaryOp::Negate, other) => {
            let found = described(other.kind());
            let message = format!("cannot negate {found}");
            Err(failure(message))
        },
        (UnaryOp::Not, other) => {
            let flag = expect_bool(other, "the operand of `not`")?;
            Ok(Literal::Bool(!flag))
        },
    }
}

fn binary(op: BinaryOp, left: &Expr, right: &Expr, scope: &dyn Scope) -> Result<Literal, EvalError> {
    let left_value = left.evaluate(scope)?;

    if matches!(op, BinaryOp::And | BinaryOp::Or) {
        let context = format!("each side of `{}`", op.symbol());
        let left_flag = expect_bool(left_value, &context)?;

        let decided = match op {
            BinaryOp::And => !left_flag,
            _ => left_flag,
        };
        if decided {
            return Ok(Literal::Bool(left_flag));
        }

        let right_value = right.evaluate(scope)?;
        let right_flag = expect_bool(right_value, &context)?;
        return Ok(Literal::Bool(right_flag));
    }

    let right_value = right.evaluate(scope)?;

    let left_is_none = matches!(left_value, Literal::None { .. });
    let right_is_none = matches!(right_value, Literal::None { .. });
    let has_none = left_is_none || right_is_none;
    match op {
        BinaryOp::Equal if has_none => Ok(Literal::Bool(false)),
        BinaryOp::NotEqual if has_none => Ok(Literal::Bool(true)),
        BinaryOp::Equal => equal(op, &left_value, &right_value).map(Literal::Bool),
        BinaryOp::NotEqual => {
            let same = equal(op, &left_value, &right_value)?;
            Ok(Literal::Bool(!same))
        },
        BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
            let ordering = compare(op, &left_value, &right_value)?;
            let holds = match op {
                BinaryOp::Less => ordering == Ordering::Less,
                BinaryOp::LessEqual => ordering != Ordering::Greater,
                BinaryOp::Greater => ordering == Ordering::Greater,
                _ => ordering != Ordering::Less,
            };
            Ok(Literal::Bool(holds))
        },
        _ => arithmetic(op, &left_value, &right_value),
    }
}

fn equal(op: BinaryOp, left: &Literal, right: &Literal) -> Result<bool, EvalError> {
    match (left, right) {
        (Literal::Integer(first), Literal::Integer(second)) => Ok(first == second),
        (Literal::String(first), Literal::String(second)) => Ok(first == second),
        (Literal::Bool(first), Literal::Bool(second)) => Ok(first == second),
        _ => {
            let (Some(first), Some(second)) = (as_float(left), as_float(right)) else {
                return Err(mismatch(op, left, right));
            };
            Ok(first == second)
        },
    }
}

fn compare(op: BinaryOp, left: &Literal, right: &Literal) -> Result<Ordering, EvalError> {
    match (left, right) {
        (Literal::Integer(first), Literal::Integer(second)) => Ok(first.cmp(second)),
        (Literal::String(first), Literal::String(second)) => Ok(first.cmp(second)),
        _ => {
            let (Some(first), Some(second)) = (as_float(left), as_float(right)) else {
                return Err(mismatch(op, left, right));
            };

            let ordering = first.partial_cmp(&second);
            ordering.ok_or_else(|| mismatch(op, left, right))
        },
    }
}

/// Integers stay integers, except `/` which always gives a float. Mixing in a float gives a float.
fn arithmetic(op: BinaryOp, left: &Literal, right: &Literal) -> Result<Literal, EvalError> {
    let symbol = op.symbol();

    if let (Literal::Integer(first), Literal::Integer(second)) = (left, right)
        && op != BinaryOp::Divide
    {
        if op == BinaryOp::Remainder && *second == 0 {
            let message = format!("`{symbol}` divides by zero");
            return Err(failure(message));
        }

        let result = match op {
            BinaryOp::Add => first.checked_add(*second),
            BinaryOp::Subtract => first.checked_sub(*second),
            BinaryOp::Multiply => first.checked_mul(*second),
            _ => first.checked_rem(*second),
        };
        return result.map(Literal::Integer).ok_or_else(|| overflow(symbol));
    }

    let (Some(first), Some(second)) = (as_float(left), as_float(right)) else {
        return Err(mismatch(op, left, right));
    };

    if matches!(op, BinaryOp::Divide | BinaryOp::Remainder) && second == 0.0 {
        let message = format!("`{symbol}` divides by zero");
        return Err(failure(message));
    }

    let result = match op {
        BinaryOp::Add => first + second,
        BinaryOp::Subtract => first - second,
        BinaryOp::Multiply => first * second,
        BinaryOp::Divide => first / second,
        _ => first % second,
    };
    finite(result, symbol)
}

fn call(function: Function, values: Vec<Literal>, scope: &dyn Scope) -> Result<Literal, EvalError> {
    if function == Function::Read {
        return read(&values[0], scope);
    }

    numeric_call(function, &values)
}

fn numeric_call(function: Function, values: &[Literal]) -> Result<Literal, EvalError> {
    let name = function.name();

    for value in values {
        if matches!(value, Literal::None { .. }) {
            let context = format!("`{name}`");
            return Err(none_misused(value, &context));
        }

        if as_float(value).is_none() {
            let found = described(value.kind());
            let message = format!("{name} takes numbers, found {found}");
            return Err(failure(message));
        }
    }

    match function {
        Function::Min | Function::Max => extreme(function, values),
        Function::Round | Function::Floor | Function::Ceil => round(function, &values[0]),
        Function::Read => unreachable!("read is handled before the numeric functions"),
        Function::Abs => match &values[0] {
            Literal::Integer(number) => {
                let absolute = number.checked_abs();
                absolute.map(Literal::Integer).ok_or_else(|| overflow(name))
            },
            other => {
                let number = as_float(other).unwrap_or_default();
                Ok(Literal::Float(number.abs()))
            },
        },
    }
}

fn read(path: &Literal, scope: &dyn Scope) -> Result<Literal, EvalError> {
    let Literal::String(text) = path else {
        if matches!(path, Literal::None { .. }) {
            return Err(none_misused(path, "`read`"));
        }

        let found = described(path.kind());
        let message = format!("read takes a string, found {found}");
        return Err(failure(message));
    };

    let contents = scope.read_file(text).map_err(failure)?;

    Ok(Literal::String(contents))
}

/// The smallest or largest of `values`, an integer when every value is one.
fn extreme(function: Function, values: &[Literal]) -> Result<Literal, EvalError> {
    let integers: Option<Vec<i64>> = values
        .iter()
        .map(|value| match value {
            Literal::Integer(number) => Some(*number),
            _ => None,
        })
        .collect();

    if let Some(integers) = integers {
        let picked = match function {
            Function::Min => integers.into_iter().min(),
            _ => integers.into_iter().max(),
        };
        let Some(picked) = picked else {
            let name = function.name();
            let message = format!("{name} needs at least one argument");
            return Err(failure(message));
        };

        return Ok(Literal::Integer(picked));
    }

    let floats = values.iter().filter_map(as_float);
    let picked = match function {
        Function::Min => floats.fold(f64::INFINITY, f64::min),
        _ => floats.fold(f64::NEG_INFINITY, f64::max),
    };
    finite(picked, function.name())
}

/// Rounds a number to an integer, refusing one outside the `i64` range.
fn round(function: Function, value: &Literal) -> Result<Literal, EvalError> {
    let number = match value {
        Literal::Integer(number) => return Ok(Literal::Integer(*number)),
        other => as_float(other).unwrap_or_default(),
    };

    let rounded = match function {
        Function::Floor => number.floor(),
        Function::Ceil => number.ceil(),
        _ => number.round(),
    };

    if !(-INTEGER_LIMIT..INTEGER_LIMIT).contains(&rounded) {
        let name = function.name();
        let message = format!("{name}({number}) does not fit in a 64-bit integer");
        return Err(failure(message));
    }

    Ok(Literal::Integer(rounded as i64))
}

/// Checks `value` against a variable's type, bounds and choices.
///
/// An integer is accepted where a float is expected, and becomes a float. A `none` given to an
/// optional variable takes on that variable's name.
fn check_rules(rules: &VariableRules, value: Literal) -> Result<Literal, String> {
    let name = &rules.name;
    if matches!(value, Literal::None { .. }) {
        if rules.optional {
            let variable = Some(name.clone());
            return Ok(Literal::None { variable });
        }

        return Err(format!("variable `{name}` is not optional, so it cannot be none"));
    }

    let value = match (rules.expected, value) {
        (LiteralKind::Float, Literal::Integer(number)) => Literal::Float(number as f64),
        (expected, value) if value.kind() == expected => value,
        (expected, value) => {
            let wanted = described(expected);
            let found = described(value.kind());
            let shown = value.to_text();
            return Err(format!(
                "variable `{name}` expects {wanted}, found {found} {shown}"
            ));
        },
    };

    let number = as_float(&value);
    let shown = value.to_text();

    if let (Some(number), Some(min)) = (number, rules.min)
        && number < min
    {
        return Err(format!(
            "variable `{name}`: {shown} is below the minimum of {min}"
        ));
    }

    if let (Some(number), Some(max)) = (number, rules.max)
        && number > max
    {
        return Err(format!(
            "variable `{name}`: {shown} is above the maximum of {max}"
        ));
    }

    let Some(choices) = &rules.choices else {
        return Ok(value);
    };

    if choices.contains(&value) {
        return Ok(value);
    }

    let listed: Vec<String> = choices.iter().map(Literal::to_text).collect();
    let joined = listed.join(", ");

    Err(format!("variable `{name}`: {shown} is not one of {joined}"))
}
