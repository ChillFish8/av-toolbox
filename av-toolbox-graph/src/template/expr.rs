use std::fmt;

use serde_json::{Number, Value};

use crate::Span;

/// A value an expression reads or produces.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Bool(bool),
    /// The absence of a value. `variable` names the optional variable it came from, if any.
    None {
        variable: Option<String>,
    },
}

impl Literal {
    pub fn kind(&self) -> LiteralKind {
        match self {
            Self::Integer(_) => LiteralKind::Integer,
            Self::Float(_) => LiteralKind::Float,
            Self::String(_) => LiteralKind::String,
            Self::Bool(_) => LiteralKind::Bool,
            Self::None { .. } => LiteralKind::None,
        }
    }

    /// The value as JSON, with a float that is not finite becoming null.
    pub fn to_json(&self) -> Value {
        match self {
            Self::Integer(number) => Value::from(*number),
            Self::Float(number) => {
                let converted = Number::from_f64(*number);
                converted.map(Value::Number).unwrap_or(Value::Null)
            },
            Self::String(text) => Value::String(text.clone()),
            Self::Bool(flag) => Value::Bool(*flag),
            Self::None { .. } => Value::Null,
        }
    }

    /// The value as it reads inside a longer string.
    pub fn to_text(&self) -> String {
        match self {
            Self::Integer(number) => number.to_string(),
            Self::Float(number) => number.to_string(),
            Self::String(text) => text.clone(),
            Self::Bool(flag) => flag.to_string(),
            Self::None { .. } => "none".to_string(),
        }
    }
}

/// The type of a [Literal].
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum LiteralKind {
    Integer,
    Float,
    String,
    Bool,
    None,
}

impl LiteralKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::Float => "float",
            Self::String => "string",
            Self::Bool => "bool",
            Self::None => "none",
        }
    }
}

impl fmt::Display for LiteralKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// One expression inside a template.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    /// A dotted name as written, such as `["input"]` or `["denoise", "clean", "width"]`.
    Path(Vec<String>),
    Variable(String),
    Field {
        stage: String,
        field: String,
    },
    Unary {
        op: UnaryOp,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// `operand is none`, or `operand is not none` when `negated`.
    IsNone {
        operand: Box<Expr>,
        negated: bool,
    },
    If {
        condition: Box<Expr>,
        then: Box<Expr>,
        otherwise: Box<Expr>,
    },
    Call {
        function: Function,
        arguments: Vec<Expr>,
    },
    Checked {
        inner: Box<Expr>,
        rules: VariableRules,
    },
}

impl Expr {
    /// Calls `visit` on every sub-expression, children first, so a caller can rewrite paths.
    pub fn rewrite(&mut self, visit: &mut dyn FnMut(&mut Expr) -> Result<(), String>) -> Result<(), String> {
        match self {
            Self::Literal(_) | Self::Path(_) | Self::Variable(_) | Self::Field { .. } => {},
            Self::Unary { operand, .. } | Self::IsNone { operand, .. } => operand.rewrite(visit)?,
            Self::Binary { left, right, .. } => {
                left.rewrite(visit)?;
                right.rewrite(visit)?;
            },
            Self::If {
                condition,
                then,
                otherwise,
            } => {
                condition.rewrite(visit)?;
                then.rewrite(visit)?;
                otherwise.rewrite(visit)?;
            },
            Self::Call { arguments, .. } => {
                for argument in arguments {
                    argument.rewrite(visit)?;
                }
            },
            Self::Checked { inner, .. } => inner.rewrite(visit)?,
        }

        visit(self)
    }

    /// Pushes a [StageRead] for every [Expr::Field] below this one, in the order written.
    ///
    /// `variable` is the span of the innermost call variable around this expression, if any.
    pub(crate) fn collect_stage_reads<'a>(&'a self, variable: Option<Span>, reads: &mut Vec<StageRead<'a>>) {
        match self {
            Self::Literal(_) | Self::Path(_) | Self::Variable(_) => {},
            Self::Field { stage, .. } => reads.push(StageRead { stage, variable }),
            Self::Unary { operand, .. } | Self::IsNone { operand, .. } => {
                operand.collect_stage_reads(variable, reads);
            },
            Self::Binary { left, right, .. } => {
                left.collect_stage_reads(variable, reads);
                right.collect_stage_reads(variable, reads);
            },
            Self::If {
                condition,
                then,
                otherwise,
            } => {
                condition.collect_stage_reads(variable, reads);
                then.collect_stage_reads(variable, reads);
                otherwise.collect_stage_reads(variable, reads);
            },
            Self::Call { arguments, .. } => {
                for argument in arguments {
                    argument.collect_stage_reads(variable, reads);
                }
            },
            Self::Checked { inner, rules } => inner.collect_stage_reads(Some(rules.span), reads),
        }
    }
}

/// A stage an expression reads, with the span of the call variable it was written in, if any.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct StageRead<'a> {
    pub stage: &'a str,
    pub variable: Option<Span>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
}

impl BinaryOp {
    pub fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::And => "and",
            Self::Or => "or",
        }
    }
}

/// A function a template may call.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Function {
    Min,
    Max,
    Round,
    Floor,
    Ceil,
    Abs,
}

impl Function {
    pub const ALL: [Function; 6] = [
        Self::Min,
        Self::Max,
        Self::Round,
        Self::Floor,
        Self::Ceil,
        Self::Abs,
    ];

    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|function| function.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
            Self::Round => "round",
            Self::Floor => "floor",
            Self::Ceil => "ceil",
            Self::Abs => "abs",
        }
    }
}

/// The rules of a called file's variable, kept so a value filled late is still checked.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableRules {
    pub name: String,
    pub expected: LiteralKind,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub choices: Option<Vec<Literal>>,
    /// Whether the variable may be `none`.
    pub optional: bool,
    pub span: Span,
}
