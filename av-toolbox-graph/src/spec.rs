use av_toolbox_core::StageKind;
use serde_json::Value;

/// A byte range in whatever the caller parsed the definitions from.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// A span for a caller with no source behind it, such as a builder UI.
    pub const fn unknown() -> Self {
        Self { start: 0, end: 0 }
    }

    pub const fn is_unknown(&self) -> bool {
        self.start == 0 && self.end == 0
    }
}

/// A value together with where it was written.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Spanned<T> {
    pub value: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }

    pub fn unknown(value: T) -> Self {
        Self {
            value,
            span: Span::unknown(),
        }
    }
}

/// A reference from one stage to another stage's output.
#[derive(Debug, Clone)]
pub struct InputRef {
    /// The input name, or `None` when the input was given positionally.
    pub name: Option<String>,
    /// The name of the stage being read.
    pub target: String,
}

/// One stage as the caller's definition file wrote it.
#[derive(Debug, Clone)]
pub struct StageSpec {
    pub name: Spanned<String>,
    pub kind: StageKind,
    pub stage_type: Spanned<String>,
    pub inputs: Vec<Spanned<InputRef>>,
    pub args: Spanned<Value>,
    pub span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_span_reports_itself_as_unknown() {
        assert!(Span::unknown().is_unknown());
        assert!(!Span::new(4, 9).is_unknown());
    }

    #[test]
    fn a_spanned_value_keeps_its_span() {
        let name = Spanned::new(String::from("denoise"), Span::new(4, 9));

        assert_eq!(name.value, "denoise");
        assert_eq!(name.span, Span::new(4, 9));
    }
}
