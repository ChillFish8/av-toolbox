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

/// Where each value under a stage's arguments was written, keyed by JSON pointer.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct ArgSpans {
    entries: Vec<(String, Span)>,
}

impl ArgSpans {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, pointer: impl Into<String>, span: Span) {
        self.entries.push((pointer.into(), span));
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The span of the deepest recorded value holding `pointer`.
    ///
    /// `/limits/0` is held by `/limits`, but `/limits` is not held by `/limit`.
    pub fn narrowest(&self, pointer: &str) -> Option<Span> {
        let mut best: Option<(usize, Span)> = None;

        for (recorded, span) in &self.entries {
            if !holds(recorded, pointer) {
                continue;
            }

            let deeper = best.is_none_or(|(length, _)| recorded.len() > length);
            if deeper {
                best = Some((recorded.len(), *span));
            }
        }

        best.map(|(_, span)| span)
    }
}

fn holds(recorded: &str, pointer: &str) -> bool {
    let Some(rest) = pointer.strip_prefix(recorded) else {
        return false;
    };

    rest.is_empty() || rest.starts_with('/')
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
    /// Where each value under `args` was written, empty when nothing recorded them.
    pub arg_spans: ArgSpans,
    /// How many instances of this stage to run, or `None` when the pipeline did not say.
    pub workers: Option<usize>,
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

    #[test]
    fn the_deepest_recorded_value_wins() {
        let mut spans = ArgSpans::new();
        spans.insert("/limits", Span::new(10, 20));
        spans.insert("/limits/0", Span::new(12, 14));

        assert_eq!(spans.narrowest("/limits/0"), Some(Span::new(12, 14)));
        assert_eq!(spans.narrowest("/limits/1"), Some(Span::new(10, 20)));
    }

    #[test]
    fn a_key_sharing_a_prefix_does_not_contain_another() {
        let mut spans = ArgSpans::new();
        spans.insert("/limit", Span::new(10, 20));

        assert_eq!(spans.narrowest("/limits"), None);
    }

    #[test]
    fn no_recorded_values_narrow_nothing() {
        let spans = ArgSpans::default();

        assert!(spans.is_empty());
        assert_eq!(spans.narrowest("/sigma"), None);
    }
}
