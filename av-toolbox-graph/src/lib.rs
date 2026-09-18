mod diagnostic;
mod spec;

pub use self::diagnostic::{Diagnostic, DiagnosticCode, Diagnostics, Note, Severity};
pub use self::spec::{InputRef, Span, Spanned, StageSpec};
