#![expect(
    dead_code,
    reason = "the crate is built up task by task, later tasks add the callers"
)]

mod build;
mod diagnostic;
mod graph;
mod spec;
#[cfg(test)]
mod testing;

pub use self::build::{Built, build};
pub use self::diagnostic::{Diagnostic, DiagnosticCode, Diagnostics, Note, Severity};
pub use self::graph::{CompiledGraph, Edge, EdgeId, Node, NodeId, Sharing};
pub use self::spec::{InputRef, Span, Spanned, StageSpec};
