#![expect(
    dead_code,
    reason = "the crate is built up task by task, later tasks add the callers"
)]

mod build;
mod diagnostic;
mod graph;
mod instantiate;
mod layout;
mod optimise;
mod render;
mod spec;
#[cfg(test)]
mod testing;

pub use self::build::{Built, build};
pub use self::diagnostic::{Diagnostic, DiagnosticCode, Diagnostics, Note, Severity};
pub use self::graph::{CompiledGraph, Edge, EdgeId, Node, NodeId, Sharing};
pub use self::instantiate::StageBox;
pub use self::layout::{Layout, Slot};
pub use self::optimise::{
    Change,
    CommonStageElimination,
    DeadStageElimination,
    FanOutAnalysis,
    OptimisationPass,
    PassReport,
    default_passes,
    optimise,
};
pub use self::render::{RenderOptions, render_mermaid};
pub use self::spec::{InputRef, Span, Spanned, StageSpec};
