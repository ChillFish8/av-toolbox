//! Compiling stage definitions into a pipeline graph
//!
//! A caller hands [build] a list of [StageSpec] values and gets back a [CompiledGraph]: the same
//! stages resolved to each other, checked against the registry and ordered by dependency. Anything
//! wrong with the definitions comes back as a [Diagnostics] set, each entry carrying the span it
//! came from.
//!
//! A graph can then be taken four ways.
//!
//! - [solve()] chooses the sample format every edge carries.
//! - [CompiledGraph::instantiate] builds every stage through the registry, or
//!   [CompiledGraph::instantiate_region] builds just one region's, for a worker's own copies.
//! - [optimise()] runs passes over it, merging identical stages and working out which outputs have
//!   to be shared.
//! - [render_ascii] and [render_html] draw it, left to right, showing how a planned graph runs.
//!
//! ```no_run
//! # use av_toolbox_core::PipelineRegistry;
//! # use av_toolbox_graph::{RenderOptions, StageSpec, build, default_passes, optimise, render_ascii};
//! # fn example(registry: &PipelineRegistry, specs: &[StageSpec]) {
//! let built = build(registry, specs).expect("the definitions are valid");
//! let mut graph = built.into_graph();
//!
//! optimise(&mut graph, &default_passes());
//! println!("{}", render_ascii(&graph, &RenderOptions::default()));
//! # }
//! ```

mod build;
mod diagnostic;
mod graph;
mod instantiate;
mod layout;
mod optimise;
mod plan;
mod render;
mod residency;
mod solve;
mod spec;
#[cfg(test)]
mod testing;

pub use self::build::{Built, build};
pub use self::diagnostic::{Diagnostic, DiagnosticCode, Diagnostics, Note, Severity};
pub use self::graph::{CompiledGraph, Edge, EdgeId, Node, NodeId, NodeRole, Sharing};
pub use self::instantiate::StageBox;
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
pub use self::plan::{ExecutionPlan, Narrowed, NodePlan, Region, RegionId, Routing, plan};
pub use self::render::{RenderOptions, render_ascii, render_html};
pub use self::residency::place_transfers;
pub use self::solve::{SolvedFormats, solve};
pub use self::spec::{InputRef, Span, Spanned, StageSpec};
