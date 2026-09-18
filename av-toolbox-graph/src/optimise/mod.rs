mod dead_code;
mod dedup;
mod fan_out;

pub use self::dead_code::DeadStageElimination;
pub use self::dedup::CommonStageElimination;
pub use self::fan_out::FanOutAnalysis;
use crate::graph::{CompiledGraph, Sharing};

/// Something a pass did to the graph.
///
/// Stages are named rather than referenced by id, because dropping a node renumbers every node
/// after it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Change {
    DroppedStage {
        name: String,
    },
    MergedStages {
        kept: String,
        removed: String,
    },
    EdgeSharing {
        consumer: String,
        producer: String,
        sharing: Sharing,
    },
}

/// What one pass did.
#[derive(Debug, Clone)]
pub struct PassReport {
    pass: &'static str,
    changes: Vec<Change>,
}

impl PassReport {
    pub fn new(pass: &'static str) -> Self {
        Self {
            pass,
            changes: Vec::new(),
        }
    }

    pub fn pass(&self) -> &'static str {
        self.pass
    }

    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    pub(crate) fn push(&mut self, change: Change) {
        self.changes.push(change);
    }
}

/// A rewrite or an analysis run over a built graph.
pub trait OptimisationPass {
    fn name(&self) -> &'static str;

    fn run(&self, graph: &mut CompiledGraph) -> PassReport;
}

/// Runs each pass in turn, returning what each one did.
pub fn optimise(graph: &mut CompiledGraph, passes: &[Box<dyn OptimisationPass>]) -> Vec<PassReport> {
    passes.iter().map(|pass| pass.run(graph)).collect()
}

/// The passes to run when the caller has no opinion.
///
/// Dead stages go first so nothing else spends work on them, merging follows, and the fan out
/// analysis runs last because merging changes how many consumers a stage has.
pub fn default_passes() -> Vec<Box<dyn OptimisationPass>> {
    vec![
        Box::new(DeadStageElimination),
        Box::new(CommonStageElimination),
        Box::new(FanOutAnalysis),
    ]
}
