use av_toolbox_core::{ArtefactSink, PipelineRegistry, StageKind};
use serde_json::Value;

use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, NodeId};
use crate::instantiate::describe;
use crate::{Diagnostic, DiagnosticCode, Diagnostics, Span};

/// An artefact sink, kept beside the graph rather than in it, with the artefacts it selected.
#[derive(Debug, Clone)]
pub struct ArtefactSinkNode {
    name: String,
    identifier: String,
    args: Value,
    span: Span,
    selected: Vec<(NodeId, &'static str)>,
}

impl ArtefactSinkNode {
    pub(crate) fn new(
        name: String,
        identifier: String,
        args: Value,
        span: Span,
        selected: Vec<(NodeId, &'static str)>,
    ) -> Self {
        Self {
            name,
            identifier,
            args,
            span,
            selected,
        }
    }

    /// Points every selection at its node's new id, dropping one whose node is gone and any repeat.
    pub(crate) fn remap<F: Fn(NodeId) -> Option<NodeId>>(&mut self, reindex: F) {
        let mut remapped: Vec<(NodeId, &'static str)> = Vec::with_capacity(self.selected.len());

        for (node, name) in &self.selected {
            let Some(moved) = reindex(*node) else {
                continue;
            };

            let pair = (moved, *name);
            if remapped.contains(&pair) {
                continue;
            }

            remapped.push(pair);
        }

        self.selected = remapped;
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The kind and name as one path, such as `sink.reports`.
    pub fn path(&self) -> String {
        stage_path(StageKind::Sink, &self.name)
    }

    pub fn identifier(&self) -> &str {
        &self.identifier
    }

    pub fn args(&self) -> &Value {
        &self.args
    }

    pub fn span(&self) -> Span {
        self.span
    }

    /// The node and artefact name of everything this sink takes, in node order and then declaration order.
    pub fn selected(&self) -> &[(NodeId, &'static str)] {
        &self.selected
    }
}

impl CompiledGraph {
    /// Builds every artefact sink, in the order the pipeline declared them.
    pub fn build_artefact_sinks(
        &self,
        registry: &PipelineRegistry,
    ) -> Result<Vec<Box<dyn ArtefactSink>>, Diagnostics> {
        let mut diagnostics = Diagnostics::new();
        let mut built = Vec::with_capacity(self.artefact_sinks().len());

        for sink in self.artefact_sinks() {
            let prepared = registry.prepare_artefact_sink(sink.identifier(), sink.args());
            let result = prepared.and_then(|prepared| prepared.build());

            match result {
                Ok(writer) => built.push(writer),
                Err(error) => diagnostics.push(Diagnostic::error(
                    DiagnosticCode::StageBuildFailed,
                    sink.path(),
                    sink.span(),
                    describe(&error),
                )),
            }
        }

        if diagnostics.has_errors() {
            return Err(diagnostics);
        }

        Ok(built)
    }
}
