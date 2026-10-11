mod reach;
mod run;

use av_toolbox_core::{AnyProbe, PipelineRegistry, PreparedStage};

pub use self::run::run_probes;
use crate::demands::bound_edge;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// One probe a stage made, as the build recorded it.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeProbe {
    pub input: &'static str,
    pub probe: AnyProbe,
    /// The probe as `PipelineRegistry::describe_probe` puts it, such as `13 samples`.
    pub description: String,
    /// What the probe found, as `PipelineRegistry::show_probed` puts it, once probes have run.
    pub found: Option<String>,
}

fn unknown_probe(node: &Node, probe: &AnyProbe) -> Diagnostic {
    let message = format!(
        "`{}` probes {}, which is no registered probe kind",
        node.name(),
        probe.kind().name()
    );

    Diagnostic::error(DiagnosticCode::UnknownProbe, node.path(), node.span(), message)
}

/// Records on each node the probes its stage makes of bound inputs.
pub fn record_probes(graph: &mut CompiledGraph, registry: &PipelineRegistry) -> Result<(), Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let recorded = recorded_probes(graph, registry, &prepared, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    for (id, probes) in recorded {
        graph.node_mut(id).set_probes(probes);
    }

    Ok(())
}

fn recorded_probes(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    prepared: &[Option<PreparedStage>],
    diagnostics: &mut Diagnostics,
) -> Vec<(NodeId, Vec<NodeProbe>)> {
    let mut recorded = Vec::new();

    for id in graph.order() {
        let Some(stage) = prepared[id.index()].as_ref() else {
            continue;
        };

        let node = graph.node(*id);
        let mut probes = Vec::new();
        for asked in stage.probes() {
            if bound_edge(node, asked.input).is_none() {
                continue;
            }

            let kind = asked.probe.kind();
            if registry.probe_kind(kind.name()).is_none() {
                let diagnostic = unknown_probe(node, &asked.probe);
                diagnostics.push(diagnostic);
                continue;
            }

            let description = registry.describe_probe(&asked.probe);
            let probe = NodeProbe {
                input: asked.input,
                probe: asked.probe,
                description,
                found: None,
            };
            probes.push(probe);
        }

        if !probes.is_empty() {
            recorded.push((*id, probes));
        }
    }

    recorded
}
