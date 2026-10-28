use std::collections::{HashMap, HashSet};

use av_toolbox_core::{PipelineRegistry, StageError};
use av_toolbox_gpu::{Capabilities, StreamFormat};

use crate::diagnostic::stage_path;
use crate::fill::note_templates;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::solve::{signature_slots, slot_inputs};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// The natural format of each node in `wanted`, working out their ancestors on the way.
///
/// A node's natural format is what it produces when every input carries the natural format of
/// the node feeding it, settled on its most preferred sample. Nodes are worked out sources first,
/// and every failure is reported at the node it happened on.
pub(crate) fn natural_formats(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
    wanted: &[NodeId],
) -> Result<HashMap<NodeId, StreamFormat>, Diagnostics> {
    let needed = lineage(graph, wanted);
    let mut diagnostics = Diagnostics::new();
    let mut formats: HashMap<NodeId, StreamFormat> = HashMap::new();

    let in_order = graph.order().iter().filter(|id| needed.contains(id));
    for id in in_order {
        let node = graph.node(*id);
        let natural = natural_format(graph, registry, capabilities, node, &formats, &mut diagnostics);

        if let Some(format) = natural {
            formats.insert(*id, format);
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let wanted_formats = wanted.iter().map(|id| (*id, formats[id])).collect();

    Ok(wanted_formats)
}

/// Every node in `wanted` and every node feeding them, directly or not.
pub(crate) fn lineage(graph: &CompiledGraph, wanted: &[NodeId]) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut pending: Vec<NodeId> = wanted.to_vec();

    while let Some(id) = pending.pop() {
        if !seen.insert(id) {
            continue;
        }

        let sources = graph.node(id).inputs().iter().map(|edge| edge.source());
        pending.extend(sources);
    }

    seen
}

/// Works out one node's natural format from its inputs' natural formats.
///
/// A node reading one that has no natural format has none either, without a diagnostic of its
/// own, since the failure further up was already reported.
fn natural_format(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
    node: &Node,
    formats: &HashMap<NodeId, StreamFormat>,
    diagnostics: &mut Diagnostics,
) -> Option<StreamFormat> {
    if let Some(diagnostic) = unknowable(node) {
        diagnostics.push(diagnostic);
        return None;
    }

    let slots = signature_slots(registry, node);
    let inputs = slot_inputs(&slots, |edge| {
        let source = node.inputs()[edge].source();
        formats.get(&source).copied()
    })?;

    let stage = graph.prepared_node(registry, node, diagnostics)?;
    let produced = stage.produces(capabilities, &inputs);
    let output = match produced {
        Ok(output) => output,
        Err(error) => {
            let diagnostic = build_failed(node, &error);
            diagnostics.push(diagnostic);
            return None;
        },
    };

    let Some(preferred) = output.samples.first() else {
        let error = StageError::msg("the stage offered no sample format to emit");
        let diagnostic = build_failed(node, &error);
        diagnostics.push(diagnostic);
        return None;
    };

    let natural = output.template.with_sample(preferred);

    Some(natural)
}

/// Why `node` cannot have a natural format yet, if it cannot.
fn unknowable(node: &Node) -> Option<Diagnostic> {
    if node.kind().output().is_none() {
        let message = format!("`{}` is a sink, so it has no format to read", node.name());
        let diagnostic = Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        );

        return Some(diagnostic);
    }

    if node.is_deferred() {
        let message = format!(
            "`{}` reads other stages in its own args, so its format is not known yet",
            node.name(),
        );
        let diagnostic = Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        );

        return Some(diagnostic);
    }

    let unprobed = !node.probes().is_empty() && node.probed().is_none();
    if unprobed {
        let message = format!(
            "`{}` depends on probes that have not run, so its format is not known yet",
            node.name(),
        );
        let diagnostic = Diagnostic::error(
            DiagnosticCode::ProbeFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        );

        return Some(diagnostic);
    }

    None
}

/// A build failure at `node`, noting each template that filled it.
fn build_failed(node: &Node, error: &StageError) -> Diagnostic {
    let diagnostic = Diagnostic::error(
        DiagnosticCode::StageBuildFailed,
        stage_path(node.kind(), node.name()),
        node.span(),
        error.to_string(),
    );

    note_templates(node, diagnostic)
}
