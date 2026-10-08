use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// How far a depth first walk has got with a node.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Colour {
    Unvisited,
    OnStack,
    Done,
}

/// Reports the first cycle it finds, returning whether the graph has one.
///
/// Only the first cycle is reported. Every later one shares nodes with it, so reporting them all
/// would bury the one the user has to break.
pub(crate) fn detect_cycles(nodes: &[Node], diagnostics: &mut Diagnostics) -> bool {
    let mut colours = vec![Colour::Unvisited; nodes.len()];
    let mut stack = Vec::new();

    for index in 0..nodes.len() {
        let found = walk(nodes, NodeId::new(index), &mut colours, &mut stack, diagnostics);
        if found {
            return true;
        }
    }

    false
}

fn walk(
    nodes: &[Node],
    id: NodeId,
    colours: &mut [Colour],
    stack: &mut Vec<NodeId>,
    diagnostics: &mut Diagnostics,
) -> bool {
    match colours[id.index()] {
        Colour::Done => return false,
        Colour::OnStack => {
            report_cycle(nodes, id, stack, diagnostics);
            return true;
        },
        Colour::Unvisited => {},
    }

    colours[id.index()] = Colour::OnStack;
    stack.push(id);

    for edge in nodes[id.index()].inputs() {
        let found = walk(nodes, edge.source(), colours, stack, diagnostics);
        if found {
            return true;
        }
    }

    stack.pop();
    colours[id.index()] = Colour::Done;

    false
}

/// Renders the cycle as the path back round to the node the walk met again.
fn report_cycle(nodes: &[Node], repeated: NodeId, stack: &[NodeId], diagnostics: &mut Diagnostics) {
    let start = stack
        .iter()
        .position(|id| *id == repeated)
        .expect("the repeated node is on the stack");

    let cycle = &stack[start..];
    let mut names: Vec<&str> = cycle.iter().map(|id| nodes[id.index()].name()).collect();
    names.push(nodes[repeated.index()].name());

    let node = &nodes[repeated.index()];
    let message = format!("the pipeline has a cycle: {}", names.join(" -> "));
    let mut diagnostic = Diagnostic::error(
        DiagnosticCode::Cycle,
        stage_path(node.kind(), node.name()),
        node.span(),
        message,
    );

    for id in cycle {
        let member = &nodes[id.index()];
        for edge in member.inputs() {
            if cycle.contains(&edge.source()) {
                let note = format!("{} reads {}", member.name(), nodes[edge.source().index()].name());

                diagnostic = diagnostic.with_note(note, edge.span());
            }
        }
    }

    diagnostics.push(diagnostic);
}

/// Warns about every stage no sink transitively depends on, frame or artefact.
pub(crate) fn report_unused(graph: &CompiledGraph, diagnostics: &mut Diagnostics) {
    let mut reachable = vec![false; graph.len()];
    let producers = graph.artefact_producers();
    let mut pending: Vec<NodeId> = graph.sinks().collect();
    pending.extend(producers);

    while let Some(id) = pending.pop() {
        if reachable[id.index()] {
            continue;
        }

        reachable[id.index()] = true;
        for edge in graph.node(id).inputs() {
            pending.push(edge.source());
        }
    }

    for node in graph.nodes() {
        if reachable[node.id().index()] {
            continue;
        }

        diagnostics.push(Diagnostic::warning(
            DiagnosticCode::UnusedStage,
            stage_path(node.kind(), node.name()),
            node.span(),
            "no sink depends on this stage",
        ));
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::{StageArtefacts, StageKind};
    use serde_json::json;

    use super::*;
    use crate::graph::{Edge, NodeParts};
    use crate::{ArgSpans, DiagnosticCode, Severity, Span};

    fn node(index: usize, name: &str, kind: StageKind, inputs: &[usize]) -> Node {
        let edges = inputs
            .iter()
            .enumerate()
            .map(|(slot, source)| Edge::new(None, None, slot, NodeId::new(*source), Span::unknown()))
            .collect();

        Node::new(NodeParts {
            id: NodeId::new(index),
            name: name.to_string(),
            kind,
            identifier: "test/example".to_string(),
            args: json!({}),
            args_span: Span::unknown(),
            arg_spans: ArgSpans::default(),
            inputs: edges,
            workers: None,
            span: Span::unknown(),
            artefacts: StageArtefacts::NONE,
        })
    }

    #[test]
    fn an_acyclic_graph_passes() {
        let nodes = vec![
            node(0, "source", StageKind::Source, &[]),
            node(1, "sink", StageKind::Sink, &[0]),
        ];
        let mut diagnostics = Diagnostics::new();

        assert!(!detect_cycles(&nodes, &mut diagnostics));
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn a_cycle_is_reported_once_and_names_the_path() {
        let nodes = vec![
            node(0, "first", StageKind::Filter, &[2]),
            node(1, "second", StageKind::Filter, &[0]),
            node(2, "third", StageKind::Filter, &[1]),
        ];
        let mut diagnostics = Diagnostics::new();

        assert!(detect_cycles(&nodes, &mut diagnostics));

        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0].code(), DiagnosticCode::Cycle);
        assert!(reported[0].message().contains("first"));
        assert!(reported[0].message().contains("->"));
    }

    #[test]
    fn a_stage_no_sink_depends_on_is_a_warning() {
        let nodes = vec![
            node(0, "source", StageKind::Source, &[]),
            node(1, "unused", StageKind::Filter, &[0]),
            node(2, "sink", StageKind::Sink, &[0]),
        ];
        let graph = CompiledGraph::new(nodes);
        let mut diagnostics = Diagnostics::new();

        report_unused(&graph, &mut diagnostics);

        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();
        assert_eq!(reported.len(), 1);
        assert_eq!(reported[0].code(), DiagnosticCode::UnusedStage);
        assert_eq!(reported[0].path(), "filter.unused");
        assert_eq!(reported[0].severity(), Severity::Warning);
    }
}
