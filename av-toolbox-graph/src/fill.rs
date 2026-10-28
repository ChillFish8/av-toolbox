use std::collections::{HashMap, HashSet};

use av_toolbox_core::PipelineRegistry;
use av_toolbox_gpu::{Capabilities, StreamFormat};
use serde_json::Value;

use crate::diagnostic::args_path;
use crate::graph::{CompiledGraph, Node, NodeId, Reference};
use crate::natural::{lineage, natural_formats};
use crate::template::{Literal, Scope, TemplateError, field_of};
use crate::{DeferredArg, Diagnostic, DiagnosticCode, Diagnostics};

/// Fills every deferred node in dependency order, returning the nodes filled.
///
/// Each template reads the natural formats of the stages it names. Every node is attempted, but a
/// node whose references read an unfilled node is skipped quietly, since that node's own failure
/// is already reported.
pub fn fill_references(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
) -> Result<Vec<NodeId>, Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let mut filled = Vec::new();
    let mut unfilled: HashSet<NodeId> = HashSet::new();

    let order = graph.order().to_vec();
    for id in order {
        let node = graph.node(id);
        if !node.is_deferred() {
            continue;
        }

        if reads_unfilled(graph, node, &unfilled) {
            unfilled.insert(id);
            continue;
        }

        match filled_args(graph, registry, capabilities, node) {
            Ok(args) => {
                graph.fill_args(id, args);
                filled.push(id);
            },
            Err(errors) => {
                diagnostics.extend(errors);
                unfilled.insert(id);
            },
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    Ok(filled)
}

fn reads_unfilled(graph: &CompiledGraph, node: &Node, unfilled: &HashSet<NodeId>) -> bool {
    let sources: Vec<NodeId> = node
        .references()
        .iter()
        .map(|reference| reference.source)
        .collect();
    let needed = lineage(graph, &sources);

    needed.iter().any(|id| unfilled.contains(id))
}

/// A copy of `node`'s args with every deferred arg filled in.
fn filled_args(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
    node: &Node,
) -> Result<Value, Diagnostics> {
    let formats = referenced_formats(graph, registry, capabilities, node)?;
    let scope = NaturalScope { formats };

    let mut args = node.args().clone();
    let mut diagnostics = Diagnostics::new();

    for deferred in node.deferred() {
        let filled = deferred.template.fill(&scope);
        match filled {
            Ok(value) => write_at(&mut args, &deferred.pointer, value),
            Err(error) => {
                let diagnostic = fill_failed(node, deferred, error);
                diagnostics.push(diagnostic);
            },
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    Ok(args)
}

/// The natural format of every stage `node` references, keyed by stage name.
///
/// Each reference is worked out on its own, so a failure carries a note at the reference that
/// needed it.
fn referenced_formats<'a>(
    graph: &'a CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
    node: &Node,
) -> Result<HashMap<&'a str, StreamFormat>, Diagnostics> {
    let mut formats = HashMap::new();
    let mut diagnostics = Diagnostics::new();

    for reference in node.references() {
        let worked_out = natural_formats(graph, registry, capabilities, &[reference.source]);

        match worked_out {
            Ok(found) => {
                let name = graph.node(reference.source).name();
                formats.insert(name, found[&reference.source]);
            },
            Err(errors) => {
                let noted = note_reference(errors, node, reference);
                diagnostics.extend(noted);
            },
        }
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    Ok(formats)
}

fn note_reference(errors: Diagnostics, node: &Node, reference: &Reference) -> Diagnostics {
    let message = format!("needed for the format {} reads", node.name());
    let noted: Vec<Diagnostic> = errors
        .into_inner()
        .into_iter()
        .map(|diagnostic| diagnostic.with_note(message.clone(), reference.span))
        .collect();

    Diagnostics::from(noted)
}

fn write_at(args: &mut Value, pointer: &str, value: Value) {
    let slot = args
        .pointer_mut(pointer)
        .expect("a deferred arg points at a value in its own args");

    *slot = value;
}

/// Places a template failure at the deferred arg, or at the call's variable when its rules broke.
fn fill_failed(node: &Node, deferred: &DeferredArg, error: TemplateError) -> Diagnostic {
    let field = deferred.pointer.replace('/', ".");
    let args = args_path(node.kind(), node.name());
    let path = format!("{args}{field}");

    let Some(rules_span) = error.checked else {
        return Diagnostic::error(
            DiagnosticCode::InvalidExpression,
            path,
            deferred.span,
            error.message,
        );
    };

    Diagnostic::error(DiagnosticCode::InvalidVariable, path, rules_span, error.message)
}

/// Answers fields from the natural formats of the stages a node references.
struct NaturalScope<'a> {
    formats: HashMap<&'a str, StreamFormat>,
}

impl Scope for NaturalScope<'_> {
    fn variable(&self, _name: &str) -> Option<Literal> {
        None
    }

    fn field(&self, stage: &str, field: &str) -> Result<Literal, String> {
        let Some(format) = self.formats.get(stage) else {
            let message = format!("{stage} is not a stage these args reference");
            return Err(message);
        };

        match field_of(format, field) {
            Ok(value) => Ok(value),
            Err(reason) => {
                let message = format!("{stage} {reason}");
                Err(message)
            },
        }
    }
}
