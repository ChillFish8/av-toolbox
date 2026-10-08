use av_toolbox_core::{KeyFilter, PipelineRegistry};

use crate::artefacts::ArtefactSinkNode;
use crate::diagnostic::{args_path, input_path, stage_path};
use crate::graph::{Node, NodeId};
use crate::instantiate::describe;
use crate::{Diagnostic, DiagnosticCode, Diagnostics, StageSpec, suggest_name};

/// One artefact a node declares, under its key.
struct ArtefactKey {
    key: String,
    node: NodeId,
    name: &'static str,
}

/// Checks every artefact sink's arguments and patterns, and works out what each selects.
///
/// With `earlier_errors` set, a rejected stage's artefacts may be missing from `nodes`. Only each
/// sink's inputs and arguments are checked then, and no sink is returned.
pub(crate) fn resolve_artefact_sinks(
    registry: &PipelineRegistry,
    specs: &[StageSpec],
    indices: &[usize],
    nodes: &[Node],
    earlier_errors: bool,
    diagnostics: &mut Diagnostics,
) -> Vec<ArtefactSinkNode> {
    let keys = artefact_keys(nodes);
    let mut sinks = Vec::with_capacity(indices.len());

    for index in indices {
        let spec = &specs[*index];
        let Some(selection) = prepare_selection(registry, spec, diagnostics) else {
            continue;
        };

        if earlier_errors {
            continue;
        }

        let Some(sink) = resolve_one(spec, &selection, &keys, diagnostics) else {
            continue;
        };

        sinks.push(sink);
    }

    sinks
}

/// Every artefact every node declares, in node order and then declaration order.
fn artefact_keys(nodes: &[Node]) -> Vec<ArtefactKey> {
    let mut keys = Vec::new();

    for node in nodes {
        let path = node.path();
        let artefacts = node.artefacts();

        for decl in artefacts.declared() {
            keys.push(ArtefactKey {
                key: format!("{path}.{}", decl.name),
                node: node.id(),
                name: decl.name,
            });
        }
    }

    keys
}

/// Checks an artefact sink takes no inputs and accepts its arguments, answering with what it keeps.
fn prepare_selection(
    registry: &PipelineRegistry,
    spec: &StageSpec,
    diagnostics: &mut Diagnostics,
) -> Option<KeyFilter> {
    let name = &spec.name.value;

    if let Some(input) = spec.inputs.first() {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::ArtefactSinkHasInputs,
            input_path(spec.kind, name, 0),
            input.span,
            "an artefact sink writes the artefacts stages hand over, so it takes no inputs",
        ));

        return None;
    }

    let prepared = registry.prepare_artefact_sink(&spec.stage_type.value, &spec.args.value);
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            diagnostics.push(Diagnostic::error(
                DiagnosticCode::InvalidArgs,
                args_path(spec.kind, name),
                spec.args.span,
                describe(&error),
            ));

            return None;
        },
    };

    Some(prepared.selection())
}

/// Checks a sink's patterns against `keys`, answering with the sink when it selects something.
fn resolve_one(
    spec: &StageSpec,
    selection: &KeyFilter,
    keys: &[ArtefactKey],
    diagnostics: &mut Diagnostics,
) -> Option<ArtefactSinkNode> {
    let name = &spec.name.value;

    if !check_patterns(spec, selection, keys, diagnostics) {
        return None;
    }

    let selected: Vec<(NodeId, &'static str)> = keys
        .iter()
        .filter(|entry| selection.keeps(&entry.key))
        .map(|entry| (entry.node, entry.name))
        .collect();
    if selected.is_empty() {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::ArtefactSinkSelectsNothing,
            stage_path(spec.kind, name),
            spec.span,
            "this sink's include and exclude keep no artefact, so it would write nothing",
        ));

        return None;
    }

    let sink = ArtefactSinkNode::new(
        name.clone(),
        spec.stage_type.value.clone(),
        spec.args.value.clone(),
        spec.span,
        selected,
    );

    Some(sink)
}

/// Reports every pattern matching no declared artefact, returning whether all of them matched one.
fn check_patterns(
    spec: &StageSpec,
    selection: &KeyFilter,
    keys: &[ArtefactKey],
    diagnostics: &mut Diagnostics,
) -> bool {
    let mut all_matched = true;

    for pattern in selection.patterns() {
        let matched = keys.iter().any(|entry| pattern.matches(&entry.key));
        if matched {
            continue;
        }

        all_matched = false;

        let mut message = format!(
            "the pattern {} matches no artefact any stage declares",
            pattern.as_str()
        );
        let known_keys = keys.iter().map(|entry| entry.key.as_str());
        if let Some(closest) = suggest_name(pattern.as_str(), known_keys) {
            let hint = format!(", did you mean {closest}");
            message.push_str(&hint);
        }

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::UnmatchedArtefactPattern,
            args_path(spec.kind, &spec.name.value),
            spec.args.span,
            message,
        ));
    }

    all_matched
}
