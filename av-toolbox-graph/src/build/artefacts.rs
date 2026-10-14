use av_toolbox_core::{KeyFilter, PipelineRegistry};

use crate::artefacts::ArtefactSinkNode;
use crate::diagnostic::{args_path, input_path, stage_path};
use crate::graph::{Node, NodeId};
use crate::instantiate::describe;
use crate::{Diagnostic, DiagnosticCode, Diagnostics, StageSpec, suggest_name};

/// One artefact an input's node declares.
struct InputArtefact {
    node: NodeId,
    name: &'static str,
}

/// Checks every artefact sink's inputs, arguments and patterns, and works out what each selects.
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
    let mut sinks = Vec::with_capacity(indices.len());

    for index in indices {
        let spec = &specs[*index];
        let Some(selection) = prepare_selection(registry, spec, diagnostics) else {
            continue;
        };

        if earlier_errors {
            continue;
        }

        let Some(artefacts) = input_artefacts(spec, nodes, diagnostics) else {
            continue;
        };

        let Some(sink) = resolve_one(spec, &selection, &artefacts, diagnostics) else {
            continue;
        };

        sinks.push(sink);
    }

    sinks
}

/// Checks an artefact sink's inputs are positional and its arguments are accepted, answering with what it keeps.
fn prepare_selection(
    registry: &PipelineRegistry,
    spec: &StageSpec,
    diagnostics: &mut Diagnostics,
) -> Option<KeyFilter> {
    let inputs_accepted = check_input_shape(spec, diagnostics);

    let prepared = registry.prepare_artefact_sink(&spec.stage_type.value, &spec.args.value);
    let prepared = match prepared {
        Ok(prepared) => prepared,
        Err(error) => {
            diagnostics.push(Diagnostic::error(
                DiagnosticCode::InvalidArgs,
                args_path(spec.kind, &spec.name.value),
                spec.args.span,
                describe(&error),
            ));

            return None;
        },
    };

    if !inputs_accepted {
        return None;
    }

    Some(prepared.selection())
}

/// Reports an artefact sink with no inputs or with a named one, returning whether its inputs are usable.
fn check_input_shape(spec: &StageSpec, diagnostics: &mut Diagnostics) -> bool {
    let name = &spec.name.value;

    if spec.inputs.is_empty() {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::ArtefactSinkNoInputs,
            stage_path(spec.kind, name),
            spec.span,
            "an artefact sink must name the stages it writes artefacts from in its inputs",
        ));

        return false;
    }

    let mut accepted = true;

    for (index, input) in spec.inputs.iter().enumerate() {
        if input.value.name.is_none() {
            continue;
        }

        accepted = false;

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            input_path(spec.kind, name, index),
            input.span,
            "an artefact sink takes positional inputs only",
        ));
    }

    accepted
}

/// Every artefact the sink's inputs declare, in input order and then declaration order.
///
/// An input naming a stage already read adds nothing. An input whose stage declares no artefacts is reported.
fn input_artefacts(
    spec: &StageSpec,
    nodes: &[Node],
    diagnostics: &mut Diagnostics,
) -> Option<Vec<InputArtefact>> {
    let mut artefacts = Vec::new();
    let mut read: Vec<NodeId> = Vec::new();
    let mut accepted = true;

    for (index, input) in spec.inputs.iter().enumerate() {
        let target = input.value.target.as_str();
        let node = nodes.iter().find(|node| node.name() == target)?;
        let stage_artefacts = node.artefacts();
        let declared = stage_artefacts.declared();
        if declared.is_empty() {
            accepted = false;

            let message = format!("{} declares no artefacts to write", node.path());
            diagnostics.push(Diagnostic::error(
                DiagnosticCode::InputDeclaresNoArtefacts,
                input_path(spec.kind, &spec.name.value, index),
                input.span,
                message,
            ));

            continue;
        }

        if read.contains(&node.id()) {
            continue;
        }

        read.push(node.id());

        for decl in declared {
            artefacts.push(InputArtefact {
                node: node.id(),
                name: decl.name,
            });
        }
    }

    if !accepted {
        return None;
    }

    Some(artefacts)
}

/// Checks a sink's patterns against its inputs' artefacts, answering with the sink when it selects something.
fn resolve_one(
    spec: &StageSpec,
    selection: &KeyFilter,
    artefacts: &[InputArtefact],
    diagnostics: &mut Diagnostics,
) -> Option<ArtefactSinkNode> {
    let name = &spec.name.value;

    if !check_patterns(spec, selection, artefacts, diagnostics) {
        return None;
    }

    let selected: Vec<(NodeId, &'static str)> = artefacts
        .iter()
        .filter(|artefact| selection.keeps(artefact.name))
        .map(|artefact| (artefact.node, artefact.name))
        .collect();
    if selected.is_empty() {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::ArtefactSinkSelectsNothing,
            stage_path(spec.kind, name),
            spec.span,
            "this sink's include and exclude keep none of its inputs' artefacts, so it would write nothing",
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

/// Reports every pattern matching no artefact the inputs declare, returning whether all of them matched one.
fn check_patterns(
    spec: &StageSpec,
    selection: &KeyFilter,
    artefacts: &[InputArtefact],
    diagnostics: &mut Diagnostics,
) -> bool {
    let mut all_matched = true;

    for pattern in selection.patterns() {
        let matched = artefacts.iter().any(|artefact| pattern.matches(artefact.name));
        if matched {
            continue;
        }

        all_matched = false;

        let mut message = format!(
            "the pattern {} matches no artefact this sink's inputs declare",
            pattern.as_str()
        );
        let known_names = artefacts.iter().map(|artefact| artefact.name);
        if let Some(closest) = suggest_name(pattern.as_str(), known_names) {
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
