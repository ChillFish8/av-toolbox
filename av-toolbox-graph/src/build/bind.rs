use std::collections::HashMap;

use av_toolbox_core::{BindInputsError, InputArgument, InputSignature, StageHandle};

use super::resolve::{Resolution, ResolvedStage};
use crate::diagnostic::{input_path, stage_path};
use crate::graph::{Edge, Node, NodeId, NodeParts};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, StageSpec};

/// Binds every resolved stage against its signature and turns it into a node.
pub(crate) fn bind(
    specs: &[StageSpec],
    resolution: &Resolution<'_>,
    diagnostics: &mut Diagnostics,
) -> Vec<Node> {
    let mut accepted: Vec<bool> = resolution.stages.iter().map(|stage| stage.is_some()).collect();

    check_signatures(specs, resolution, &mut accepted, diagnostics);
    drop_broken_consumers(resolution, &mut accepted);

    let ids = assign_ids(&accepted);
    build_nodes(resolution, &accepted, &ids)
}

/// Rejects any stage whose inputs do not match its signature.
fn check_signatures(
    specs: &[StageSpec],
    resolution: &Resolution<'_>,
    accepted: &mut [bool],
    diagnostics: &mut Diagnostics,
) {
    for (index, stage) in resolution.stages.iter().enumerate() {
        let Some(stage) = stage else {
            continue;
        };

        let arguments = placeholder_arguments(stage.spec, specs, &resolution.by_name);
        let bound = av_toolbox_core::bind_inputs(stage.info.inputs(), &arguments);
        let Err(error) = bound else {
            continue;
        };

        diagnostics.push(bind_diagnostic(stage.spec, error));
        accepted[index] = false;
    }
}

/// Builds the argument list binding needs, with handles that binding never looks at.
///
/// The handles are placeholders because ids are not assigned until rejections have settled, but
/// each producer's stream type is already known from its kind.
fn placeholder_arguments(
    spec: &StageSpec,
    specs: &[StageSpec],
    by_name: &HashMap<String, usize>,
) -> Vec<InputArgument> {
    let placeholder = StageHandle::new(0);

    spec.inputs
        .iter()
        .map(|input| {
            let target = by_name[&input.value.target];
            let stream = specs[target]
                .kind
                .output()
                .expect("resolution rejects a stage that reads a sink");

            match &input.value.name {
                Some(name) => InputArgument::named(name.clone(), placeholder, stream),
                None => InputArgument::positional(placeholder, stream),
            }
        })
        .collect()
}

/// Puts a binding failure on the input that caused it, or on the stage when it has no position.
fn bind_diagnostic(spec: &StageSpec, error: BindInputsError) -> Diagnostic {
    let position = match &error {
        BindInputsError::PositionalAfterNamed { position } => Some(*position),
        BindInputsError::DuplicateInput { position, .. } => Some(*position),
        BindInputsError::TooManyInputs { position, .. } => Some(*position),
        BindInputsError::UnknownInput { position, .. } => Some(*position),
        BindInputsError::InputTypeMismatch { position, .. } => Some(*position),
        BindInputsError::MissingRequiredInput { .. } => None,
    };

    let message = describe(spec, &error, position);
    let Some(position) = position else {
        return Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            stage_path(spec.kind, &spec.name.value),
            spec.span,
            message,
        );
    };

    let span = spec.inputs[position].span;
    Diagnostic::error(
        DiagnosticCode::InvalidInputs,
        input_path(spec.kind, &spec.name.value, position),
        span,
        message,
    )
}

/// Names the producing stage when a type did not match, which the core error cannot do.
fn describe(spec: &StageSpec, error: &BindInputsError, position: Option<usize>) -> String {
    let BindInputsError::InputTypeMismatch {
        name,
        expected,
        found,
        ..
    } = error
    else {
        return error.to_string();
    };

    let index = position.expect("a type mismatch always carries a position");
    let producer = &spec.inputs[index].value.target;

    let Some(name) = name else {
        return format!("the variadic tail takes {expected}, but {producer} produces {found}");
    };

    format!("the input {name} takes {expected}, but {producer} produces {found}")
}

/// Drops any stage that reads a stage which was itself rejected, until nothing changes.
fn drop_broken_consumers(resolution: &Resolution<'_>, accepted: &mut [bool]) {
    loop {
        let mut changed = false;

        for (index, stage) in resolution.stages.iter().enumerate() {
            if !accepted[index] {
                continue;
            }

            let Some(stage) = stage else {
                continue;
            };

            let reads_a_rejected_stage = stage.spec.inputs.iter().any(|input| {
                let target = resolution.by_name.get(&input.value.target).copied();
                target.is_none_or(|position| !accepted[position])
            });

            if reads_a_rejected_stage {
                accepted[index] = false;
                changed = true;
            }
        }

        if !changed {
            return;
        }
    }
}

/// Numbers the surviving stages in declaration order.
fn assign_ids(accepted: &[bool]) -> HashMap<usize, NodeId> {
    let mut ids = HashMap::new();

    for (index, is_accepted) in accepted.iter().enumerate() {
        if !is_accepted {
            continue;
        }

        ids.insert(index, NodeId::new(ids.len()));
    }

    ids
}

fn build_nodes(resolution: &Resolution<'_>, accepted: &[bool], ids: &HashMap<usize, NodeId>) -> Vec<Node> {
    let mut nodes = Vec::with_capacity(ids.len());

    for (index, stage) in resolution.stages.iter().enumerate() {
        if !accepted[index] {
            continue;
        }

        let Some(stage) = stage else {
            continue;
        };

        let id = ids[&index];
        let inputs = build_edges(stage, resolution, ids);

        nodes.push(Node::new(NodeParts {
            id,
            name: stage.spec.name.value.clone(),
            kind: stage.spec.kind,
            identifier: stage.info.full_identifier().to_string(),
            args: stage.spec.args.value.clone(),
            args_span: stage.spec.args.span,
            inputs,
            workers: stage.spec.workers,
            span: stage.spec.span,
        }));
    }

    nodes
}

fn build_edges(
    stage: &ResolvedStage<'_>,
    resolution: &Resolution<'_>,
    ids: &HashMap<usize, NodeId>,
) -> Vec<Edge> {
    let signature = stage.info.inputs();
    let mut next_positional = 0;
    let mut next_variadic = 0;
    let mut edges = Vec::with_capacity(stage.spec.inputs.len());

    for input in &stage.spec.inputs {
        let position = resolution.by_name[&input.value.target];
        let source = ids[&position];
        let (param, slot) = slot_for(
            signature,
            &input.value.name,
            &mut next_positional,
            &mut next_variadic,
        );

        edges.push(Edge::new(
            input.value.name.clone(),
            param,
            slot,
            source,
            input.span,
        ));
    }

    edges
}

/// Works out which signature param an input filled and the `Inputs` slot it lines up with.
///
/// Mirrors the rules `av_toolbox_core::bind_inputs` binds by, so the slot returned here always
/// agrees with the space [InputSignature::chunk_authority] and the chunk authority it resolves
/// are in. A positional input takes the next unfilled param, a named one takes the param of that
/// name, and anything left over goes to the variadic tail and has no param. A param's slot is
/// its index in the signature, and a variadic input's slot follows every param's, in the order
/// it was bound.
fn slot_for(
    signature: &'static InputSignature,
    name: &Option<String>,
    next_positional: &mut usize,
    next_variadic: &mut usize,
) -> (Option<&'static str>, usize) {
    let params = signature.params();

    let matched = match name {
        Some(name) => params.iter().position(|param| param.name() == name),
        None => {
            let position = *next_positional;
            *next_positional += 1;
            (position < params.len()).then_some(position)
        },
    };

    let Some(index) = matched else {
        let slot = params.len() + *next_variadic;
        *next_variadic += 1;
        return (None, slot);
    };

    (Some(params[index].name()), index)
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;
    use crate::build::resolve::resolve;
    use crate::{DiagnosticCode, StageSpec, testing};

    /// Resolves then binds, returning the nodes and every diagnostic raised.
    fn run(specs: &[StageSpec]) -> (Vec<Node>, Vec<DiagnosticCode>) {
        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();

        let resolution = resolve(&registry, specs, &mut diagnostics);
        let nodes = bind(specs, &resolution, &mut diagnostics);
        let codes = diagnostics.iter().map(|entry| entry.code()).collect();

        (nodes, codes)
    }

    fn source_and_sink() -> Vec<StageSpec> {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        vec![source, sink]
    }

    #[test]
    fn a_positional_input_becomes_an_edge_labelled_with_its_param() {
        let (nodes, codes) = run(&source_and_sink());

        assert!(codes.is_empty());
        assert_eq!(nodes.len(), 2);

        let edge = &nodes[1].inputs()[0];
        assert_eq!(edge.source(), NodeId::new(0));
        assert_eq!(edge.param(), Some("input"));
        assert_eq!(edge.name(), None);
    }

    #[test]
    fn a_named_input_keeps_the_name_the_definition_gave_it() {
        let mut specs = source_and_sink();
        specs[1].inputs = vec![testing::named("input", "input")];

        let (nodes, codes) = run(&specs);

        assert!(codes.is_empty());
        assert_eq!(nodes[1].inputs()[0].name(), Some("input"));
        assert_eq!(nodes[1].inputs()[0].param(), Some("input"));
    }

    #[test]
    fn an_input_landing_in_the_variadic_tail_has_no_param() {
        let source = testing::spec(StageKind::Source, "input", "test/file");
        let other = testing::spec(StageKind::Source, "other", "test/file");

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("input"), testing::positional("other")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("merge")];

        let (nodes, codes) = run(&[source, other, merge, sink]);

        assert!(codes.is_empty());
        assert_eq!(nodes[2].inputs()[0].param(), Some("base"));
        assert_eq!(nodes[2].inputs()[1].param(), None);
    }

    #[test]
    fn a_missing_required_input_is_reported_on_the_stage() {
        let mut specs = source_and_sink();
        specs[1].inputs = Vec::new();

        let (nodes, codes) = run(&specs);

        assert_eq!(codes, vec![DiagnosticCode::InvalidInputs]);
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn too_many_inputs_are_reported_on_the_extra_one() {
        let mut specs = source_and_sink();
        specs[1].inputs = vec![testing::positional("input"), testing::positional("input")];

        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        let resolution = resolve(&registry, &specs, &mut diagnostics);
        bind(&specs, &resolution, &mut diagnostics);

        let invalid = diagnostics
            .iter()
            .find(|entry| entry.code() == DiagnosticCode::InvalidInputs)
            .expect("should report invalid inputs");

        assert_eq!(invalid.path(), "sink.out.inputs[1]");
    }

    #[test]
    fn a_stage_reading_a_rejected_stage_is_dropped_without_a_second_diagnostic() {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut broken = testing::spec(StageKind::Filter, "broken", "test/blur");
        broken.inputs = vec![testing::positional("missing")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("broken")];

        let (nodes, codes) = run(&[source, broken, sink]);

        assert_eq!(codes, vec![DiagnosticCode::UnknownStageReference]);
        assert_eq!(nodes.len(), 1);
    }

    /// A source feeding deband's frames input, and a mask feeding its overlay input.
    fn well_typed_deband() -> Vec<StageSpec> {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut merge = testing::spec(StageKind::Mask, "selection", "test/merge");
        merge.inputs = vec![testing::positional("input")];

        let mut deband = testing::spec(StageKind::Filter, "deband", "test/deband");
        deband.inputs = vec![testing::positional("input"), testing::positional("selection")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("deband")];

        vec![source, merge, deband, sink]
    }

    #[test]
    fn a_correctly_typed_pipeline_binds() {
        let (nodes, codes) = run(&well_typed_deband());

        assert!(codes.is_empty());
        assert_eq!(nodes.len(), 4);
    }

    #[test]
    fn a_mask_input_given_frames_names_the_param_and_the_producer() {
        let mut specs = well_typed_deband();
        specs[2].inputs = vec![testing::positional("input"), testing::positional("other")];
        specs.push(testing::spec(StageKind::Source, "other", "test/file"));

        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        let resolution = resolve(&registry, &specs, &mut diagnostics);
        bind(&specs, &resolution, &mut diagnostics);

        let reported = diagnostics
            .iter()
            .find(|entry| entry.code() == DiagnosticCode::InvalidInputs)
            .expect("should report invalid inputs");

        assert_eq!(reported.path(), "filter.deband.inputs[1]");
        assert!(reported.message().contains("overlay"));
        assert!(reported.message().contains("other"));
        assert!(reported.message().contains("mask"));
        assert!(reported.message().contains("frames"));
    }

    #[test]
    fn a_variadic_tail_given_the_wrong_type_is_named_as_the_tail() {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut selection = testing::spec(StageKind::Mask, "selection", "test/merge");
        selection.inputs = vec![testing::positional("input")];

        let mut composite = testing::spec(StageKind::Mask, "composite", "test/merge");
        composite.inputs = vec![testing::positional("input"), testing::positional("selection")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("composite")];

        let specs = vec![source, selection, composite, sink];

        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        let resolution = resolve(&registry, &specs, &mut diagnostics);
        bind(&specs, &resolution, &mut diagnostics);

        let reported = diagnostics
            .iter()
            .find(|entry| entry.code() == DiagnosticCode::InvalidInputs)
            .expect("should report invalid inputs");

        assert_eq!(reported.path(), "mask.composite.inputs[1]");
        assert!(reported.message().contains("the variadic tail"));
        assert!(reported.message().contains("selection"));
        assert!(reported.message().contains("frames"));
        assert!(reported.message().contains("mask"));
    }
}
