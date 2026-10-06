use std::collections::HashMap;

use av_toolbox_core::{PipelineRegistry, StageInfo, StageKind};

use crate::diagnostic::{input_path, stage_path, type_path};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, Spanned, StageSpec, suggest_name};

/// A spec that passed every structural check, paired with its registry entry.
pub(crate) struct ResolvedStage<'a> {
    pub(crate) spec: &'a StageSpec,
    pub(crate) info: StageInfo<'a>,
}

/// The outcome of resolving a set of specs.
pub(crate) struct Resolution<'a> {
    /// One entry per spec, in the order they were given, `None` where the spec was rejected.
    pub(crate) stages: Vec<Option<ResolvedStage<'a>>>,
    /// Every name a spec declared, mapped to the index of its first definition.
    pub(crate) by_name: HashMap<String, usize>,
}

/// Checks names, types and the rules about how kinds may connect.
pub(crate) fn resolve<'a>(
    registry: &'a PipelineRegistry,
    specs: &'a [StageSpec],
    diagnostics: &mut Diagnostics,
) -> Resolution<'a> {
    let by_name = collect_names(specs, diagnostics);
    let registered = index_registry(registry);

    let mut stages: Vec<Option<ResolvedStage<'a>>> = Vec::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let is_first_definition = by_name.get(&spec.name.value) == Some(&index);
        if !is_first_definition {
            stages.push(None);
            continue;
        }

        let Some(info) = look_up(&registered, spec, diagnostics) else {
            stages.push(None);
            continue;
        };

        if !check_inputs(spec, specs, &by_name, diagnostics) {
            stages.push(None);
            continue;
        }

        stages.push(Some(ResolvedStage { spec, info }));
    }

    check_has_sink(specs, diagnostics);

    Resolution { stages, by_name }
}

/// Maps every name to its first definition, rejecting anything that repeats one.
fn collect_names(specs: &[StageSpec], diagnostics: &mut Diagnostics) -> HashMap<String, usize> {
    let mut by_name: HashMap<String, usize> = HashMap::new();

    for (index, spec) in specs.iter().enumerate() {
        let existing = by_name.get(&spec.name.value).copied();
        let Some(first) = existing else {
            by_name.insert(spec.name.value.clone(), index);
            continue;
        };

        let message = format!("a stage named {} is already defined", spec.name.value);
        let diagnostic = Diagnostic::error(
            DiagnosticCode::DuplicateStageName,
            stage_path(spec.kind, &spec.name.value),
            spec.name.span,
            message,
        )
        .with_note("first defined here", specs[first].name.span);

        diagnostics.push(diagnostic);
    }

    by_name
}

fn index_registry(registry: &PipelineRegistry) -> HashMap<(StageKind, &str), StageInfo<'_>> {
    registry
        .stages()
        .map(|info| ((info.kind(), info.full_identifier()), info))
        .collect()
}

/// Finds the stage's registry entry, reporting whether it is missing or under another kind.
fn look_up<'a>(
    registered: &HashMap<(StageKind, &'a str), StageInfo<'a>>,
    spec: &StageSpec,
    diagnostics: &mut Diagnostics,
) -> Option<StageInfo<'a>> {
    let identifier = spec.stage_type.value.as_str();
    let found = registered.get(&(spec.kind, identifier)).copied();
    if let Some(info) = found {
        return Some(info);
    }

    let elsewhere = registered
        .values()
        .find(|info| info.full_identifier() == identifier);

    let (code, message) = match elsewhere {
        Some(info) => (
            DiagnosticCode::KindMismatch,
            format!("{identifier} is a {} stage, not a {}", info.kind(), spec.kind),
        ),
        None => (
            DiagnosticCode::UnknownStage,
            format!("no {} stage is registered as {identifier}", spec.kind),
        ),
    };

    diagnostics.push(Diagnostic::error(
        code,
        type_path(spec.kind, &spec.name.value),
        spec.stage_type.span,
        message,
    ));

    None
}

/// Checks the rules about which kinds may be wired to which, returning whether the spec passed.
fn check_inputs(
    spec: &StageSpec,
    specs: &[StageSpec],
    by_name: &HashMap<String, usize>,
    diagnostics: &mut Diagnostics,
) -> bool {
    if spec.kind == StageKind::Source && !spec.inputs.is_empty() {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::SourceTakesInputs,
            input_path(spec.kind, &spec.name.value, 0),
            spec.inputs[0].span,
            "a source reads nothing, so it takes no inputs",
        ));

        return false;
    }

    let mut accepted = true;
    for (index, input) in spec.inputs.iter().enumerate() {
        if !check_input(spec, specs, by_name, input, index, diagnostics) {
            accepted = false;
        }
    }

    accepted
}

fn check_input(
    spec: &StageSpec,
    specs: &[StageSpec],
    by_name: &HashMap<String, usize>,
    input: &Spanned<crate::InputRef>,
    index: usize,
    diagnostics: &mut Diagnostics,
) -> bool {
    let path = input_path(spec.kind, &spec.name.value, index);
    let target = by_name.get(&input.value.target).copied();

    let Some(position) = target else {
        let mut message = format!("no stage named {} is defined", input.value.target);
        let known_names = by_name.keys().map(String::as_str);
        if let Some(closest) = suggest_name(&input.value.target, known_names) {
            message.push_str(&format!(", did you mean {closest}"));
        }

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::UnknownStageReference,
            path,
            input.span,
            message,
        ));

        return false;
    };

    if specs[position].kind != StageKind::Sink {
        return true;
    }

    let message = format!("{} is a sink, which produces no output", input.value.target);
    diagnostics.push(Diagnostic::error(
        DiagnosticCode::SinkHasNoOutput,
        path,
        input.span,
        message,
    ));

    false
}

fn check_has_sink(specs: &[StageSpec], diagnostics: &mut Diagnostics) {
    if specs.iter().any(|spec| spec.kind == StageKind::Sink) {
        return;
    }

    diagnostics.push(Diagnostic::error(
        DiagnosticCode::NoSinks,
        "",
        crate::Span::unknown(),
        "the pipeline has no sink, so nothing consumes its output",
    ));
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;
    use crate::testing;

    /// Resolves the specs and returns the codes raised, in order.
    fn codes(specs: &[StageSpec]) -> Vec<DiagnosticCode> {
        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        resolve(&registry, specs, &mut diagnostics);

        diagnostics.iter().map(|entry| entry.code()).collect()
    }

    /// A source, a filter reading it and a sink reading the filter.
    fn working_specs() -> Vec<StageSpec> {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, sink]
    }

    #[test]
    fn a_working_pipeline_resolves_every_stage() {
        let specs = working_specs();
        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();

        let resolution = resolve(&registry, &specs, &mut diagnostics);

        assert!(diagnostics.is_empty());
        assert!(resolution.stages.iter().all(|stage| stage.is_some()));
    }

    #[test]
    fn a_repeated_name_is_a_duplicate() {
        let mut specs = working_specs();
        specs.push(testing::spec(StageKind::Source, "input", "test/file"));

        assert!(codes(&specs).contains(&DiagnosticCode::DuplicateStageName));
    }

    #[test]
    fn an_unregistered_type_is_unknown() {
        let mut specs = working_specs();
        specs[1].stage_type = Spanned::unknown(String::from("test/nope"));

        assert!(codes(&specs).contains(&DiagnosticCode::UnknownStage));
    }

    #[test]
    fn a_type_registered_under_another_kind_is_a_kind_mismatch() {
        let mut specs = working_specs();
        specs[1].stage_type = Spanned::unknown(String::from("test/merge"));

        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        resolve(&registry, &specs, &mut diagnostics);

        let mismatch = diagnostics
            .iter()
            .find(|entry| entry.code() == DiagnosticCode::KindMismatch)
            .expect("should report a kind mismatch");

        assert!(mismatch.message().contains("mask"));
    }

    #[test]
    fn a_source_given_inputs_is_rejected() {
        let mut specs = working_specs();
        specs[0].inputs = vec![testing::positional("blur")];

        assert!(codes(&specs).contains(&DiagnosticCode::SourceTakesInputs));
    }

    #[test]
    fn an_input_naming_nothing_suggests_the_closest_name() {
        let mut specs = working_specs();
        specs[1].inputs = vec![testing::positional("inputt")];

        let registry = testing::registry();
        let mut diagnostics = Diagnostics::new();
        resolve(&registry, &specs, &mut diagnostics);

        let unknown = diagnostics
            .iter()
            .find(|entry| entry.code() == DiagnosticCode::UnknownStageReference)
            .expect("should report an unknown reference");

        assert!(unknown.message().contains("input"));
        assert_eq!(unknown.path(), "filter.blur.inputs[0]");
    }

    #[test]
    fn reading_a_sink_is_rejected() {
        let mut specs = working_specs();
        specs[1].inputs = vec![testing::positional("out")];

        assert!(codes(&specs).contains(&DiagnosticCode::SinkHasNoOutput));
    }

    #[test]
    fn a_pipeline_with_no_sink_is_rejected() {
        let specs = vec![testing::spec(StageKind::Source, "input", "test/file")];

        assert!(codes(&specs).contains(&DiagnosticCode::NoSinks));
    }
}
