mod bind;
mod checks;
mod resolve;

use av_toolbox_core::PipelineRegistry;

use crate::graph::CompiledGraph;
use crate::{Diagnostic, Diagnostics, StageSpec};

/// A graph that built, together with anything worth warning about.
#[derive(Debug)]
pub struct Built {
    graph: CompiledGraph,
    warnings: Vec<Diagnostic>,
}

impl Built {
    pub fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    pub fn into_graph(self) -> CompiledGraph {
        self.graph
    }

    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }
}

/// Resolves and validates a set of definitions into a graph.
///
/// Every phase collects all of its problems before the next one runs, so one call reports
/// everything wrong with the definitions rather than only the first thing.
pub fn build(registry: &PipelineRegistry, specs: &[StageSpec]) -> Result<Built, Diagnostics> {
    let mut diagnostics = Diagnostics::new();

    let resolution = resolve::resolve(registry, specs, &mut diagnostics);
    let nodes = bind::bind(specs, &resolution, &mut diagnostics);
    checks::detect_cycles(&nodes, &mut diagnostics);

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let graph = CompiledGraph::new(nodes);
    checks::report_unused(&graph, &mut diagnostics);

    Ok(Built {
        graph,
        warnings: diagnostics.into_inner(),
    })
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;

    use super::*;
    use crate::{DiagnosticCode, StageSpec, testing};

    fn pipeline() -> Vec<StageSpec> {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, sink]
    }

    #[test]
    fn a_working_pipeline_builds_in_dependency_order() {
        let registry = testing::registry();
        let built = build(&registry, &pipeline()).expect("should build");

        let graph = built.graph();
        let names: Vec<&str> = graph.order().iter().map(|id| graph.node(*id).name()).collect();

        assert_eq!(names, vec!["input", "blur", "out"]);
        assert!(built.warnings().is_empty());
    }

    #[test]
    fn a_stage_no_sink_reads_builds_with_a_warning() {
        let mut specs = pipeline();

        let mut spare = testing::spec(StageKind::Filter, "spare", "test/blur");
        spare.inputs = vec![testing::positional("input")];
        specs.push(spare);

        let registry = testing::registry();
        let built = build(&registry, &specs).expect("should build");

        assert_eq!(built.warnings().len(), 1);
        assert_eq!(built.warnings()[0].code(), DiagnosticCode::UnusedStage);
    }

    #[test]
    fn problems_from_different_phases_are_reported_together() {
        let mut specs = pipeline();
        specs[1].stage_type = crate::Spanned::unknown(String::from("test/nope"));
        specs[2].inputs = Vec::new();

        let registry = testing::registry();
        let diagnostics = build(&registry, &specs).expect_err("should fail");
        let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

        assert!(codes.contains(&DiagnosticCode::UnknownStage));
        assert!(codes.contains(&DiagnosticCode::InvalidInputs));
    }

    #[test]
    fn a_cycle_fails_the_build() {
        let mut first = testing::spec(StageKind::Filter, "first", "test/blur");
        first.inputs = vec![testing::positional("second")];

        let mut second = testing::spec(StageKind::Filter, "second", "test/blur");
        second.inputs = vec![testing::positional("first")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("first")];

        let registry = testing::registry();
        let diagnostics = build(&registry, &[first, second, sink]).expect_err("should fail");
        let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|entry| entry.code()).collect();

        assert_eq!(codes, vec![DiagnosticCode::Cycle]);
    }
}
