use av_toolbox_core::{PipelineRegistry, StageKind};
use av_toolbox_gpu::Backend;
use av_toolbox_graph::{
    CompiledGraph,
    Diagnostic,
    NodeId,
    NodeRole,
    RenderOptions,
    SolvedFormats,
    Span,
    StageSpec,
    default_passes,
    optimise,
    place_transfers,
    plan,
    render_ascii,
    render_html,
    solve,
};
use av_toolbox_runtime::{Engine, Settings};

use crate::PipelineError;

/// A stage as the pipeline wrote it.
#[derive(Debug, Copy, Clone)]
pub struct StageRef<'a> {
    pub kind: StageKind,
    pub name: &'a str,
    pub span: Span,
}

/// A checked and optimised pipeline, ready to draw or compile.
pub struct Pipeline<'r> {
    registry: &'r PipelineRegistry,
    graph: CompiledGraph,
    warnings: Vec<Diagnostic>,
}

impl<'r> Pipeline<'r> {
    pub fn build(registry: &'r PipelineRegistry, specs: &[StageSpec]) -> Result<Self, PipelineError> {
        let built = av_toolbox_graph::build(registry, specs)?;
        let warnings = built.warnings().to_vec();
        let mut graph = built.into_graph();

        let passes = default_passes();
        optimise(&mut graph, &passes);

        Ok(Self {
            registry,
            graph,
            warnings,
        })
    }

    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }

    pub fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    pub fn render_ascii(&self) -> String {
        let options = RenderOptions::default();
        render_ascii(&self.graph, &options)
    }

    pub fn render_html(&self) -> String {
        let options = RenderOptions::default();
        render_html(&self.graph, &options)
    }

    pub fn stage_of(&self, id: NodeId) -> Option<StageRef<'_>> {
        stage_of(&self.graph, id)
    }

    /// Settles every edge's format, places the transfers and plans the run on `backend`.
    pub fn compile<'b>(self, backend: &'b Backend) -> Result<CompiledPipeline<'r, 'b>, PipelineError> {
        let Self {
            registry,
            mut graph,
            mut warnings,
        } = self;

        let capabilities = backend.capabilities();
        let mut formats = solve(&graph, registry, capabilities)?;

        let placed = place_transfers(&mut graph, registry, &mut formats)?;
        warnings.extend(placed.iter().cloned());

        let made = plan(&graph, registry)?;
        graph.set_execution_plan(made);

        Ok(CompiledPipeline {
            registry,
            backend,
            graph,
            formats,
            warnings,
        })
    }
}

/// A pipeline planned on a backend, ready to draw or run.
pub struct CompiledPipeline<'r, 'b> {
    registry: &'r PipelineRegistry,
    backend: &'b Backend,
    graph: CompiledGraph,
    formats: SolvedFormats,
    warnings: Vec<Diagnostic>,
}

impl CompiledPipeline<'_, '_> {
    pub fn warnings(&self) -> &[Diagnostic] {
        &self.warnings
    }

    pub fn graph(&self) -> &CompiledGraph {
        &self.graph
    }

    pub fn render_ascii(&self) -> String {
        let options = RenderOptions::default();
        render_ascii(&self.graph, &options)
    }

    pub fn render_html(&self) -> String {
        let options = RenderOptions::default();
        render_html(&self.graph, &options)
    }

    pub fn stage_of(&self, id: NodeId) -> Option<StageRef<'_>> {
        stage_of(&self.graph, id)
    }

    /// Builds every stage and the engine that runs them.
    pub fn engine(&self, settings: Settings) -> Result<Engine, PipelineError> {
        let engine = Engine::build(&self.graph, self.registry, self.backend, &self.formats, settings)?;

        Ok(engine)
    }
}

/// The written stage behind a node, or `None` for a node the compiler inserted.
fn stage_of(graph: &CompiledGraph, id: NodeId) -> Option<StageRef<'_>> {
    let node = graph.nodes().get(id.index())?;
    if node.role() != NodeRole::Stage {
        return None;
    }

    Some(StageRef {
        kind: node.kind(),
        name: node.name(),
        span: node.span(),
    })
}

#[cfg(test)]
mod tests {
    use av_toolbox_graph::{DiagnosticCode, NodeId};

    use super::*;

    const PASSTHROUGH: &str = r#"
[source.input]
type = "y4m/stdin"

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;

    fn specs(text: &str) -> Vec<StageSpec> {
        crate::toml::parse(text).expect("the file should parse")
    }

    #[test]
    fn a_valid_pipeline_builds_and_renders() {
        let registry = crate::registry().expect("the built-in stages register");
        let specs = specs(PASSTHROUGH);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let drawn = pipeline.render_ascii();

        assert!(drawn.contains("input"));
        assert!(drawn.contains("output"));
    }

    #[test]
    fn a_broken_pipeline_returns_its_diagnostics() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = PASSTHROUGH.replace("[\"input\"]", "[\"missing\"]");
        let specs = specs(&text);

        let result = Pipeline::build(&registry, &specs);
        let Err(error) = result else {
            panic!("should fail");
        };
        let codes: Vec<DiagnosticCode> = error
            .diagnostics()
            .iter()
            .map(|diagnostic| diagnostic.code())
            .collect();

        assert!(codes.contains(&DiagnosticCode::UnknownStageReference));
    }

    #[test]
    fn an_unused_stage_is_kept_as_a_warning() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = format!("{PASSTHROUGH}\n[source.spare]\ntype = \"y4m/pipe\"\nargs.path = \"spare.y4m\"\n");
        let specs = specs(&text);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let codes: Vec<DiagnosticCode> = pipeline.warnings().iter().map(|warning| warning.code()).collect();

        assert_eq!(codes, [DiagnosticCode::UnusedStage]);
    }

    #[test]
    fn stage_of_names_each_written_stage() {
        let registry = crate::registry().expect("the built-in stages register");
        let specs = specs(PASSTHROUGH);
        let pipeline = Pipeline::build(&registry, &specs).expect("should build");

        let mut found: Vec<(String, Span)> = pipeline
            .graph()
            .nodes()
            .iter()
            .filter_map(|node| pipeline.stage_of(node.id()))
            .map(|stage| (stage.name.to_string(), stage.span))
            .collect();
        found.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(
            found,
            vec![
                ("input".to_string(), specs[0].span),
                ("output".to_string(), specs[1].span)
            ]
        );
    }

    #[test]
    fn stage_of_an_unknown_node_is_none() {
        let registry = crate::registry().expect("the built-in stages register");
        let specs = specs(PASSTHROUGH);
        let pipeline = Pipeline::build(&registry, &specs).expect("should build");

        assert!(pipeline.stage_of(NodeId::new(99)).is_none());
    }
}
