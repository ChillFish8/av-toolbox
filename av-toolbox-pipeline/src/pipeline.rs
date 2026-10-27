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
    inject_output_depth,
    insert_widens,
    optimise,
    place_transfers,
    plan,
    record_probes,
    render_ascii,
    render_html,
    run_probes,
    satisfy_demands,
    settle_mask_planes,
    solve,
};
use av_toolbox_runtime::{Engine, Settings};

use crate::PipelineError;

/// A stage in the pipeline, written or injected by the graph.
#[derive(Debug, Copy, Clone)]
pub struct StageRef<'a> {
    pub kind: StageKind,
    pub name: &'a str,
    pub span: Span,
}

/// A written source whose input is a live stream.
#[derive(Debug, Copy, Clone)]
pub struct LiveSource<'a> {
    pub stage: StageRef<'a>,
    pub stage_type: &'a str,
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
        satisfy_demands(&mut graph, registry)?;
        record_probes(&mut graph, registry)?;

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

    /// Every written source in the graph whose input is a live stream.
    pub fn live_sources(&self) -> Vec<LiveSource<'_>> {
        live_sources(&self.graph, self.registry)
    }

    /// Brings each sink that wants it to the output depth, settles every edge's format, places the transfers and
    /// plans the run on `backend`.
    pub fn compile<'b>(self, backend: &'b Backend) -> Result<CompiledPipeline<'r, 'b>, PipelineError> {
        let Self {
            registry,
            mut graph,
            mut warnings,
        } = self;

        run_probes(&mut graph, registry)?;

        let capabilities = backend.capabilities();
        inject_output_depth(&mut graph, registry, capabilities)?;

        let mut formats = solve(&graph, registry, capabilities)?;
        insert_widens(&mut graph, &mut formats);
        settle_mask_planes(&graph, registry, &mut formats)?;

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

    /// The format settled on each node's inputs and output.
    pub fn formats(&self) -> &SolvedFormats {
        &self.formats
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

/// The stage behind a node, whether written or injected by the graph, or `None` for a provider or crossing.
fn stage_of(graph: &CompiledGraph, id: NodeId) -> Option<StageRef<'_>> {
    let node = graph.nodes().get(id.index())?;
    if *node.role() != NodeRole::Stage {
        return None;
    }

    Some(StageRef {
        kind: node.kind(),
        name: node.name(),
        span: node.span(),
    })
}

/// Every written source in `graph` whose input is a live stream, according to `registry`.
fn live_sources<'a>(graph: &'a CompiledGraph, registry: &PipelineRegistry) -> Vec<LiveSource<'a>> {
    let mut found = Vec::new();

    for node in graph.nodes() {
        let is_written_source = *node.role() == NodeRole::Stage && node.kind() == StageKind::Source;
        if !is_written_source {
            continue;
        }

        let identifier = node.identifier();
        let args = node.args();
        let reads_stream = registry.reads_stream(identifier, args).unwrap_or(false);

        if !reads_stream {
            continue;
        }

        let stage = StageRef {
            kind: node.kind(),
            name: node.name(),
            span: node.span(),
        };

        found.push(LiveSource {
            stage,
            stage_type: identifier,
        });
    }

    found
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use av_toolbox_graph::{Diagnostic, DiagnosticCode, Node, NodeId};

    use super::*;
    use crate::testing::MemoryFiles;
    use crate::{Supplied, load};

    const PASSTHROUGH: &str = r#"
[source.input]
type = "y4m/stdin"

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;

    fn specs(text: &str) -> Vec<StageSpec> {
        let files = MemoryFiles::new(&[("/pipeline.toml", text)]);
        let supplied = Supplied::new();
        let path = Path::new("/pipeline.toml");

        let loaded = load(path, &files, &supplied).expect("the file should load");
        loaded.specs
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
    fn building_places_a_scene_provider_for_a_chunk_stage() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/stdin"

[filter.deinterlace]
type = "deinterlace/bwdif"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["deinterlace"]
"#;
        let specs = specs(text);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let graph = pipeline.graph();
        let provider = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "scenes(input)")
            .expect("a scene provider follows the source");
        let deinterlace = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "deinterlace")
            .expect("the deinterlacer is kept");

        assert!(matches!(provider.role(), NodeRole::Provider(_)));
        assert_eq!(provider.identifier(), "provider/scene-split");
        assert_eq!(deinterlace.inputs()[0].source(), provider.id());
        assert!(deinterlace.route().is_some());
        assert!(pipeline.stage_of(provider.id()).is_none());
    }

    #[test]
    fn an_after_moves_only_the_demands_naming_it() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/stdin"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["denoise"]

[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "scenes.ndjson"
args.scenes = { breaks = { mode = "natural" }, after = "denoise" }
"#;
        let specs = specs(text);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let graph = pipeline.graph();
        let name_of = |id: NodeId| graph.node(id).name().to_string();
        let node_named = |name: &str| {
            graph
                .nodes()
                .iter()
                .find(|node| node.name() == name)
                .unwrap_or_else(|| panic!("the graph has no node named {name}"))
        };

        let providers: Vec<&Node> = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node.role(), NodeRole::Provider(_)))
            .collect();
        assert_eq!(providers.len(), 4);

        let denoise = node_named("denoise");
        let log = node_named("log");
        let output = node_named("output");
        let before = reads(graph, denoise);
        let after = reads(graph, log);
        let before_score = reads(graph, before);
        let after_score = reads(graph, after);

        assert_eq!(before_score.identifier(), "provider/scene-score");
        assert_eq!(after_score.identifier(), "provider/scene-score");
        assert_eq!(reads(graph, before_score).name(), "input");
        assert_eq!(reads(graph, after_score).name(), "denoise");
        assert_eq!(name_of(output.inputs()[0].source()), "denoise");

        let NodeRole::Provider(before_placement) = before.role() else {
            panic!("{} is not a provider", before.name());
        };
        let NodeRole::Provider(after_placement) = after.role() else {
            panic!("{} is not a provider", after.name());
        };
        let before_demanders: Vec<&[String]> = before_placement
            .layers
            .iter()
            .map(|served| served.demanded_by.as_slice())
            .collect();
        let after_demanders: Vec<&[String]> = after_placement
            .layers
            .iter()
            .map(|served| served.demanded_by.as_slice())
            .collect();
        assert_eq!(before_demanders, [["denoise".to_string()]]);
        assert_eq!(after_demanders, [["log".to_string()]]);
    }

    #[test]
    fn an_after_typo_is_reported_once_at_the_sink() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/stdin"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["denoise"]

[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "scenes.ndjson"
args.scenes = { breaks = { mode = "natural" }, after = "denoize" }
"#;
        let specs = specs(text);

        let result = Pipeline::build(&registry, &specs);
        let Err(error) = result else {
            panic!("should fail");
        };
        let diagnostics: Vec<&Diagnostic> = error.diagnostics().iter().collect();

        assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
        assert_eq!(diagnostics[0].code(), DiagnosticCode::PlaceFailed);
        assert_eq!(diagnostics[0].path(), "sink.log");
        assert!(
            diagnostics[0]
                .message()
                .contains("there is no stage named `denoize`"),
            "{}",
            diagnostics[0].message(),
        );
    }

    fn nodes_of<'a>(graph: &'a CompiledGraph, identifier: &str) -> Vec<&'a Node> {
        graph
            .nodes()
            .iter()
            .filter(|node| node.identifier() == identifier)
            .collect()
    }

    fn reads<'a>(graph: &'a CompiledGraph, node: &Node) -> &'a Node {
        let source = node.inputs()[0].source();

        graph.node(source)
    }

    fn layer_count(node: &Node) -> usize {
        let NodeRole::Provider(placement) = node.role() else {
            panic!("{} is not a provider", node.name());
        };

        placement.layers.len()
    }

    #[test]
    fn natural_and_limited_on_one_source_detect_once() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/stdin"

[sink.whole]
type = "metadata/ndjson"
inputs = ["input"]
args.path = "whole.ndjson"
args.scenes = { breaks = { mode = "natural" } }

[sink.pieces]
type = "metadata/ndjson"
inputs = ["input"]
args.path = "pieces.ndjson"
args.scenes = { breaks = { mode = "limited", max_frames = 12 } }
"#;
        let specs = specs(text);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let graph = pipeline.graph();
        let scores = nodes_of(graph, "provider/scene-score");
        let splits = nodes_of(graph, "provider/scene-split");

        assert_eq!(scores.len(), 1);
        assert_eq!(splits.len(), 1);
        assert_eq!(layer_count(splits[0]), 2);
        assert_eq!(reads(graph, splits[0]).id(), scores[0].id());
        assert_eq!(reads(graph, scores[0]).name(), "input");
    }

    #[test]
    fn a_split_after_a_stage_gets_its_own_score() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/stdin"

[filter.denoise]
type = "av-denoise/nl4d"
inputs = ["input"]

[sink.output]
type = "y4m/stdout"
inputs = ["denoise"]

[sink.log]
type = "metadata/ndjson"
inputs = ["denoise"]
args.path = "scenes.ndjson"
args.scenes = { breaks = { mode = "natural" }, after = "denoise" }
"#;
        let specs = specs(text);

        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let graph = pipeline.graph();
        let scores = nodes_of(graph, "provider/scene-score");
        let splits = nodes_of(graph, "provider/scene-split");

        assert_eq!(scores.len(), 2);
        assert_eq!(splits.len(), 2);

        let mut follows = Vec::new();
        for split in splits {
            let score = reads(graph, split);
            assert_eq!(score.identifier(), "provider/scene-score", "{}", split.name());

            let producer = reads(graph, score);
            follows.push(producer.name().to_string());
        }

        follows.sort();
        assert_eq!(follows, ["denoise", "input"]);
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

    #[test]
    fn live_sources_lists_a_source_that_reads_a_stream() {
        let registry = crate::registry().expect("the built-in stages register");
        let specs = specs(PASSTHROUGH);
        let pipeline = Pipeline::build(&registry, &specs).expect("should build");

        let live_sources = pipeline.live_sources();
        let names: Vec<&str> = live_sources
            .iter()
            .map(|live_source| live_source.stage.name)
            .collect();

        assert_eq!(names, vec!["input"]);
    }

    #[test]
    fn live_sources_is_empty_for_a_regular_file_pipe() {
        let registry = crate::registry().expect("the built-in stages register");
        let text = r#"
[source.input]
type = "y4m/pipe"
args.path = "clip.y4m"

[sink.output]
type = "y4m/stdout"
inputs = ["input"]
"#;
        let specs = specs(text);
        let pipeline = Pipeline::build(&registry, &specs).expect("should build");
        let live_sources = pipeline.live_sources();

        assert!(live_sources.is_empty());
    }
}
