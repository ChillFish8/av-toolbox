use std::fmt;

use av_toolbox_core::{
    ArgsError,
    BuildStageError,
    Filter,
    InputArgument,
    Mask,
    PipelineRegistry,
    PreparedStage,
    Sink,
    Source,
    Transform,
};
use av_toolbox_gpu::{Backend, ResolvedFormats};

use crate::diagnostic::{args_path, stage_path};
use crate::graph::{CompiledGraph, Node};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, SolvedFormats};

/// A built stage, in whichever of the five kinds it belongs to.
pub enum StageBox {
    Source(Box<dyn Source>),
    Filter(Box<dyn Filter>),
    Mask(Box<dyn Mask>),
    Transform(Box<dyn Transform>),
    Sink(Box<dyn Sink>),
}

impl fmt::Debug for StageBox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Source(_) => "Source",
            Self::Filter(_) => "Filter",
            Self::Mask(_) => "Mask",
            Self::Transform(_) => "Transform",
            Self::Sink(_) => "Sink",
        };

        write!(formatter, "StageBox::{name}")
    }
}

impl CompiledGraph {
    /// Validates and binds every stage's arguments and inputs, without needing a device.
    ///
    /// Every node is attempted, so a node whose arguments or inputs are invalid does not stop
    /// another from being prepared in the same run. Its failure is pushed onto `diagnostics` and
    /// its slot in the result is left empty, rather than stopping the whole run early.
    pub(crate) fn prepare_stages(
        &self,
        registry: &PipelineRegistry,
        diagnostics: &mut Diagnostics,
    ) -> Vec<Option<PreparedStage>> {
        let mut prepared: Vec<Option<PreparedStage>> = (0..self.len()).map(|_| None).collect();

        for id in self.order() {
            let node = self.node(*id);
            let arguments = arguments_for(self, node);

            match registry.prepare(node.kind(), node.identifier(), node.args(), &arguments) {
                Ok(stage) => prepared[id.index()] = Some(stage),
                Err(error) => report(node, error, diagnostics),
            }
        }

        prepared
    }

    /// Builds every stage through the registry, in dependency order.
    ///
    /// `formats` is what solving the same graph produced, so it always covers every node here.
    ///
    /// Arguments and inputs are checked before any stage is built, so a schema violation and a
    /// failure raised by the stage itself are reported together. A node whose arguments or inputs
    /// are invalid is skipped rather than built, but every other node is still attempted, so one
    /// run names every stage that could not be prepared or built.
    pub fn instantiate(
        &self,
        registry: &PipelineRegistry,
        backend: &Backend,
        formats: &SolvedFormats,
    ) -> Result<Vec<StageBox>, Diagnostics> {
        let mut diagnostics = Diagnostics::new();
        let mut prepared = self.prepare_stages(registry, &mut diagnostics);

        let mut built: Vec<Option<StageBox>> = (0..self.len()).map(|_| None).collect();

        for id in self.order() {
            let Some(stage) = prepared[id.index()].take() else {
                continue;
            };

            let node = self.node(*id);
            let node_formats = formats.get(*id);

            match build_stage(stage, backend, node_formats) {
                Ok(stage_box) => built[id.index()] = Some(stage_box),
                Err(error) => report(node, error, &mut diagnostics),
            }
        }

        if diagnostics.has_errors() {
            return Err(diagnostics);
        }

        let stages = built
            .into_iter()
            .map(|stage| stage.expect("every stage built without a diagnostic"))
            .collect();

        Ok(stages)
    }
}

/// Rebuilds the input list exactly as the definition wrote it, so binding repeats what it did.
fn arguments_for(graph: &CompiledGraph, node: &Node) -> Vec<InputArgument> {
    node.inputs()
        .iter()
        .map(|edge| {
            let handle = av_toolbox_core::StageHandle::new(edge.source().index() as u64);
            let stream = graph
                .node(edge.source())
                .kind()
                .output()
                .expect("a sink cannot be an input");

            match edge.name() {
                Some(name) => InputArgument::named(name, handle, stream),
                None => InputArgument::positional(handle, stream),
            }
        })
        .collect()
}

fn build_stage(
    prepared: PreparedStage,
    backend: &Backend,
    formats: &ResolvedFormats,
) -> Result<StageBox, BuildStageError> {
    let stage = match prepared {
        PreparedStage::Source(prepared) => StageBox::Source(prepared.build(backend, formats)?),
        PreparedStage::Filter(prepared) => StageBox::Filter(prepared.build(backend, formats)?),
        PreparedStage::Mask(prepared) => StageBox::Mask(prepared.build(backend, formats)?),
        PreparedStage::Transform(prepared) => StageBox::Transform(prepared.build(backend, formats)?),
        PreparedStage::Sink(prepared) => StageBox::Sink(prepared.build(backend, formats)?),
    };

    Ok(stage)
}

/// Puts a build failure where the definition can be corrected.
///
/// A schema violation lands on the arguments table with the violation's own pointer appended, and
/// anything else lands on the stage.
fn report(node: &Node, error: BuildStageError, diagnostics: &mut Diagnostics) {
    let BuildStageError::InvalidArgs { source, .. } = &error else {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            describe(&error),
        ));

        return;
    };

    let ArgsError::SchemaViolations(violations) = source else {
        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidArgs,
            args_path(node.kind(), node.name()),
            node.args_span(),
            describe(&error),
        ));

        return;
    };

    for violation in violations {
        let path = format!(
            "{}{}",
            args_path(node.kind(), node.name()),
            violation.path().replace('/', ".")
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidArgs,
            path,
            node.args_span(),
            violation.message(),
        ));
    }
}

/// Renders an error together with whatever it was caused by.
fn describe(error: &BuildStageError) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);

    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }

    message
}

/// Diagnostics raised while preparing a graph need no device, so these tests join the fast suite
/// rather than the device-gated module below.
#[cfg(test)]
mod diagnostics_tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use crate::{Diagnostic, DiagnosticCode, Diagnostics, Span, Spanned, StageSpec, build, testing};

    /// The pipeline every test in this file builds from, tweaking whichever node it needs to.
    ///
    /// `pub(super)` rather than a copy, so the device-gated `tests` module below shares it.
    pub(super) fn pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "example.mkv" }));

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 0.8 }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, sink]
    }

    #[test]
    fn a_schema_violation_lands_on_the_arguments() {
        let mut specs = pipeline();
        specs[1].args = Spanned::new(json!({ "sigma": "wide" }), Span::new(40, 60));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let mut diagnostics = Diagnostics::new();
        graph.prepare_stages(&registry, &mut diagnostics);
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::InvalidArgs);
        assert!(reported[0].path().starts_with("filter.blur.args"));
        assert_eq!(reported[0].span(), Span::new(40, 60));
    }

    #[test]
    fn every_failing_stage_is_reported_not_just_the_first() {
        let mut specs = pipeline();
        specs[0].args = Spanned::unknown(json!({}));
        specs[1].args = Spanned::unknown(json!({ "sigma": "wide" }));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let mut diagnostics = Diagnostics::new();
        graph.prepare_stages(&registry, &mut diagnostics);

        assert_eq!(diagnostics.iter().count(), 2);
    }
}

// Every test here calls `instantiate` through to a real build, which needs a real backend, so the
// whole module needs a device. The argument-diagnostics tests that do not need one live in
// `diagnostics_tests` above.
#[cfg(test)]
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod tests {
    use std::sync::atomic::Ordering;

    use av_toolbox_gpu::{
        ColourInfo,
        Device,
        FrameFormat,
        FrameLayout,
        SampleFormat,
        StreamFormat,
        Subsampling,
    };
    use serde_json::json;

    use super::diagnostics_tests::pipeline;
    use super::*;
    use crate::solve::solve;
    use crate::{DiagnosticCode, Spanned, build, testing};

    fn backend() -> Backend {
        Backend::open_default(Device::Default).expect("should open")
    }

    /// Solves `graph` for real, against the capabilities of the backend it will build against.
    fn formats(graph: &CompiledGraph, registry: &PipelineRegistry, backend: &Backend) -> SolvedFormats {
        solve(graph, registry, backend.capabilities()).expect("should solve")
    }

    /// One set of formats per node, each carrying a width that identifies its node.
    ///
    /// The fixture stages here pass a format through unchanged, so a real solve settles every node
    /// on the same width, and a test proving that node 1 got node 1's formats needs this synthetic
    /// path instead.
    fn synthetic_formats(graph: &CompiledGraph) -> SolvedFormats {
        let per_node = (0..graph.len())
            .map(|index| {
                let format = frames(100 + index as u32);
                ResolvedFormats::new(vec![Some(format)], Some(format))
            })
            .collect();

        SolvedFormats::new(per_node)
    }

    fn frames(width: u32) -> StreamFormat {
        StreamFormat::Frames(FrameFormat {
            width,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
        })
    }

    #[test]
    fn every_stage_is_built_and_indexed_by_node() {
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();
        let backend = backend();
        let formats = formats(&graph, &registry, &backend);

        let stages = graph
            .instantiate(&registry, &backend, &formats)
            .expect("should instantiate");

        assert_eq!(stages.len(), 3);
        assert!(matches!(stages[0], StageBox::Source(_)));
        assert!(matches!(stages[1], StageBox::Filter(_)));
        assert!(matches!(stages[2], StageBox::Sink(_)));
    }

    /// The formats a solved graph resolves for a node have to arrive at that node's stage, and no
    /// other node's.
    #[test]
    fn a_stage_is_built_with_its_own_nodes_formats() {
        let mut specs = pipeline();
        specs[1].stage_type = Spanned::unknown(String::from("test/recording"));
        specs[1].args = Spanned::unknown(json!({}));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = synthetic_formats(&graph);

        testing::RECORDED_INPUT_WIDTH.store(0, Ordering::SeqCst);

        graph
            .instantiate(&registry, &backend, &formats)
            .expect("should instantiate");

        assert_eq!(testing::RECORDED_INPUT_WIDTH.load(Ordering::SeqCst), 101);
    }

    #[test]
    fn a_stage_that_fails_to_build_is_reported_with_its_own_message() {
        let mut specs = pipeline();
        specs[1].stage_type = Spanned::unknown(String::from("test/failing"));
        specs[1].args = Spanned::unknown(json!({}));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = formats(&graph, &registry, &backend);

        let diagnostics = graph
            .instantiate(&registry, &backend, &formats)
            .expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::StageBuildFailed);
        assert!(reported[0].message().contains("this stage always fails"));
    }

    /// A node that never reaches the backend and a node that fails once it does are two different
    /// phases of the same run, and both should come back from one call rather than needing two.
    ///
    /// The first node's arguments are invalid here, so solving this graph fails before it ever
    /// reaches the second node, which is why this uses the synthetic formats instead.
    #[test]
    fn an_argument_failure_and_a_build_failure_are_reported_together() {
        let mut specs = pipeline();
        specs[0].args = Spanned::unknown(json!({}));
        specs[1].stage_type = Spanned::unknown(String::from("test/failing"));
        specs[1].args = Spanned::unknown(json!({}));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = synthetic_formats(&graph);

        let diagnostics = graph
            .instantiate(&registry, &backend, &formats)
            .expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported.len(), 2);
        assert!(
            reported
                .iter()
                .any(|diagnostic| diagnostic.code() == DiagnosticCode::InvalidArgs)
        );
        assert!(
            reported
                .iter()
                .any(|diagnostic| diagnostic.code() == DiagnosticCode::StageBuildFailed)
        );
    }

    /// The one thing the pure solver tests cannot cover: capabilities from a real opened backend,
    /// carried through solving into the stages that backend then builds.
    #[test]
    fn a_pipeline_solves_then_instantiates_every_stage() {
        let backend = backend();
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();

        let solved = solve(&graph, &registry, backend.capabilities()).expect("should solve");
        let stages = graph
            .instantiate(&registry, &backend, &solved)
            .expect("should instantiate");

        assert_eq!(stages.len(), 3);
    }
}
