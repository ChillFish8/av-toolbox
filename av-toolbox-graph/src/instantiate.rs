use std::fmt;

use av_toolbox_core::{
    ArgsError,
    BuildStageError,
    BuiltProvider,
    Crossing,
    Filter,
    InputArgument,
    Mask,
    MetaProvider,
    MetaSink,
    PipelineRegistry,
    PreparedStage,
    Provider,
    Sided,
    Sink,
    Source,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{Backend, HostSide, ResolvedFormats};

use crate::demands::note_demanders;
use crate::diagnostic::{args_path, stage_path};
use crate::graph::{CompiledGraph, Edge, Node, NodeId, NodeRole};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, Region, SolvedFormats};

/// A built stage, in whichever kind and on whichever side it belongs to.
pub enum StageBox {
    Source(Box<dyn Source>),
    Filter(Box<dyn Filter>),
    Mask(Box<dyn Mask>),
    Provider(Box<dyn Provider>),
    Sink(Box<dyn Sink>),
    MetaProvider(Box<dyn MetaProvider>),
    MetaSink(Box<dyn MetaSink>),
    HostSource(Box<dyn Source<HostSide>>),
    HostFilter(Box<dyn Filter<HostSide>>),
    HostMask(Box<dyn Mask<HostSide>>),
    HostProvider(Box<dyn Provider<HostSide>>),
    HostSink(Box<dyn Sink<HostSide>>),
    /// A crossing that moves one item between host and device.
    Crossing(Crossing),
}

impl StageBox {
    /// The stage behind this box, or `None` for a crossing, which has no instance.
    pub fn instance_mut(&mut self) -> Option<&mut dyn StageInstance> {
        let instance: &mut dyn StageInstance = match self {
            Self::Source(stage) => stage.as_mut(),
            Self::Filter(stage) => stage.as_mut(),
            Self::Mask(stage) => stage.as_mut(),
            Self::Provider(stage) => stage.as_mut(),
            Self::Sink(stage) => stage.as_mut(),
            Self::MetaProvider(stage) => stage.as_mut(),
            Self::MetaSink(stage) => stage.as_mut(),
            Self::HostSource(stage) => stage.as_mut(),
            Self::HostFilter(stage) => stage.as_mut(),
            Self::HostMask(stage) => stage.as_mut(),
            Self::HostProvider(stage) => stage.as_mut(),
            Self::HostSink(stage) => stage.as_mut(),
            Self::Crossing(_) => return None,
        };

        Some(instance)
    }
}

impl fmt::Debug for StageBox {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Source(_) => "Source",
            Self::Filter(_) => "Filter",
            Self::Mask(_) => "Mask",
            Self::Provider(_) => "Provider",
            Self::Sink(_) => "Sink",
            Self::MetaProvider(_) => "MetaProvider",
            Self::MetaSink(_) => "MetaSink",
            Self::HostSource(_) => "HostSource",
            Self::HostFilter(_) => "HostFilter",
            Self::HostMask(_) => "HostMask",
            Self::HostProvider(_) => "HostProvider",
            Self::HostSink(_) => "HostSink",
            Self::Crossing(_) => "Crossing",
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

            if let NodeRole::Crossing(crossing) = node.role() {
                prepared[id.index()] = Some(PreparedStage::Crossing(*crossing));
                continue;
            }

            let arguments = arguments_for(self, node);

            match registry.prepare_with(
                node.kind(),
                node.identifier(),
                node.args(),
                &arguments,
                node.open_slot(),
            ) {
                Ok(mut stage) => {
                    let settled = settle_stage(node, &mut stage);
                    match settled {
                        Ok(()) => prepared[id.index()] = Some(stage),
                        Err(error) => {
                            let diagnostic = settle_failed(node, error);
                            diagnostics.push(diagnostic);
                        },
                    }
                },
                Err(error) => report(self, node, error, diagnostics),
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
                Err(error) => report(self, node, error, &mut diagnostics),
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

    /// The most frames each node may hold back at once, indexed by node.
    ///
    /// A node that cannot be prepared is reported the same way [CompiledGraph::instantiate]
    /// reports it.
    pub fn held_frames(
        &self,
        registry: &PipelineRegistry,
        formats: &SolvedFormats,
    ) -> Result<Vec<usize>, Diagnostics> {
        let mut diagnostics = Diagnostics::new();
        let prepared = self.prepare_stages(registry, &mut diagnostics);

        if diagnostics.has_errors() {
            return Err(diagnostics);
        }

        let mut held = Vec::with_capacity(prepared.len());
        for (index, stage) in prepared.iter().enumerate() {
            let node_formats = formats.get(NodeId::new(index));
            let node_held = stage.as_ref().map_or(0, |stage| stage.held_frames(node_formats));
            held.push(node_held);
        }

        Ok(held)
    }

    /// Builds one instance of every stage in `region`, in `region.nodes()` order.
    ///
    /// A worker running that region calls this once per copy it needs, so every call builds its
    /// own stages rather than sharing them with another call. Only the region's own nodes are
    /// prepared and built, since building the whole graph for every worker would build every other
    /// region's nodes too, and a source more than the one time it may exist.
    ///
    /// Every node in the region is attempted, so one call names every stage that could not be
    /// prepared or built, rather than stopping at the first failure.
    pub fn instantiate_region(
        &self,
        registry: &PipelineRegistry,
        backend: &Backend,
        formats: &SolvedFormats,
        region: &Region,
    ) -> Result<Vec<(NodeId, StageBox)>, Diagnostics> {
        let mut diagnostics = Diagnostics::new();
        let mut prepared: Vec<Option<PreparedStage>> = region.nodes().iter().map(|_| None).collect();

        for (slot, id) in region.nodes().iter().enumerate() {
            let node = self.node(*id);

            if let NodeRole::Crossing(crossing) = node.role() {
                prepared[slot] = Some(PreparedStage::Crossing(*crossing));
                continue;
            }

            let arguments = arguments_for(self, node);

            match registry.prepare_with(
                node.kind(),
                node.identifier(),
                node.args(),
                &arguments,
                node.open_slot(),
            ) {
                Ok(mut stage) => match settle_stage(node, &mut stage) {
                    Ok(()) => prepared[slot] = Some(stage),
                    Err(error) => diagnostics.push(settle_failed(node, error)),
                },
                Err(error) => report(self, node, error, &mut diagnostics),
            }
        }

        let mut stages = Vec::with_capacity(region.nodes().len());

        for (slot, id) in region.nodes().iter().enumerate() {
            let Some(stage) = prepared[slot].take() else {
                continue;
            };

            let node = self.node(*id);
            let node_formats = formats.get(*id);

            match build_stage(stage, backend, node_formats) {
                Ok(stage_box) => stages.push((*id, stage_box)),
                Err(error) => report(self, node, error, &mut diagnostics),
            }
        }

        if diagnostics.has_errors() {
            return Err(diagnostics);
        }

        Ok(stages)
    }
}

/// Rebuilds the input list exactly as the definition wrote it, so binding repeats what it did.
///
/// An input the node demanded something on carries the layer the demand pass gave that demand.
fn arguments_for(graph: &CompiledGraph, node: &Node) -> Vec<InputArgument> {
    node.inputs()
        .iter()
        .map(|edge| {
            let handle = av_toolbox_core::StageHandle::new(edge.source().index() as u64);
            let stream = graph
                .node(edge.source())
                .output_stream()
                .expect("a sink cannot be an input");

            let argument = match edge.name() {
                Some(name) => InputArgument::named(name, handle, stream),
                None => InputArgument::positional(handle, stream),
            };

            demanded_on(argument, node, edge)
        })
        .collect()
}

/// `argument` carrying the kind and layer of every demand `node` made on the input `edge` fills.
fn demanded_on(mut argument: InputArgument, node: &Node, edge: &Edge) -> InputArgument {
    let Some(input) = edge.param().or(edge.name()) else {
        return argument;
    };

    for demand in node.demands().iter().filter(|demand| demand.input == input) {
        argument = argument.with_demand(demand.kind, demand.layer);
    }

    argument
}

fn build_stage(
    prepared: PreparedStage,
    backend: &Backend,
    formats: &ResolvedFormats,
) -> Result<StageBox, BuildStageError> {
    let stage = match prepared {
        PreparedStage::Source(prepared) => match prepared.build(backend, formats)? {
            Sided::Device(stage) => StageBox::Source(stage),
            Sided::Host(stage) => StageBox::HostSource(stage),
        },
        PreparedStage::Filter(prepared) => match prepared.build(backend, formats)? {
            Sided::Device(stage) => StageBox::Filter(stage),
            Sided::Host(stage) => StageBox::HostFilter(stage),
        },
        PreparedStage::Mask(prepared) => match prepared.build(backend, formats)? {
            Sided::Device(stage) => StageBox::Mask(stage),
            Sided::Host(stage) => StageBox::HostMask(stage),
        },
        PreparedStage::Provider(prepared) => match prepared.build(backend, formats)? {
            BuiltProvider::Device(stage) => StageBox::Provider(stage),
            BuiltProvider::Host(stage) => StageBox::HostProvider(stage),
            BuiltProvider::Meta(stage) => StageBox::MetaProvider(stage),
        },
        PreparedStage::Sink(prepared) => match prepared.build(backend, formats)? {
            Sided::Device(stage) => StageBox::Sink(stage),
            Sided::Host(stage) => StageBox::HostSink(stage),
        },
        PreparedStage::MetaSink(prepared) => {
            let stage = prepared.build(backend, formats)?;

            StageBox::MetaSink(stage)
        },
        PreparedStage::Crossing(crossing) => StageBox::Crossing(crossing),
    };

    Ok(stage)
}

/// Puts a build failure where the definition can be corrected.
///
/// A schema violation lands on the deepest recorded value it sits under, falling back to the
/// arguments table, and anything else lands on the stage. A provider's failure also notes what it
/// reads and the stages that demanded it.
fn report(graph: &CompiledGraph, node: &Node, error: BuildStageError, diagnostics: &mut Diagnostics) {
    let BuildStageError::InvalidArgs { source, .. } = &error else {
        let diagnostic = Diagnostic::error(
            DiagnosticCode::StageBuildFailed,
            stage_path(node.kind(), node.name()),
            node.span(),
            describe(&error),
        );
        let noted = note_demanders(graph, node, diagnostic);
        diagnostics.push(noted);

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
        let pointer = violation.path();
        let arg_spans = node.arg_spans();
        let narrowed = arg_spans.narrowest(pointer);
        let fallback = node.args_span();
        let span = narrowed.unwrap_or(fallback);

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidArgs,
            path,
            span,
            violation.message(),
        ));
    }
}

/// Renders an error together with whatever it was caused by.
pub(crate) fn describe(error: &BuildStageError) -> String {
    let mut message = error.to_string();
    let mut source = std::error::Error::source(error);

    while let Some(cause) = source {
        message.push_str(&format!(": {cause}"));
        source = cause.source();
    }

    message
}

/// Hands a freshly prepared stage what its probes found, when they have run.
fn settle_stage(node: &Node, stage: &mut PreparedStage) -> Result<(), StageError> {
    let Some(probed) = node.probed() else {
        return Ok(());
    };

    stage.settle(probed)
}

fn settle_failed(node: &Node, error: StageError) -> Diagnostic {
    let message = format!("`{}` could not use what its probes found: {error}", node.name());

    Diagnostic::error(DiagnosticCode::ProbeFailed, node.path(), node.span(), message)
}

/// Diagnostics raised while preparing a graph need no device, so these tests join the fast suite
/// rather than the device-gated module below.
#[cfg(test)]
mod diagnostics_tests {
    use av_toolbox_core::{PreparedStage, StageKind};
    use av_toolbox_gpu::ResolvedFormats;
    use serde_json::json;

    use crate::{
        Diagnostic,
        DiagnosticCode,
        Diagnostics,
        SolvedFormats,
        Span,
        Spanned,
        StageSpec,
        build,
        testing,
    };

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
    fn each_violation_lands_on_its_own_value_or_the_arguments() {
        let mut source = testing::spec(StageKind::Source, "input", "test/rated");
        source.args = Spanned::new(json!({ "numerator": "a", "denominator": "b" }), Span::new(40, 90));
        source.arg_spans.insert("/numerator", Span::new(52, 55));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let specs = vec![source, sink];
        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let mut diagnostics = Diagnostics::new();
        graph.prepare_stages(&registry, &mut diagnostics);
        let mut spans: Vec<(String, Span)> = diagnostics
            .iter()
            .map(|diagnostic| (diagnostic.path().to_string(), diagnostic.span()))
            .collect();
        spans.sort_by(|left, right| left.0.cmp(&right.0));

        assert_eq!(
            spans,
            vec![
                ("source.input.args.denominator".to_string(), Span::new(40, 90)),
                ("source.input.args.numerator".to_string(), Span::new(52, 55)),
            ]
        );
    }

    #[test]
    fn a_crossing_prepares_without_the_registry() {
        let mut graph = crate::graph::tests::file_blur_stdout();
        let blur = crate::graph::tests::node_named(&graph, "blur");
        let crossing = graph.push_crossing(blur, av_toolbox_core::Crossing::Unload);
        graph.finish_edit();

        let mut diagnostics = Diagnostics::new();
        let prepared = graph.prepare_stages(&crate::testing::registry(), &mut diagnostics);

        assert!(!diagnostics.has_errors());
        assert!(matches!(
            prepared[crossing.index()],
            Some(PreparedStage::Crossing(av_toolbox_core::Crossing::Unload))
        ));
    }

    #[test]
    fn a_graph_of_stages_holding_nothing_holds_nothing() {
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();
        let per_node = (0..graph.len())
            .map(|_| ResolvedFormats::new(Vec::new(), None))
            .collect();
        let formats = SolvedFormats::new(per_node);

        let held = graph.held_frames(&registry, &formats).expect("should prepare");

        assert_eq!(held, vec![0, 0, 0]);
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

    use av_toolbox_core::testing::{FixtureChunk, fixture_chunk};
    use av_toolbox_core::{Inputs, Residency, StageKind, run_meta_provider, run_meta_sink};
    use av_toolbox_gpu::{
        AnyItem,
        ColourInfo,
        FrameContext,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        HostStreamItem,
        SampleFormat,
        StreamFormat,
        StreamItem,
        Subsampling,
        test_device,
        testing_context,
    };
    use serde_json::json;

    use super::diagnostics_tests::pipeline;
    use super::*;
    use crate::solve::solve;
    use crate::{DiagnosticCode, Spanned, StageSpec, build, testing};

    fn backend() -> Backend {
        Backend::open_default(test_device()).expect("should open")
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
            frame_rate: None,
            field_order: None,
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

    /// A source, a filter widened to four workers, and a sink narrowed back to one.
    ///
    /// The widened filter's own granularity and width mismatch put it in a region of its own,
    /// between the source's region and the sink's, so it lands alone in the middle.
    fn region_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "example.mkv" }));

        let mut wide = testing::spec(StageKind::Filter, "wide", "test/temporal");
        wide.inputs = vec![testing::positional("input")];
        wide.args = Spanned::unknown(json!({ "radius": 0 }));
        wide.workers = Some(4);

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("wide")];

        vec![source, wide, sink]
    }

    /// Builds, plans and solves [region_pipeline], and opens a real backend to build its stages
    /// against.
    fn region_fixture() -> (CompiledGraph, PipelineRegistry, SolvedFormats, Backend) {
        let registry = testing::registry();
        let mut graph = build(&registry, &region_pipeline())
            .expect("should build")
            .into_graph();
        let backend = backend();
        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);
        let solved = formats(&graph, &registry, &backend);

        (graph, registry, solved, backend)
    }

    #[test]
    fn a_region_builds_only_its_own_nodes() {
        let (graph, registry, formats, backend) = region_fixture();
        let plan = graph.execution_plan().expect("planned");
        let middle = plan.region(plan.node(NodeId::new(1)).region());

        let built = graph
            .instantiate_region(&registry, &backend, &formats, middle)
            .expect("should build");

        assert_eq!(built.len(), middle.nodes().len());
        let ids: Vec<NodeId> = built.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.as_slice(), middle.nodes());
    }

    /// The point of `instantiate_region`. A worker calls it once per copy it needs, so two calls
    /// must each build their own stages rather than one call's stages being handed back twice.
    #[test]
    fn building_a_region_twice_gives_two_independent_sets() {
        let (graph, registry, formats, backend) = region_fixture();
        let plan = graph.execution_plan().expect("planned");
        let region = plan.region(plan.node(NodeId::new(1)).region());

        testing::TEMPORAL_BUILD_CALLS.store(0, Ordering::SeqCst);

        let first = graph
            .instantiate_region(&registry, &backend, &formats, region)
            .expect("should build");
        let second = graph
            .instantiate_region(&registry, &backend, &formats, region)
            .expect("should build");

        assert_eq!(first.len(), second.len());
        assert_eq!(first[0].0, NodeId::new(1));
        assert_eq!(second[0].0, NodeId::new(1));

        // Each call built its own copy of the region's one node. A cached or shared box would
        // leave this at one no matter how many times the region was instantiated.
        assert_eq!(testing::TEMPORAL_BUILD_CALLS.load(Ordering::SeqCst), 2);
    }

    /// Runs one frame through the provider named `name`, as the region would.
    fn provide(stages: &mut [StageBox], graph: &CompiledGraph, name: &str, item: StreamItem) -> StreamItem {
        let node = graph
            .nodes()
            .iter()
            .find(|node| node.name() == name)
            .unwrap_or_else(|| panic!("the graph has no node named {name}"));
        let StageBox::Provider(provider) = &mut stages[node.id().index()] else {
            panic!("{name} is not a device provider");
        };

        let mut context = testing_context();
        let inputs = Inputs::new(vec![Some(item)]);
        let mut emitted = provider.process(&mut context, inputs).expect("should provide");

        emitted.remove(0)
    }

    #[test]
    fn a_provider_reads_its_demands_through_provisions() {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "example.mkv" }));
        let mut wants = testing::spec(StageKind::Filter, "wants", "test/demanding");
        wants.inputs = vec![testing::positional("input")];
        wants.args = Spanned::unknown(json!({ "every": 4, "marks": true }));
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("wants")];

        let registry = testing::registry();
        let mut graph = build(&registry, &[source, wants, sink])
            .expect("should build")
            .into_graph();
        crate::satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");
        let backend = backend();
        let solved = formats(&graph, &registry, &backend);

        let mut stages = graph
            .instantiate(&registry, &backend, &solved)
            .expect("should instantiate");

        let context = testing_context();
        let format = frames(64).as_frames().copied().expect("frames");
        let meta = fixture_chunk(FrameMeta::at(0), 7);
        let frame = context.alloc_frame(&format, meta);

        let broken = provide(
            &mut stages,
            &graph,
            "fixture-breaks(input)",
            StreamItem::Frame(frame),
        );
        let marked = provide(&mut stages, &graph, "fixture-marks(input)", broken);

        let marking = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "fixture-marks(input)")
            .expect("a marks provider");
        let wanting = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "wants")
            .expect("the demanding stage");
        let breaks_layer = marking.demands()[0].layer;
        let marks_layer = wanting.demands()[0].layer;

        let provided = &marked.meta().provided;
        assert_eq!(provided.get::<FixtureChunk>(breaks_layer), Some(&FixtureChunk(7)));
        assert_eq!(provided.get::<FixtureChunk>(marks_layer), Some(&FixtureChunk(7)));
    }

    fn device_item(context: &FrameContext, index: u64) -> AnyItem {
        let format = frames(64).as_frames().copied().expect("frames");
        let frame = context.alloc_frame(&format, FrameMeta::at(index));

        AnyItem::from(StreamItem::Frame(frame))
    }

    fn host_item(context: &FrameContext, index: u64) -> AnyItem {
        let format = frames(64).as_frames().copied().expect("frames");
        let frame = context.alloc_host_frame(&format, FrameMeta::at(index)).freeze();

        AnyItem::Host(HostStreamItem::from(frame))
    }

    /// Runs three frames from `item_of` through the tags provider placed behind `source` and
    /// into a sink that fails on any frame not tagged `Tag(true)`.
    fn tags_every_frame(source: StageSpec, side: Residency, item_of: fn(&FrameContext, u64) -> AnyItem) {
        let mut reader = testing::spec(StageKind::Sink, "read", "test/tag-reading");
        reader.inputs = vec![testing::positional("input")];

        let registry = testing::registry();
        let mut graph = build(&registry, &[source, reader])
            .expect("should build")
            .into_graph();
        crate::satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");
        let backend = backend();
        let mut solved = formats(&graph, &registry, &backend);
        crate::place_transfers(&mut graph, &registry, &mut solved).expect("should place");

        let mut stages = graph
            .instantiate(&registry, &backend, &solved)
            .expect("should instantiate");

        let tagging = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "tags(input)")
            .expect("a tags provider");
        let reading = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "read")
            .expect("the reading sink");
        assert_eq!(tagging.residency(), Some(side));

        let context = testing_context();
        let mut run_context = testing_context();
        for index in 0..3 {
            let item = item_of(&context, index);
            let StageBox::MetaProvider(provider) = &mut stages[tagging.id().index()] else {
                panic!("tags(input) is not a metadata provider");
            };
            let tagged = run_meta_provider(provider.as_mut(), &mut run_context, item).expect("should tag");

            let StageBox::MetaSink(sink) = &mut stages[reading.id().index()] else {
                panic!("read is not a metadata sink");
            };
            let slots = tagged.into_iter().map(Some).collect();
            run_meta_sink(sink.as_mut(), &mut run_context, slots).expect("every frame reads Tag(true)");
        }
    }

    #[test]
    fn a_meta_provider_tags_frames_on_either_side() {
        let host_source = testing::spec(StageKind::Source, "input", "test/host-file");
        tags_every_frame(host_source, Residency::Host, host_item);

        let mut device_source = testing::spec(StageKind::Source, "input", "test/file");
        device_source.args = Spanned::unknown(json!({ "path": "example.mkv" }));
        tags_every_frame(device_source, Residency::Device, device_item);
    }

    #[test]
    fn a_region_that_fails_to_build_reports_its_diagnostics() {
        let mut specs = pipeline();
        specs[1].stage_type = Spanned::unknown(String::from("test/failing"));
        specs[1].args = Spanned::unknown(json!({}));

        let registry = testing::registry();
        let mut graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let solved = formats(&graph, &registry, &backend);
        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        let plan = graph.execution_plan().expect("planned");
        let region = plan.region(plan.node(NodeId::new(1)).region());

        let diagnostics = graph
            .instantiate_region(&registry, &backend, &solved, region)
            .expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::StageBuildFailed);
        assert!(reported[0].message().contains("this stage always fails"));
    }
}
