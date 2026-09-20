use std::fmt;

use av_toolbox_core::{
    ArgsError,
    BuildStageError,
    Filter,
    InputArgument,
    Mask,
    PipelineRegistry,
    Sink,
    Source,
    StageKind,
    Transform,
};
use av_toolbox_gpu::{Backend, ResolvedFormats};

use crate::diagnostic::{args_path, stage_path};
use crate::graph::{CompiledGraph, Node};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

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
    /// Builds every stage through the registry, in dependency order.
    ///
    /// `formats` holds one entry per node, in the same order as the graph's nodes, which is what a
    /// solved graph produces.
    ///
    /// This is where arguments are checked, so a schema violation and a failure raised by the
    /// stage itself are reported together. Every stage is attempted, so one run names every stage
    /// that could not be built.
    pub fn instantiate(
        &self,
        registry: &PipelineRegistry,
        backend: &Backend,
        formats: &[ResolvedFormats],
    ) -> Result<Vec<StageBox>, Diagnostics> {
        debug_assert_eq!(
            formats.len(),
            self.len(),
            "a solved graph has one set of formats per node",
        );

        let mut diagnostics = Diagnostics::new();
        let mut built: Vec<Option<StageBox>> = (0..self.len()).map(|_| None).collect();

        for id in self.order() {
            let node = self.node(*id);
            let arguments = arguments_for(self, node);
            let node_formats = &formats[id.index()];

            match build_stage(registry, node, &arguments, backend, node_formats) {
                Ok(stage) => built[id.index()] = Some(stage),
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
    registry: &PipelineRegistry,
    node: &Node,
    arguments: &[InputArgument],
    backend: &Backend,
    formats: &ResolvedFormats,
) -> Result<StageBox, BuildStageError> {
    let identifier = node.identifier();
    let args = node.args();

    let stage = match node.kind() {
        StageKind::Source => {
            let source = registry.build_source(identifier, backend, args, arguments, formats)?;
            StageBox::Source(source)
        },
        StageKind::Filter => {
            let filter = registry.build_filter(identifier, backend, args, arguments, formats)?;
            StageBox::Filter(filter)
        },
        StageKind::Mask => {
            let mask = registry.build_mask(identifier, backend, args, arguments, formats)?;
            StageBox::Mask(mask)
        },
        StageKind::Transform => {
            let transform = registry.build_transform(identifier, backend, args, arguments, formats)?;
            StageBox::Transform(transform)
        },
        StageKind::Sink => {
            let sink = registry.build_sink(identifier, backend, args, arguments, formats)?;
            StageBox::Sink(sink)
        },
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

// Every test here calls `instantiate`, which now needs a real backend, so the whole module needs
// a device. There is no non-device test to keep running separately.
#[cfg(test)]
#[cfg(all(feature = "gpu-tests", any(feature = "vulkan", feature = "metal")))]
mod tests {
    use std::sync::atomic::Ordering;

    use av_toolbox_core::StageKind;
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

    use super::*;
    use crate::{DiagnosticCode, Span, Spanned, StageSpec, build, testing};

    fn pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "example.mkv" }));

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 0.8 }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, sink]
    }

    fn backend() -> Backend {
        Backend::open_default(Device::Default).expect("should open")
    }

    /// One set of formats per node, each carrying a width that identifies its node.
    fn formats(graph: &CompiledGraph) -> Vec<ResolvedFormats> {
        (0..graph.len())
            .map(|index| {
                let format = frames(100 + index as u32);
                ResolvedFormats::new(vec![format], Some(format))
            })
            .collect()
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
        let formats = formats(&graph);

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
        let formats = formats(&graph);

        testing::RECORDED_INPUT_WIDTH.store(0, Ordering::SeqCst);

        graph
            .instantiate(&registry, &backend, &formats)
            .expect("should instantiate");

        assert_eq!(testing::RECORDED_INPUT_WIDTH.load(Ordering::SeqCst), 101);
    }

    #[test]
    fn a_schema_violation_lands_on_the_arguments() {
        let mut specs = pipeline();
        specs[1].args = Spanned::new(json!({ "sigma": "wide" }), Span::new(40, 60));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = formats(&graph);

        let diagnostics = graph
            .instantiate(&registry, &backend, &formats)
            .expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::InvalidArgs);
        assert!(reported[0].path().starts_with("filter.blur.args"));
        assert_eq!(reported[0].span(), Span::new(40, 60));
    }

    #[test]
    fn a_stage_that_fails_to_build_is_reported_with_its_own_message() {
        let mut specs = pipeline();
        specs[1].stage_type = Spanned::unknown(String::from("test/failing"));
        specs[1].args = Spanned::unknown(json!({}));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = formats(&graph);

        let diagnostics = graph
            .instantiate(&registry, &backend, &formats)
            .expect_err("should fail");
        let reported: Vec<&Diagnostic> = diagnostics.iter().collect();

        assert_eq!(reported[0].code(), DiagnosticCode::StageBuildFailed);
        assert!(reported[0].message().contains("this stage always fails"));
    }

    #[test]
    fn every_failing_stage_is_reported_not_just_the_first() {
        let mut specs = pipeline();
        specs[0].args = Spanned::unknown(json!({}));
        specs[1].args = Spanned::unknown(json!({ "sigma": "wide" }));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();
        let backend = backend();
        let formats = formats(&graph);

        let diagnostics = graph
            .instantiate(&registry, &backend, &formats)
            .expect_err("should fail");

        assert_eq!(diagnostics.iter().count(), 2);
    }
}
