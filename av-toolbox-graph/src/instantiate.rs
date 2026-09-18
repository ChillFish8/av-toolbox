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
    /// This is where arguments are checked, so a schema violation and a failure raised by the
    /// stage itself are reported together. Every stage is attempted, so one run names every stage
    /// that could not be built.
    pub fn instantiate(&self, registry: &PipelineRegistry) -> Result<Vec<StageBox>, Diagnostics> {
        let mut diagnostics = Diagnostics::new();
        let mut built: Vec<Option<StageBox>> = (0..self.len()).map(|_| None).collect();

        for id in self.order() {
            let node = self.node(*id);
            let arguments = arguments_for(node);

            match build_stage(registry, node, &arguments) {
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
fn arguments_for(node: &Node) -> Vec<InputArgument> {
    node.inputs()
        .iter()
        .map(|edge| {
            let handle = av_toolbox_core::StageHandle::new(edge.source().index() as u64);

            match edge.name() {
                Some(name) => InputArgument::named(name, handle),
                None => InputArgument::positional(handle),
            }
        })
        .collect()
}

fn build_stage(
    registry: &PipelineRegistry,
    node: &Node,
    arguments: &[InputArgument],
) -> Result<StageBox, BuildStageError> {
    let identifier = node.identifier();
    let args = node.args();

    let stage = match node.kind() {
        StageKind::Source => StageBox::Source(registry.build_source(identifier, args, arguments)?),
        StageKind::Filter => StageBox::Filter(registry.build_filter(identifier, args, arguments)?),
        StageKind::Mask => StageBox::Mask(registry.build_mask(identifier, args, arguments)?),
        StageKind::Transform => StageBox::Transform(registry.build_transform(identifier, args, arguments)?),
        StageKind::Sink => StageBox::Sink(registry.build_sink(identifier, args, arguments)?),
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

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
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

    #[test]
    fn every_stage_is_built_and_indexed_by_node() {
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();

        let stages = graph.instantiate(&registry).expect("should instantiate");

        assert_eq!(stages.len(), 3);
        assert!(matches!(stages[0], StageBox::Source(_)));
        assert!(matches!(stages[1], StageBox::Filter(_)));
        assert!(matches!(stages[2], StageBox::Sink(_)));
    }

    #[test]
    fn a_schema_violation_lands_on_the_arguments() {
        let mut specs = pipeline();
        specs[1].args = Spanned::new(json!({ "sigma": "wide" }), Span::new(40, 60));

        let registry = testing::registry();
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let diagnostics = graph.instantiate(&registry).expect_err("should fail");
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

        let diagnostics = graph.instantiate(&registry).expect_err("should fail");
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

        let diagnostics = graph.instantiate(&registry).expect_err("should fail");

        assert_eq!(diagnostics.iter().count(), 2);
    }
}
