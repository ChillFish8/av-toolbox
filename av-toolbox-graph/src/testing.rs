use av_toolbox_core::{
    BoundInputs,
    Filter,
    InputParam,
    InputSignature,
    Mask,
    PipelineRegistry,
    Sink,
    Source,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
    Transform,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::{InputRef, Span, Spanned, StageSpec};

macro_rules! fixture_stage {
    (
        $definition:ident,
        $instance:ident,
        $kind_trait:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}
        impl $kind_trait for $instance {}

        struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;
            type Instance = $instance;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = $signature;
                SIGNATURE
            }

            fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
                Ok($instance)
            }
        }
    };
}

/// The fixture arguments exist to give the registry a schema to validate against, so nothing ever
/// reads the fields back out.
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct FileArgs {
    pub(crate) path: String,
}

#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct BlurArgs {
    pub(crate) sigma: f64,
}

#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct SplitArgs {
    pub(crate) seconds: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NoArgs {}

fixture_stage!(File, FileStage, Source, "file", FileArgs, &InputSignature::EMPTY);
fixture_stage!(
    Blur,
    BlurStage,
    Filter,
    "blur",
    BlurArgs,
    &InputSignature::new(&[InputParam::required("reference")])
);
fixture_stage!(
    Merge,
    MergeStage,
    Mask,
    "merge",
    NoArgs,
    &InputSignature::with_variadic(&[InputParam::required("base")], "extra")
);
fixture_stage!(
    Split,
    SplitStage,
    Transform,
    "split",
    SplitArgs,
    &InputSignature::new(&[InputParam::required("input")])
);
fixture_stage!(
    Stdout,
    StdoutStage,
    Sink,
    "stdout",
    NoArgs,
    &InputSignature::new(&[InputParam::required("input")])
);

struct FailingStage;

impl StageInstance for FailingStage {}
impl Filter for FailingStage {}

struct Failing;

impl StageDefinition for Failing {
    type Args = NoArgs;
    type Instance = FailingStage;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "failing"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }

    fn build(_args: Self::Args, _inputs: BoundInputs) -> Result<Self::Instance, StageError> {
        Err(StageError::msg("this stage always fails"))
    }
}

pub(crate) fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();

    registry.register_source::<File>().expect("should register");
    registry.register_filter::<Blur>().expect("should register");
    registry.register_filter::<Failing>().expect("should register");
    registry.register_mask::<Merge>().expect("should register");
    registry.register_transform::<Split>().expect("should register");
    registry.register_sink::<Stdout>().expect("should register");

    registry
}

/// Builds a spec with no inputs, empty arguments and unknown spans.
///
/// Every field is public, so a test that cares about a span or an argument sets it directly.
#[expect(dead_code, reason = "used by later tasks' tests")]
pub(crate) fn spec(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        span: Span::unknown(),
    }
}

#[expect(dead_code, reason = "used by later tasks' tests")]
pub(crate) fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

#[expect(dead_code, reason = "used by later tasks' tests")]
pub(crate) fn named(name: &str, target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: Some(name.to_string()),
        target: target.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_test_registry_holds_every_fixture_stage() {
        let registry = registry();
        let identifiers: Vec<&str> = registry.stages().map(|stage| stage.full_identifier()).collect();

        assert_eq!(identifiers.len(), 6);
        assert!(identifiers.contains(&"test/blur"));
        assert!(identifiers.contains(&"test/failing"));
    }
}
