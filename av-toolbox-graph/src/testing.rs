use std::sync::atomic::{AtomicU32, Ordering};

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Mask,
    MaskDefinition,
    PipelineRegistry,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
    StreamType,
    Transform,
    TransformDefinition,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourInfo,
    FormatSet,
    FrameFormat,
    FrameLayout,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::{InputRef, Span, Spanned, StageSpec};

/// A fixed frame format used as the starting point for every fixture source.
fn fixture_format() -> StreamFormat {
    StreamFormat::Frames(FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    })
}

/// Writes a source fixture, which has nothing to negotiate against since it takes no inputs.
macro_rules! fixture_source {
    (
        $definition:ident,
        $instance:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}
        impl Source for $instance {}

        struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;

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
        }

        impl SourceDefinition for $definition {
            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
            ) -> Result<OutputFormats, StageError> {
                let format = fixture_format();

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: FormatSet::of([format.sample()]),
                })
            }

            fn build<R: Runtime>(
                _context: &BuildContext<R>,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Source>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes a filter, mask or transform fixture: something with both an input and an output side.
///
/// It accepts whatever it is offered and produces what it was given, which is the least
/// surprising behaviour for a test stage.
macro_rules! fixture_stage {
    (
        $definition:ident,
        $instance:ident,
        $kind_trait:ident,
        $definition_trait:ident,
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
        }

        impl $definition_trait for $definition {
            fn accepts(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                offered: &[StreamFormat],
            ) -> Result<Vec<FormatSet>, StageError> {
                Ok(offered
                    .iter()
                    .map(|format| FormatSet::of([format.sample()]))
                    .collect())
            }

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[StreamFormat],
            ) -> Result<OutputFormats, StageError> {
                let format = chosen
                    .first()
                    .copied()
                    .ok_or_else(|| StageError::msg("a stage with inputs was given none"))?;

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: FormatSet::of([format.sample()]),
                })
            }

            fn build<R: Runtime>(
                _context: &BuildContext<R>,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn $kind_trait>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// Writes a sink fixture, which has nothing to produce since it ends the stream.
macro_rules! fixture_sink {
    (
        $definition:ident,
        $instance:ident,
        $identifier:literal,
        $args:ty,
        $signature:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}
        impl Sink for $instance {}

        struct $definition;

        impl StageDefinition for $definition {
            type Args = $args;

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
        }

        impl SinkDefinition for $definition {
            fn accepts(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                offered: &[StreamFormat],
            ) -> Result<Vec<FormatSet>, StageError> {
                Ok(offered
                    .iter()
                    .map(|format| FormatSet::of([format.sample()]))
                    .collect())
            }

            fn build<R: Runtime>(
                _context: &BuildContext<R>,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Sink>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

/// The fixture arguments exist to give the registry a schema to validate against, so nothing ever
/// reads the fields back out.
#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct FileArgs {
    pub(crate) path: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct BlurArgs {
    pub(crate) sigma: f64,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[expect(dead_code, reason = "the fixtures only deserialize their arguments")]
pub(crate) struct SplitArgs {
    pub(crate) seconds: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct NoArgs {}

fixture_source!(File, FileStage, "file", FileArgs, &InputSignature::EMPTY);
fixture_stage!(
    Blur,
    BlurStage,
    Filter,
    FilterDefinition,
    "blur",
    BlurArgs,
    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)])
);
fixture_stage!(
    Merge,
    MergeStage,
    Mask,
    MaskDefinition,
    "merge",
    NoArgs,
    &InputSignature::with_variadic_of(
        &[InputParam::required("base").accepting(StreamType::Frames)],
        StreamType::Frames,
    )
);
fixture_stage!(
    Split,
    SplitStage,
    Transform,
    TransformDefinition,
    "split",
    SplitArgs,
    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)])
);
fixture_sink!(
    Stdout,
    StdoutStage,
    "stdout",
    NoArgs,
    &InputSignature::new(&[InputParam::required("input")])
);
fixture_stage!(
    Deband,
    DebandStage,
    Filter,
    FilterDefinition,
    "deband",
    NoArgs,
    &InputSignature::new(&[
        InputParam::required("reference").accepting(StreamType::Frames),
        InputParam::optional("overlay").accepting(StreamType::Mask),
    ])
);

struct Failing;

impl StageDefinition for Failing {
    type Args = NoArgs;

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
}

impl FilterDefinition for Failing {
    fn accepts(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        offered: &[StreamFormat],
    ) -> Result<Vec<FormatSet>, StageError> {
        Ok(offered
            .iter()
            .map(|format| FormatSet::of([format.sample()]))
            .collect())
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[StreamFormat],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen
            .first()
            .copied()
            .ok_or_else(|| StageError::msg("a stage with inputs was given none"))?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Err(StageError::msg("this stage always fails"))
    }
}

/// The width [Recording] was built with, read from the first of its resolved inputs.
///
/// Stays `0` when the stage never built, and when that input was not a frame stream.
pub(crate) static RECORDED_INPUT_WIDTH: AtomicU32 = AtomicU32::new(0);

struct RecordingStage;

impl StageInstance for RecordingStage {}
impl Filter for RecordingStage {}

struct Recording;

impl StageDefinition for Recording {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "recording"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for Recording {
    fn accepts(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        offered: &[StreamFormat],
    ) -> Result<Vec<FormatSet>, StageError> {
        Ok(offered
            .iter()
            .map(|format| FormatSet::of([format.sample()]))
            .collect())
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[StreamFormat],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen
            .first()
            .copied()
            .ok_or_else(|| StageError::msg("a stage with inputs was given none"))?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let width = match formats.input(0) {
            Some(StreamFormat::Frames(format)) => format.width,
            _ => 0,
        };
        RECORDED_INPUT_WIDTH.store(width, Ordering::SeqCst);

        Ok(Box::new(RecordingStage))
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
    registry.register_filter::<Deband>().expect("should register");
    registry.register_filter::<Recording>().expect("should register");

    registry
}

/// Builds a spec with no inputs, empty arguments and unknown spans.
///
/// Every field is public, so a test that cares about a span or an argument sets it directly.
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

pub(crate) fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

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

        assert_eq!(identifiers.len(), 8);
        assert!(identifiers.contains(&"test/blur"));
        assert!(identifiers.contains(&"test/failing"));
        assert!(identifiers.contains(&"test/recording"));
    }
}
