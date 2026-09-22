use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
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
    Depth,
    ElementType,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Mask as GpuMask,
    MaskFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamItem,
    StreamTemplate,
    Subsampling,
    testing_context,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::{InputRef, Span, Spanned, StageSpec};

/// The frame format every fixture works in, so they all name the same dimensions.
fn frame_format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    }
}

/// The format of the first bound input, since every fixture passes its first input through.
///
/// A signature with an optional parameter can leave a hole before the input a fixture wants, so it
/// takes the first one that is actually there rather than the first slot.
fn first_bound(chosen: &[Option<StreamFormat>]) -> Result<StreamFormat, StageError> {
    let format = chosen.iter().flatten().next().copied();

    format.ok_or_else(|| StageError::msg("a stage with inputs was given none"))
}

/// A fixed frame format used as the starting point for every fixture source.
fn fixture_format() -> StreamFormat {
    StreamFormat::Frames(frame_format())
}

/// The mask format every fixture mask allocates into, so a select fixture needs nothing from its
/// inputs to produce one.
///
/// The sample is arbitrary and does not have to agree with whatever `mask_template` negotiated,
/// because nothing reads a fixture's allocated output.
fn fixture_mask_format() -> MaskFormat {
    MaskFormat {
        width: 64,
        height: 64,
        sample: SampleFormat::U8,
    }
}

fn frames_template(chosen: StreamFormat) -> StreamTemplate {
    StreamTemplate::from_format(&chosen)
}

/// A mask of the fixture dimensions, carrying whatever sample was chosen.
///
/// A mask stage emits a mask, so it cannot pass a frames template straight through.
fn mask_template(chosen: StreamFormat) -> StreamTemplate {
    StreamTemplate::Mask(MaskFormat {
        width: 64,
        height: 64,
        sample: chosen.sample(),
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

        impl Source for $instance {
            fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
                Ok(None)
            }
        }

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

/// Writes the kind-specific instance method for a filter, mask or transform fixture.
///
/// Each does the minimum its trait requires: a filter and a transform pass their first bound
/// input through, and a mask allocates a fresh one in the fixture format.
macro_rules! fixture_kind_method {
    (Filter, $instance:ident) => {
        impl Filter for $instance {
            fn process(
                &mut self,
                _context: &mut FrameContext,
                mut inputs: Inputs,
            ) -> Result<Frame, StageError> {
                inputs
                    .take_frame(0)
                    .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
            }
        }
    };
    (Mask, $instance:ident) => {
        impl Mask for $instance {
            fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
                Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
            }
        }
    };
    (Transform, $instance:ident) => {
        impl Transform for $instance {
            fn process(
                &mut self,
                _context: &mut FrameContext,
                mut inputs: Inputs,
            ) -> Result<Vec<StreamItem>, StageError> {
                Ok(inputs.take(0).into_iter().collect())
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
        $signature:expr,
        $template:expr
    ) => {
        struct $instance;

        impl StageInstance for $instance {}
        fixture_kind_method!($kind_trait, $instance);

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
            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
            }

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                let format = first_bound(chosen)?;

                Ok(OutputFormats {
                    template: $template(format),
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

        impl Sink for $instance {
            fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
                Ok(())
            }
        }

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
            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok(FormatSet::of([offered.sample()]))
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
    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]),
    frames_template
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
    ),
    mask_template
);
fixture_stage!(
    Split,
    SplitStage,
    Transform,
    TransformDefinition,
    "split",
    SplitArgs,
    &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]),
    frames_template
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
    ]),
    frames_template
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
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

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

impl Filter for RecordingStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

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
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

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

/// Writes a source fixture offering more than one sample, so an edge from it has a choice to make.
macro_rules! opinionated_source {
    ($definition:ident, $instance:ident, $identifier:literal, $samples:expr) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Source for $instance {
            fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
                Ok(None)
            }
        }

        struct $definition;

        impl StageDefinition for $definition {
            type Args = NoArgs;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
                SIGNATURE
            }
        }

        impl SourceDefinition for $definition {
            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
            ) -> Result<OutputFormats, StageError> {
                Ok(OutputFormats {
                    template: StreamTemplate::Frames(frame_format()),
                    samples: $samples,
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

/// Writes a filter fixture that takes only certain samples and emits the same ones.
macro_rules! opinionated_filter {
    ($definition:ident, $instance:ident, $identifier:literal, $samples:expr) => {
        struct $instance;

        impl StageInstance for $instance {}

        impl Filter for $instance {
            fn process(
                &mut self,
                _context: &mut FrameContext,
                mut inputs: Inputs,
            ) -> Result<Frame, StageError> {
                inputs
                    .take_frame(0)
                    .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
            }
        }

        struct $definition;

        impl StageDefinition for $definition {
            type Args = NoArgs;

            fn package() -> &'static str {
                "test"
            }

            fn identifier() -> &'static str {
                $identifier
            }

            fn inputs() -> &'static InputSignature {
                const SIGNATURE: &InputSignature =
                    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
                SIGNATURE
            }
        }

        impl FilterDefinition for $definition {
            fn accepts_input(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                _index: usize,
                _offered: StreamFormat,
                _inputs: &[Offered],
            ) -> Result<FormatSet, StageError> {
                Ok($samples)
            }

            fn produces(
                _capabilities: &Capabilities,
                _args: &Self::Args,
                chosen: &[Option<StreamFormat>],
            ) -> Result<OutputFormats, StageError> {
                let format = first_bound(chosen)?;

                Ok(OutputFormats {
                    template: StreamTemplate::from_format(&format),
                    samples: $samples,
                })
            }

            fn build<R: Runtime>(
                _context: &BuildContext<R>,
                _args: Self::Args,
                _inputs: BoundInputs,
                _formats: &ResolvedFormats,
            ) -> Result<Box<dyn Filter>, StageError> {
                Ok(Box::new($instance))
            }
        }
    };
}

opinionated_source!(
    Wide,
    WideStage,
    "wide",
    FormatSet::of([SampleFormat::U8, SampleFormat::F32])
);
opinionated_source!(
    Deep,
    DeepStage,
    "deep",
    FormatSet::of([SampleFormat::U16 { depth: Depth::TEN }, SampleFormat::F32])
);
// FloatFirst would rather emit the wider of its two samples, so its preference and its narrowest
// sample are not the same.
opinionated_source!(
    FloatFirst,
    FloatFirstStage,
    "floatfirst",
    FormatSet::of([SampleFormat::F32, SampleFormat::U8])
);
opinionated_filter!(
    Precise,
    PreciseStage,
    "precise",
    FormatSet::of([SampleFormat::F32])
);
opinionated_filter!(
    Either,
    EitherStage,
    "either",
    FormatSet::of([SampleFormat::U8, SampleFormat::F32])
);
opinionated_filter!(Narrow, NarrowStage, "narrow", FormatSet::of([SampleFormat::U8]));

struct Prefers16Stage;

impl StageInstance for Prefers16Stage {}

impl Filter for Prefers16Stage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

/// A filter that would rather work in a ten-bit integer, and asks the device before offering it.
struct Prefers16;

impl StageDefinition for Prefers16 {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "prefers16"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Prefers16 {
    fn accepts_input(
        capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        _offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let mut formats = FormatSet::new();
        if capabilities.can_compute(ElementType::U16) {
            formats.push(SampleFormat::U16 { depth: Depth::TEN });
        }
        formats.push(SampleFormat::F32);

        Ok(formats)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

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
        Ok(Box::new(Prefers16Stage))
    }
}

/// The signature both coupled masks take, one frame stream per input.
const COUPLED_INPUTS: &InputSignature = &InputSignature::new(&[
    InputParam::required("base").accepting(StreamType::Frames),
    InputParam::required("overlay").accepting(StreamType::Frames),
]);

struct MatchedStage;

impl StageInstance for MatchedStage {}

impl Mask for MatchedStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A mask whose two inputs have to carry the same sample, because its kernel reads them together.
struct Matched;

impl StageDefinition for Matched {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "matched"
    }

    fn inputs() -> &'static InputSignature {
        COUPLED_INPUTS
    }
}

impl MaskDefinition for Matched {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let sibling = inputs
            .iter()
            .enumerate()
            .find(|(slot, _)| *slot != index)
            .and_then(|(_, other)| other.format());

        match sibling {
            Some(other) => Ok(FormatSet::of([other.sample()])),
            None => Ok(FormatSet::of([offered.sample()])),
        }
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: mask_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(MatchedStage))
    }
}

struct AnchoredStage;

impl StageInstance for AnchoredStage {}

impl Mask for AnchoredStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A mask whose base has to carry its overlay's sample, while the overlay itself takes anything.
struct Anchored;

impl StageDefinition for Anchored {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "anchored"
    }

    fn inputs() -> &'static InputSignature {
        COUPLED_INPUTS
    }
}

impl MaskDefinition for Anchored {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        if index != 0 {
            return Ok(FormatSet::of([offered.sample()]));
        }

        let overlay = inputs.get(1).and_then(|input| input.format());

        match overlay {
            Some(format) => Ok(FormatSet::of([format.sample()])),
            None => Ok(FormatSet::of([offered.sample()])),
        }
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: mask_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(AnchoredStage))
    }
}

struct MislabelledStage;

impl StageInstance for MislabelledStage {}

impl Mask for MislabelledStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A mask that produces frames, which is not what a mask emits.
struct Mislabelled;

impl StageDefinition for Mislabelled {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "mislabelled"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("base").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Mislabelled {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(MislabelledStage))
    }
}

/// What [Sparse] was told about its inputs, recorded so a test can compare the three places a
/// stage hears about them.
///
/// The slot it was asked about and how many it was offered, then how many formats it was given to
/// produce from and which of those slots were filled, as a bit per slot.
pub(crate) static SPARSE_ASKED_SLOT: AtomicUsize = AtomicUsize::new(usize::MAX);
pub(crate) static SPARSE_OFFERED_SLOTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static SPARSE_CHOSEN_SLOTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static SPARSE_FILLED_SLOTS: AtomicUsize = AtomicUsize::new(0);

struct SparseStage;

impl StageInstance for SparseStage {}

impl Filter for SparseStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))
    }
}

/// A filter whose first input is optional, so a pipeline can leave a hole before a bound slot.
struct Sparse;

impl StageDefinition for Sparse {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "sparse"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::optional("first").accepting(StreamType::Frames),
            InputParam::required("second").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Sparse {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        SPARSE_ASKED_SLOT.store(index, Ordering::SeqCst);
        SPARSE_OFFERED_SLOTS.store(inputs.len(), Ordering::SeqCst);

        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let filled: usize = chosen
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(index, _)| 1 << index)
            .sum();

        SPARSE_CHOSEN_SLOTS.store(chosen.len(), Ordering::SeqCst);
        SPARSE_FILLED_SLOTS.store(filled, Ordering::SeqCst);

        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(SparseStage))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct TemporalArgs {
    #[serde(default)]
    radius: u32,
}

struct TemporalFilter;

impl StageInstance for TemporalFilter {}

impl Filter for TemporalFilter {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

/// A filter whose granularity comes from its arguments, unlike every other fixture.
struct Temporal;

impl StageDefinition for Temporal {
    type Args = TemporalArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "temporal"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.radius {
            0 => Parallelism::Frame,
            _ => Parallelism::Chunk,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for Temporal {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(TemporalFilter))
    }
}

struct TemporalMergeStage;

impl StageInstance for TemporalMergeStage {}

impl Mask for TemporalMergeStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A two input mask whose granularity comes from its arguments, so it can widen where
/// `test/merge` cannot.
struct TemporalMerge;

impl StageDefinition for TemporalMerge {
    type Args = TemporalArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "temporal-merge"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.radius {
            0 => Parallelism::Frame,
            _ => Parallelism::Chunk,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base").accepting(StreamType::Frames),
            InputParam::required("overlay").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl MaskDefinition for TemporalMerge {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: mask_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(TemporalMergeStage))
    }
}

struct GatherStage;

impl StageInstance for GatherStage {}

impl Mask for GatherStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A mask with an optional param nobody has to bind, ahead of its variadic tail.
///
/// Lets a pipeline put the chunk authority in the tail while the leading slot stays empty, the
/// shape that puts the authority past the end of a signature's own params.
struct Gather;

impl StageDefinition for Gather {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "gather"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::with_variadic_of(
            &[InputParam::optional("guide").accepting(StreamType::Frames)],
            StreamType::Frames,
        );
        SIGNATURE
    }
}

impl MaskDefinition for Gather {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: mask_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(GatherStage))
    }
}

struct WideningSourceStage;

impl StageInstance for WideningSourceStage {}

impl Source for WideningSourceStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A source whose granularity is per frame, unlike every other fixture source.
struct WideningSource;

impl StageDefinition for WideningSource {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "widening-source"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
        SIGNATURE
    }
}

impl SourceDefinition for WideningSource {
    fn produces(_capabilities: &Capabilities, _args: &Self::Args) -> Result<OutputFormats, StageError> {
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
        Ok(Box::new(WideningSourceStage))
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
    registry.register_source::<Wide>().expect("should register");
    registry.register_source::<Deep>().expect("should register");
    registry.register_filter::<Precise>().expect("should register");
    registry.register_filter::<Either>().expect("should register");
    registry.register_filter::<Narrow>().expect("should register");
    registry.register_filter::<Prefers16>().expect("should register");
    registry.register_mask::<Matched>().expect("should register");
    registry.register_mask::<Anchored>().expect("should register");
    registry.register_source::<FloatFirst>().expect("should register");
    registry.register_mask::<Mislabelled>().expect("should register");
    registry.register_filter::<Sparse>().expect("should register");
    registry.register_filter::<Temporal>().expect("should register");
    registry
        .register_mask::<TemporalMerge>()
        .expect("should register");
    registry.register_mask::<Gather>().expect("should register");
    registry
        .register_source::<WideningSource>()
        .expect("should register");

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
        workers: None,
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

        assert_eq!(identifiers.len(), 23);
        assert!(identifiers.contains(&"test/blur"));
        assert!(identifiers.contains(&"test/failing"));
        assert!(identifiers.contains(&"test/recording"));
        assert!(identifiers.contains(&"test/temporal"));
        assert!(identifiers.contains(&"test/temporal-merge"));
        assert!(identifiers.contains(&"test/gather"));
        assert!(identifiers.contains(&"test/widening-source"));
    }

    #[test]
    fn testing_context_allocates_a_frame_carrying_its_metadata() {
        let context = testing_context();

        let frame = context.alloc_frame(&frame_format(), FrameMeta::at(4));

        assert_eq!(frame.meta().index, 4);
    }
}
