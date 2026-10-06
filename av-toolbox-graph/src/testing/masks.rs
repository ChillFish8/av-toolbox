use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    FrameMeta,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};

use super::shared::{NoArgs, TemporalArgs, first_bound, fixture_mask_format, frames_template, mask_template};

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
pub(super) struct Matched;

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
    type Side = DeviceSide;

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
pub(super) struct Anchored;

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
    type Side = DeviceSide;

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
pub(super) struct Mislabelled;

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
    type Side = DeviceSide;

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

struct TemporalMergeStage;

impl StageInstance for TemporalMergeStage {}

impl Mask for TemporalMergeStage {
    fn select(&mut self, context: &mut FrameContext, _inputs: Inputs) -> Result<GpuMask, StageError> {
        Ok(context.alloc_mask(&fixture_mask_format(), FrameMeta::at(0)))
    }
}

/// A two input mask whose granularity comes from its arguments, so it can widen where
/// `test/merge` cannot.
pub(super) struct TemporalMerge;

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
            _ => Parallelism::Chunk(None),
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
    type Side = DeviceSide;

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
pub(super) struct Gather;

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
    type Side = DeviceSide;

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
