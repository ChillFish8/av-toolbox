use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    MetaInputs,
    MetaItem,
    MetaTransform,
    MetaTransformDefinition,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
    Transform,
    TransformDefinition,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    Mask as GpuMask,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamItem,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::engine::tests::support::{StageLife, mask_format};

// A mask stage, whose output is far larger than any frame in these tests, so a budget that only
// measures frames is visibly too small for it.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct MaskingArgs {}

pub(crate) struct MaskingInstance {
    _life: StageLife,
}

impl MaskingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for MaskingInstance {}

impl Mask for MaskingInstance {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let input = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(context.alloc_mask(&mask_format(), *input.meta()))
    }
}

pub(crate) struct Masking;

impl StageDefinition for Masking {
    type Args = MaskingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "masking"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Masking {
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
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::Mask(mask_format()),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(MaskingInstance::new()))
    }
}

// A transform whose fan-out depends on the frame it is given rather than on which call this is,
// so a fixed identity always produces the same number of outputs however the run is chunked or
// widened. It emits three copies, then none, then one, cycling by the input's own timestamp.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct MultiplyingArgs {}

pub(crate) struct MultiplyingInstance {
    _life: StageLife,
}

impl MultiplyingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for MultiplyingInstance {}

impl Transform for MultiplyingInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs,
    ) -> Result<Vec<StreamItem>, StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
        let timestamp = item
            .meta()
            .timestamp
            .expect("every test frame carries a timestamp")
            .as_secs();

        let copies = match timestamp % 3 {
            0 => 3,
            1 => 0,
            _ => 1,
        };

        Ok((0..copies).map(|_| item.clone()).collect())
    }
}

pub(crate) struct Multiplying;

impl StageDefinition for Multiplying {
    type Args = MultiplyingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "multiplying"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl TransformDefinition for Multiplying {
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
        let format = chosen.iter().flatten().next().copied().expect("an input");

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
    ) -> Result<Box<dyn Transform>, StageError> {
        Ok(Box::new(MultiplyingInstance::new()))
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct RestampArgs {}

pub(crate) struct RestampInstance;

impl StageInstance for RestampInstance {}

impl MetaTransform for RestampInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        Ok(inputs.take(0).into_iter().collect())
    }
}

pub(crate) struct RestampDefinition;

impl StageDefinition for RestampDefinition {
    type Args = RestampArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "restamp"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl MetaTransformDefinition for RestampDefinition {
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
        let format = chosen.iter().flatten().next().copied().expect("an input");

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
    ) -> Result<Box<dyn MetaTransform>, StageError> {
        Ok(Box::new(RestampInstance))
    }
}
