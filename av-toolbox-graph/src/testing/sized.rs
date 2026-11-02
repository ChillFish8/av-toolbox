use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Source,
    SourceDefinition,
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
    Frame,
    FrameContext,
    FrameFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::shared::{NoArgs, first_bound, frame_format};

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct SizedArgs {
    width: u32,
    height: u32,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct ResizeArgs {
    width: u32,
}

struct PassingStage;

impl StageInstance for PassingStage {}

impl Source for PassingStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

impl Filter for PassingStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A source producing U8 frames of the size its args name.
pub(super) struct SizedSource;

impl StageDefinition for SizedSource {
    type Args = SizedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "sized"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
        SIGNATURE
    }
}

impl SourceDefinition for SizedSource {
    type Side = DeviceSide;
    type Opened = FrameFormat;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(FrameFormat {
            width: args.width,
            height: args.height,
            ..frame_format()
        })
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        Ok(OutputFormats {
            template: StreamTemplate::Frames(*opened),
            samples: FormatSet::of([opened.sample]),
        })
    }

    fn build(
        _context: &BuildContext,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(PassingStage))
    }
}

/// A filter passing frames through at the width its args name.
pub(super) struct Resize;

impl StageDefinition for Resize {
    type Args = ResizeArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "resize"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("clip").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Resize {
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
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;
        let Some(frames) = format.as_frames() else {
            return Err(StageError::msg("resize takes frames"));
        };

        let resized = FrameFormat {
            width: args.width,
            ..*frames
        };

        Ok(OutputFormats {
            template: StreamTemplate::Frames(resized),
            samples: FormatSet::of([resized.sample]),
        })
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassingStage))
    }
}

/// A filter that cannot say what it produces.
pub(super) struct Unproducing;

impl StageDefinition for Unproducing {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "unproducing"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("clip").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Unproducing {
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
        _chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        Err(StageError::msg("this stage cannot say what it produces"))
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassingStage))
    }
}
