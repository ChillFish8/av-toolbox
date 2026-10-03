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
    FrameRate,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::shared::{first_bound, frame_format};

#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
pub(crate) struct RateArgs {
    numerator: u32,
    denominator: u32,
}

impl RateArgs {
    fn rate(self) -> Result<FrameRate, StageError> {
        let rate = FrameRate::new(self.numerator, self.denominator);

        rate.ok_or_else(|| StageError::msg("a fixture frame rate needs two non-zero parts"))
    }
}

fn frames_at(format: FrameFormat) -> OutputFormats {
    let stream = StreamFormat::Frames(format);

    OutputFormats {
        template: StreamTemplate::from_format(&stream),
        samples: FormatSet::of([format.sample]),
    }
}

struct RatedStage;

impl StageInstance for RatedStage {}

impl Source for RatedStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A source emitting the fixture format at the frame rate it is given.
pub(super) struct Rated;

impl StageDefinition for Rated {
    type Args = RateArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "rated"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Rated {
    type Side = DeviceSide;
    type Opened = FrameRate;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        args.rate()
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = FrameFormat {
            frame_rate: Some(*opened),
            ..frame_format()
        };

        Ok(frames_at(format))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(RatedStage))
    }
}

struct RetimeStage;

impl StageInstance for RetimeStage {}

impl Filter for RetimeStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A filter passing its input through at the frame rate it is given.
pub(super) struct Retime;

impl StageDefinition for Retime {
    type Args = RateArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "retime"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Retime {
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
        let rate = args.rate()?;
        let chosen_format = first_bound(chosen)?;

        let StreamFormat::Frames(format) = chosen_format else {
            return Err(StageError::msg("test/retime takes frames"));
        };

        let retimed = FrameFormat {
            frame_rate: Some(rate),
            ..format
        };

        Ok(frames_at(retimed))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(RetimeStage))
    }
}
