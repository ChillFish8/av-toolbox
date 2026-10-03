use std::time::Duration;

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
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
    Frame,
    FrameContext,
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

use crate::engine::tests::support::StageLife;

// A filter emitting two frames for every input, standing in for a double rate deinterlacer. The first
// keeps the input at index `2n` and timestamp `2t`, the second is a copy at index `2n + 1` and timestamp
// `2t + 1`, so the doubled stream reads as one at twice the rate.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct DoubleArgs {}

pub(crate) struct DoubleInstance {
    _life: StageLife,
}

impl DoubleInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for DoubleInstance {}

impl Filter for DoubleInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let mut first = inputs.take_frame(0).ok_or_else(missing_frame)?;
        let index = first.meta().index;
        let seconds = first.meta().timestamp.map_or(0, |timestamp| timestamp.as_secs());

        first.set_index(index * 2);
        first.meta_mut().timestamp = Some(Duration::from_secs(seconds * 2));

        let mut second = first.clone();
        second.set_index(index * 2 + 1);
        second.meta_mut().timestamp = Some(Duration::from_secs(seconds * 2 + 1));

        Ok(vec![first, second])
    }
}

fn missing_frame() -> StageError {
    StageError::msg("expected a frame in slot 0")
}

/// `format` with its frame rate, when it has one, doubled.
fn doubled_rate(format: StreamFormat) -> StreamFormat {
    let StreamFormat::Frames(mut frame_format) = format else {
        return format;
    };

    frame_format.frame_rate = frame_format
        .frame_rate
        .and_then(|rate| FrameRate::new(rate.numerator() * 2, rate.denominator()));

    StreamFormat::Frames(frame_format)
}

pub(crate) struct Double;

impl StageDefinition for Double {
    type Args = DoubleArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "double"
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

impl FilterDefinition for Double {
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
            template: StreamTemplate::from_format(&doubled_rate(format)),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(DoubleInstance::new()))
    }
}
