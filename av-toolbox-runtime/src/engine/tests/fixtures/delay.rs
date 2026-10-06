use std::collections::VecDeque;

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
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::chunked;
use crate::engine::tests::support::StageLife;

// A filter holding a window of frames and emitting each one unchanged once the window is full, so the
// engine has to carry a stage whose output lags its input. The lag is chosen per chunk id, so a run can
// vary it.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct DelayArgs {
    /// Frames held for chunk id `n` is entry `n`, the last entry repeating.
    lag: Vec<usize>,
    /// Declares the stage per frame rather than per chunk.
    #[serde(default)]
    per_frame: bool,
    /// Declares no held frames whatever `lag` is, standing in for a stage that holds frames without
    /// saying so.
    #[serde(default)]
    undeclared: bool,
}

pub(crate) struct DelayInstance {
    lag: Vec<usize>,
    held: VecDeque<Frame>,
    _life: StageLife,
}

impl DelayInstance {
    pub(crate) fn from_args(args: DelayArgs) -> Self {
        Self {
            lag: args.lag,
            held: VecDeque::new(),
            _life: StageLife::new(),
        }
    }

    fn lag_for(&self, chunk: u64) -> usize {
        let last = self.lag.len().saturating_sub(1);
        let position = (chunk as usize).min(last);

        self.lag.get(position).copied().unwrap_or(0)
    }
}

impl StageInstance for DelayInstance {}

impl Filter for DelayInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs.take_frame(0).ok_or_else(missing_frame)?;
        let chunk = frame.meta().chunk;

        if let Some(oldest) = self.held.front() {
            let held_chunk = oldest.meta().chunk;

            if held_chunk != chunk {
                return Err(window_spans_chunks(held_chunk, chunk));
            }
        }

        self.held.push_back(frame);

        let lag = self.lag_for(chunk);
        if self.held.len() <= lag {
            return Ok(Vec::new());
        }

        let oldest = self.held.pop_front().expect("more than the lag is held");
        Ok(vec![oldest])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let flushed = self.held.drain(..).collect();

        Ok(flushed)
    }
}

fn missing_frame() -> StageError {
    StageError::msg("expected a frame in slot 0")
}

fn window_spans_chunks(held: u64, arrived: u64) -> StageError {
    let message = format!("a frame of chunk {arrived} arrived while frames of chunk {held} were held");

    StageError::msg(message)
}

pub(crate) struct Delay;

impl StageDefinition for Delay {
    type Args = DelayArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "delay"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_frame {
            true => Parallelism::Frame,
            false => chunked(),
        }
    }

    fn held_frames(args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        if args.undeclared {
            return 0;
        }

        args.lag.iter().copied().max().unwrap_or(0)
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Delay {
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
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(DelayInstance::from_args(args)))
    }
}
