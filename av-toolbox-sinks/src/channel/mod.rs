use std::sync::mpsc::{SyncSender, TrySendError};
use std::thread;
use std::time::Duration;

use av_toolbox_core::{
    BoundInputs,
    CHANNEL_POLL,
    ChannelHub,
    ChannelStop,
    InputSignature,
    Inputs,
    OutputFrame,
    Sink,
    SinkDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    FormatSet,
    FrameContext,
    HostFrame,
    HostSide,
    Offered,
    ResolvedFormats,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::INPUTS;

#[cfg(test)]
mod tests;

const STAGE: &str = "channel/frames";

/// The first wait for room on a full channel, doubling up to [CHANNEL_POLL] while it stays full.
const FIRST_WAIT: Duration = Duration::from_millis(1);

/// Arguments for `channel/frames`.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ChannelArgs {
    /// Set by the av-toolbox library.
    pub(crate) session: Option<String>,
    /// Set by the av-toolbox library.
    pub(crate) stage: Option<String>,
}

pub(crate) struct Frames;

impl StageDefinition for Frames {
    type Args = ChannelArgs;

    fn package() -> &'static str {
        "channel"
    }

    fn identifier() -> &'static str {
        "frames"
    }

    fn description() -> &'static str {
        "Hands frames back to the av-toolbox library caller."
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }
}

impl SinkDefinition for Frames {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let StreamFormat::Frames(format) = offered else {
            return Err(StageError::msg(format!("{STAGE} takes frames, not a mask")));
        };

        Ok(FormatSet::of([format.sample]))
    }

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        let sink = sink_from(ChannelHub::process(), &args)?;

        Ok(Box::new(sink))
    }
}

fn sink_from(hub: &ChannelHub, args: &ChannelArgs) -> Result<FramesSink, StageError> {
    let (Some(session), Some(stage)) = (args.session.as_deref(), args.stage.as_deref()) else {
        return Err(StageError::msg(format!(
            "{STAGE} only runs through the av-toolbox library"
        )));
    };

    let bound = hub.take_sink(session, stage)?;

    Ok(FramesSink {
        stage: stage.to_owned(),
        frames: Some(bound.frames),
        stop: bound.stop,
    })
}

fn missing_frame_error(stage: &str) -> StageError {
    StageError::msg(format!("stage `{stage}` expected a frame in slot 0"))
}

fn ended_error(stage: &str) -> StageError {
    StageError::msg(format!("stage `{stage}` already reached the end of the stream"))
}

fn missing_plane_error(stage: &str, channel: Channel) -> StageError {
    StageError::msg(format!(
        "stage `{stage}` has no plane for channel {channel:?} in its frame"
    ))
}

fn stopped_receiving_error(stage: &str) -> StageError {
    StageError::msg(format!(
        "stage `{stage}` stopped because the caller stopped receiving frames"
    ))
}

struct FramesSink {
    stage: String,
    frames: Option<SyncSender<OutputFrame>>,
    stop: ChannelStop,
}

impl FramesSink {
    /// Sends `output` to the caller, waiting while the channel is full.
    ///
    /// Once the run stops, the frame is dropped and the sink returns quietly, since the run has ended.
    fn hand_over(&self, frames: &SyncSender<OutputFrame>, output: OutputFrame) -> Result<(), StageError> {
        let mut pending = output;
        let mut wait = FIRST_WAIT;

        loop {
            let sent = frames.try_send(pending);
            match sent {
                Ok(()) => return Ok(()),
                Err(TrySendError::Disconnected(_)) => return Err(stopped_receiving_error(&self.stage)),
                Err(TrySendError::Full(returned)) => pending = returned,
            }

            if self.stop.is_stopped() {
                return Ok(());
            }

            thread::sleep(wait);
            wait = wait.saturating_mul(2).min(CHANNEL_POLL);
        }
    }

    fn planes_of(&self, frame: &HostFrame) -> Result<Vec<Vec<u8>>, StageError> {
        let mut planes = Vec::new();
        for channel in frame.format().channels() {
            let plane = frame
                .plane(*channel)
                .ok_or_else(|| missing_plane_error(&self.stage, *channel))?;
            planes.push(plane.bytes().to_vec());
        }

        Ok(planes)
    }
}

impl StageInstance for FramesSink {}

impl Sink<HostSide> for FramesSink {
    fn consume(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<(), StageError> {
        let Some(frame) = inputs.take_frame(0) else {
            return Err(missing_frame_error(&self.stage));
        };
        let Some(frames) = self.frames.as_ref() else {
            return Err(ended_error(&self.stage));
        };

        let planes = self.planes_of(&frame)?;
        let output = OutputFrame {
            index: frame.meta().index,
            format: *frame.format(),
            planes,
        };

        self.hand_over(frames, output)
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.frames.take();

        Ok(())
    }
}
