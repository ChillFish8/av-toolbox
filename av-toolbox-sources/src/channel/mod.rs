use std::sync::mpsc::{Receiver, RecvTimeoutError};

use av_toolbox_core::{
    BoundInputs,
    CHANNEL_POLL,
    ChannelHub,
    ChannelStop,
    InputFrame,
    InputSignature,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FrameContext,
    FrameFormat,
    FrameMeta,
    HostFrame,
    HostSide,
    OutputFormats,
    ResolvedFormats,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::{produced_by, timestamp_of};

#[cfg(test)]
mod tests;

const STAGE: &str = "channel/frames";

/// Arguments for `channel/frames`.
#[derive(Debug, Default, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct ChannelArgs {
    /// Set by the av-toolbox library.
    pub(crate) session: Option<String>,
    /// Set by the av-toolbox library.
    pub(crate) stage: Option<String>,
}

/// The caller's frames and the name this stage reports them under.
pub(crate) struct Opened {
    stage: String,
    format: FrameFormat,
    frames: Receiver<InputFrame>,
    stop: ChannelStop,
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
        "Reads frames handed over by the av-toolbox library caller."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Frames {
    type Side = HostSide;
    type Opened = Opened;

    fn reads_stream(_args: &Self::Args) -> bool {
        true
    }

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        open_from(ChannelHub::process(), args)
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        Ok(produced_by(&opened.format))
    }

    fn build(
        _context: &BuildContext,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        let source = FramesSource {
            opened,
            next_index: 0,
        };

        Ok(Box::new(source))
    }
}

fn open_from(hub: &ChannelHub, args: &ChannelArgs) -> Result<Opened, StageError> {
    let (Some(session), Some(stage)) = (args.session.as_deref(), args.stage.as_deref()) else {
        return Err(StageError::msg(format!(
            "{STAGE} only runs through the av-toolbox library"
        )));
    };

    let bound = hub.take_source(session, stage)?;

    Ok(Opened {
        stage: stage.to_owned(),
        format: bound.format,
        frames: bound.frames,
        stop: bound.stop,
    })
}

struct FramesSource {
    opened: Opened,
    next_index: u64,
}

impl FramesSource {
    /// The caller's next frame, or `None` once the caller closes the stream or the run stops.
    fn next_frame(&self) -> Option<InputFrame> {
        loop {
            if self.opened.stop.is_stopped() {
                return None;
            }

            let received = self.opened.frames.recv_timeout(CHANNEL_POLL);
            match received {
                Ok(frame) => return Some(frame),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => return None,
            }
        }
    }

    fn check(&self, frame: &InputFrame) -> Result<(), StageError> {
        let stage = &self.opened.stage;
        let format = &self.opened.format;
        let channels = format.layout.channels();
        let planes = frame.planes();
        if planes.len() != channels.len() {
            return Err(StageError::msg(format!(
                "stage `{stage}` expected {} planes in frame {} but got {}",
                channels.len(),
                self.next_index,
                planes.len()
            )));
        }

        for (plane_index, (channel, bytes)) in channels.iter().zip(planes).enumerate() {
            let samples = format.channel_samples(*channel).unwrap_or(0);
            let expected = samples * format.sample.bytes();
            if bytes.len() != expected {
                return Err(StageError::msg(format!(
                    "stage `{stage}` expected plane {plane_index} of frame {} to be {expected} bytes but it was {}",
                    self.next_index,
                    bytes.len()
                )));
            }
        }

        Ok(())
    }
}

impl StageInstance for FramesSource {}

impl Source<HostSide> for FramesSource {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<HostFrame>, StageError> {
        let Some(frame) = self.next_frame() else {
            return Ok(None);
        };

        self.check(&frame)?;

        let index = self.next_index;
        let format = self.opened.format;
        let rate = format
            .frame_rate
            .map(|rate| (rate.numerator() as usize, rate.denominator() as usize));
        let timestamp = rate.and_then(|rate| timestamp_of(index, rate));
        let meta = FrameMeta {
            timestamp,
            ..FrameMeta::at(index)
        };
        let mut host_frame = context.alloc_host_frame(&format, meta);
        for (channel, bytes) in format.layout.channels().iter().zip(frame.planes()) {
            if let Some(plane) = host_frame.plane_mut(*channel) {
                plane.copy_from_slice(bytes);
            }
        }

        self.next_index += 1;

        Ok(Some(host_frame.freeze()))
    }
}
