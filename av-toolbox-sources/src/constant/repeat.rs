use av_toolbox_core::{Source, StageError, StageInstance};
use av_toolbox_gpu::{DeviceSide, Frame, FrameContext, FrameMeta, HostStreamItem, RuntimeError, StreamItem};

use super::convert::Converted;

fn upload_failed(error: RuntimeError) -> StageError {
    let message = format!("a constant frame could not be uploaded: {error}");

    StageError::msg(message)
}

fn not_a_frame(_item: StreamItem) -> StageError {
    StageError::msg("a constant frame uploaded as something other than a frame")
}

/// One frame, uploaded on first use and handed out as a clone after that.
pub(crate) struct Repeat {
    converted: Converted,
    uploaded: Option<Frame>,
}

impl Repeat {
    pub(crate) fn new(converted: Converted) -> Self {
        Self {
            converted,
            uploaded: None,
        }
    }

    fn upload(&self, context: &FrameContext) -> Result<Frame, StageError> {
        let format = &self.converted.format;
        let first = FrameMeta::at(0);
        let mut host = context.alloc_host_frame(format, first);

        for (channel, bytes) in format.channels().iter().zip(&self.converted.planes) {
            let plane = host
                .plane_mut(*channel)
                .expect("a frame carries every channel its format lists");
            plane.copy_from_slice(bytes);
        }

        let frozen = host.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).map_err(upload_failed)?;

        loaded.into_frame().map_err(not_a_frame)
    }

    /// The frame, carrying `meta`.
    ///
    /// Every frame handed out shares the same planes.
    pub(crate) fn frame(&mut self, context: &mut FrameContext, meta: FrameMeta) -> Result<Frame, StageError> {
        let cached = match self.uploaded.take() {
            Some(cached) => cached,
            None => self.upload(context)?,
        };

        let mut frame = cached.clone();
        *frame.meta_mut() = meta;
        self.uploaded = Some(cached);

        Ok(frame)
    }
}

/// A source handing out `frames` copies of one frame, indexed from 0.
pub(crate) struct RepeatSource {
    repeat: Repeat,
    frames: u64,
    next: u64,
}

impl RepeatSource {
    pub(crate) fn new(converted: Converted, frames: u64) -> Self {
        let repeat = Repeat::new(converted);

        Self {
            repeat,
            frames,
            next: 0,
        }
    }
}

impl StageInstance for RepeatSource {}

impl Source<DeviceSide> for RepeatSource {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        if self.next >= self.frames {
            return Ok(None);
        }

        let meta = FrameMeta::at(self.next);
        let frame = self.repeat.frame(context, meta)?;
        self.next += 1;

        Ok(Some(frame))
    }
}
