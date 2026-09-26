use av_toolbox_core::{Filter, Inputs, Source, StageError, StageInstance};
use av_toolbox_gpu::{
    DeviceSide,
    Frame,
    FrameContext,
    FrameMeta,
    FrameRate,
    HostStreamItem,
    RuntimeError,
    StreamItem,
};

use super::convert::Converted;
use crate::shared::timestamp_of;

fn upload_failed(error: RuntimeError) -> StageError {
    let message = format!("a constant frame could not be uploaded: {error}");

    StageError::msg(message)
}

fn not_a_frame(_item: StreamItem) -> StageError {
    StageError::msg("a constant frame uploaded as something other than a frame")
}

fn missing_reference(stage: &str) -> StageError {
    let message = format!("{stage} expected a reference frame in slot 0");

    StageError::msg(message)
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

/// The meta of frame `index`, with a timestamp when there is a frame rate.
fn meta_at(index: u64, frame_rate: Option<FrameRate>) -> FrameMeta {
    let mut meta = FrameMeta::at(index);

    if let Some(rate) = frame_rate {
        let numerator = rate.numerator() as usize;
        let denominator = rate.denominator() as usize;
        meta.timestamp = timestamp_of(index, (numerator, denominator));
    }

    meta
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

        let frame_rate = self.repeat.converted.format.frame_rate;
        let meta = meta_at(self.next, frame_rate);
        let frame = self.repeat.frame(context, meta)?;
        self.next += 1;

        Ok(Some(frame))
    }
}

/// A filter handing out one frame in place of each reference frame, carrying the reference's meta.
pub(crate) struct RepeatFilter {
    repeat: Repeat,
    stage: &'static str,
}

impl RepeatFilter {
    pub(crate) fn new(converted: Converted, stage: &'static str) -> Self {
        let repeat = Repeat::new(converted);

        Self { repeat, stage }
    }
}

impl StageInstance for RepeatFilter {}

impl Filter for RepeatFilter {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(reference) = inputs.take_frame(0) else {
            return Err(missing_reference(self.stage));
        };

        let meta = reference.meta().clone();
        drop(reference);

        self.repeat.frame(context, meta)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn frame_two_at_25_fps_is_at_80_ms() {
        let frame_rate = FrameRate::new(25, 1);

        let meta = meta_at(2, frame_rate);

        assert_eq!(meta.index, 2);
        assert_eq!(meta.timestamp, Some(Duration::from_millis(80)));
    }

    #[test]
    fn no_frame_rate_gives_no_timestamp() {
        let meta = meta_at(2, None);

        assert_eq!(meta.index, 2);
        assert_eq!(meta.timestamp, None);
    }
}
