mod carried;
mod timestamp;
mod y4m;

#[cfg(test)]
pub(crate) mod tests;

use std::io::Read;

use av_toolbox_core::{Source, StageError, StageInstance};
use av_toolbox_gpu::{
    FormatSet,
    FrameContext,
    FrameFormat,
    HostFrame,
    HostSide,
    OutputFormats,
    StreamFormat,
    StreamTemplate,
};

pub(crate) use self::carried::Carried;
pub(crate) use self::timestamp::timestamp_of;
pub(crate) use self::y4m::Y4mReader;

/// A source reading a y4m stream.
pub(crate) struct Y4mSource<R: Read> {
    reader: Y4mReader<R>,
}

impl<R: Read> Y4mSource<R> {
    pub(crate) fn new(reader: Y4mReader<R>) -> Self {
        Self { reader }
    }

    pub(crate) fn next_frame(&mut self, context: &FrameContext) -> Result<Option<HostFrame>, StageError> {
        self.reader.read_frame(context)
    }
}

impl<R: Read + Send + 'static> StageInstance for Y4mSource<R> {}

impl<R: Read + Send + 'static> Source<HostSide> for Y4mSource<R> {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<HostFrame>, StageError> {
        self.next_frame(context)
    }
}

pub(crate) fn produced_by(format: &FrameFormat) -> OutputFormats {
    let stream_format = StreamFormat::Frames(*format);

    OutputFormats {
        template: StreamTemplate::from_format(&stream_format),
        samples: FormatSet::of([format.sample]),
    }
}
