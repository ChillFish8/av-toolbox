use std::path::PathBuf;

use av_toolbox_core::{BoundInputs, InputSignature, Sink, SinkDefinition, StageDefinition, StageError};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    FormatSet,
    HostSide,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::{INPUTS, RateArg, TargetFile, Y4mSink, accepts, open_target, writer_for};

/// Arguments for `y4m/pipe`, which takes exactly one of `fd` or `path`.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PipeArgs {
    /// An inherited file descriptor, Unix only.
    pub(crate) fd: Option<u32>,
    /// A FIFO, or a file to create or truncate.
    pub(crate) path: Option<PathBuf>,
    /// Overrides the stream's frame rate, written as `"30000/1001"` or `"25"`.
    pub(crate) frame_rate: Option<RateArg>,
}

pub(crate) struct Pipe;

impl StageDefinition for Pipe {
    type Args = PipeArgs;

    fn package() -> &'static str {
        "y4m"
    }

    fn identifier() -> &'static str {
        "pipe"
    }

    fn description() -> &'static str {
        "Writes the stream as y4m to a FIFO, a file or an inherited file descriptor."
    }

    fn inputs() -> &'static InputSignature {
        INPUTS
    }

    fn wants_output_depth() -> bool {
        true
    }
}

impl SinkDefinition for Pipe {
    type Side = HostSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts(offered, args.frame_rate)
    }

    /// Opens the descriptor or path.
    ///
    /// Opening a FIFO waits until something opens it to read.
    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        let sink = sink_for(&args, formats)?;

        Ok(Box::new(sink))
    }
}

fn sink_for(args: &PipeArgs, formats: &ResolvedFormats) -> Result<Y4mSink<TargetFile<'static>>, StageError> {
    let path = args.path.as_deref();
    let file = open_target("y4m/pipe", args.fd, path)?;
    let writer = writer_for(file, formats, args.frame_rate)?;

    Ok(Y4mSink::new(writer))
}

#[cfg(test)]
mod tests {
    use std::fs;
    #[cfg(unix)]
    use std::io;
    #[cfg(unix)]
    use std::io::Read;
    #[cfg(unix)]
    use std::os::fd::IntoRawFd;

    use av_toolbox_gpu::{FrameRate, SampleFormat, testing_context};

    use super::*;
    use crate::shared::testing::{consume, filled_frame, resolved, yuv420};

    fn formats() -> ResolvedFormats {
        let rate = FrameRate::new(25, 1);

        resolved(yuv420(4, 2, SampleFormat::U8, rate))
    }

    fn args(fd: Option<u32>, path: Option<PathBuf>) -> PipeArgs {
        PipeArgs {
            fd,
            path,
            frame_rate: None,
        }
    }

    fn write_frames(sink: &mut Y4mSink<TargetFile<'static>>, count: u64) {
        let rate = FrameRate::new(25, 1);
        let format = yuv420(4, 2, SampleFormat::U8, rate);

        for index in 0..count {
            let frame = filled_frame(&format, index, index as u8);
            consume(sink, frame);
        }

        let mut context = testing_context();
        sink.end_of_stream(&mut context).expect("ends");
    }

    #[test]
    fn both_fd_and_path_is_refused() {
        let both = args(Some(3), Some("a.y4m".into()));

        assert!(sink_for(&both, &formats()).is_err());
    }

    #[test]
    fn neither_fd_nor_path_is_refused() {
        let neither = args(None, None);

        assert!(sink_for(&neither, &formats()).is_err());
    }

    #[test]
    fn a_path_is_written_through() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("out.y4m");
        let target = args(None, Some(path.clone()));

        let mut sink = sink_for(&target, &formats()).expect("builds");
        write_frames(&mut sink, 2);

        let bytes = fs::read(&path).expect("reads");
        let mut decoder = y4m::Decoder::new(bytes.as_slice()).expect("decodes");
        assert!(decoder.read_frame().is_ok());
        assert!(decoder.read_frame().is_ok());
        assert!(decoder.read_frame().is_err());
    }

    #[test]
    fn an_existing_file_is_truncated() {
        let folder = tempfile::tempdir().expect("a temp dir");
        let path = folder.path().join("out.y4m");
        fs::write(&path, vec![0xAA; 4096]).expect("writes the old file");
        let target = args(None, Some(path.clone()));

        let mut sink = sink_for(&target, &formats()).expect("builds");
        write_frames(&mut sink, 1);

        let bytes = fs::read(&path).expect("reads");
        assert_eq!(bytes.len(), 27 + 6 + 12);
    }

    #[cfg(unix)]
    #[test]
    fn an_fd_from_a_real_pipe_is_written_through_and_closed_afterwards() {
        let (mut reader, writer) = io::pipe().expect("a pipe");
        let fd = writer.into_raw_fd() as u32;
        let target = args(Some(fd), None);

        let mut sink = sink_for(&target, &formats()).expect("builds");
        write_frames(&mut sink, 1);
        drop(sink);

        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .expect("reads to the end once the write end closes");
        assert_eq!(bytes.len(), 27 + 6 + 12);
    }
}
