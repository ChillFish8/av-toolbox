use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use av_toolbox_core::{BoundInputs, InputSignature, Source, SourceDefinition, StageDefinition, StageError};
use av_toolbox_gpu::{BuildContext, Capabilities, HostSide, OutputFormats, ResolvedFormats, Runtime};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::shared::{Y4mReader, Y4mSource, produced_by};

/// Arguments for `y4m/pipe`, exactly one of which must be given.
#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct PipeArgs {
    /// An inherited file descriptor, Unix only.
    pub(crate) fd: Option<u32>,
    /// A FIFO or a `.y4m` file.
    pub(crate) path: Option<PathBuf>,
}

pub(crate) type Opened = Y4mReader<BufReader<File>>;

pub(crate) struct Pipe;

impl StageDefinition for Pipe {
    type Args = PipeArgs;

    fn package() -> &'static str {
        "y4m"
    }

    fn identifier() -> &'static str {
        "pipe"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Pipe {
    type Side = HostSide;
    type Opened = Opened;

    /// Opens the descriptor or path. A pipe source can be built once, since its header is read here.
    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        let file = match (args.fd, &args.path) {
            (Some(fd), None) => file_from_fd(fd)?,
            (None, Some(path)) => open_path(path)?,
            _ => return Err(StageError::msg("y4m/pipe takes exactly one of fd or path")),
        };
        let reader = BufReader::new(file);

        Y4mReader::open(reader)
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = opened.format();
        let produced = produced_by(format);

        Ok(produced)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        let source = Y4mSource::new(opened);

        Ok(Box::new(source))
    }
}

#[cfg(unix)]
fn file_from_fd(fd: u32) -> Result<File, StageError> {
    use std::os::fd::{FromRawFd, RawFd};

    let parsed = RawFd::try_from(fd);
    let raw = parsed.map_err(|_| {
        let message = format!("{fd} is not a valid file descriptor");

        StageError::msg(message)
    })?;

    // SAFETY: the pipeline was given this descriptor to read, and owns it from here until the source drops.
    let file = unsafe { File::from_raw_fd(raw) };

    Ok(file)
}

#[cfg(not(unix))]
fn file_from_fd(_fd: u32) -> Result<File, StageError> {
    Err(StageError::msg(
        "y4m/pipe can only read a file descriptor on Unix, use path instead",
    ))
}

fn open_path(path: &Path) -> Result<File, StageError> {
    let opened = File::open(path);

    opened.map_err(|error| {
        let display = path.display();
        let message = format!("{display} could not be opened: {error}");

        StageError::msg(message)
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use av_toolbox_gpu::testing_context;

    use super::*;

    fn y4m_bytes(frames: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mut encoder = y4m::encode(4, 2, y4m::Ratio::new(25, 1))
            .write_header(&mut bytes)
            .expect("writes the header");
        let planes = [vec![1u8; 8], vec![2u8; 2], vec![3u8; 2]];
        for _ in 0..frames {
            let frame = y4m::Frame::new([&planes[0], &planes[1], &planes[2]], None);
            encoder.write_frame(&frame).expect("writes");
        }

        bytes
    }

    fn read_all(opened: Opened) -> usize {
        let context = testing_context();
        let mut source = Y4mSource::new(opened);
        let mut count = 0;
        while source.next_frame(&context).expect("reads").is_some() {
            count += 1;
        }

        count
    }

    #[test]
    fn both_fd_and_path_is_refused() {
        let args = PipeArgs {
            fd: Some(3),
            path: Some("a.y4m".into()),
        };

        assert!(Pipe::open(&args).is_err());
    }

    #[test]
    fn neither_fd_nor_path_is_refused() {
        let args = PipeArgs { fd: None, path: None };

        assert!(Pipe::open(&args).is_err());
    }

    #[test]
    fn a_path_reads_through() {
        let mut file = tempfile::NamedTempFile::new().expect("a temp file");
        file.write_all(&y4m_bytes(3)).expect("writes");
        let args = PipeArgs {
            fd: None,
            path: Some(file.path().to_path_buf()),
        };

        let opened = Pipe::open(&args).expect("opens");

        assert_eq!(read_all(opened), 3);
    }

    #[cfg(unix)]
    #[test]
    fn an_fd_from_a_real_pipe_reads_through() {
        use std::os::fd::IntoRawFd;

        let (reader, mut writer) = std::io::pipe().expect("a pipe");
        writer.write_all(&y4m_bytes(2)).expect("writes");
        drop(writer);
        let args = PipeArgs {
            fd: Some(reader.into_raw_fd() as u32),
            path: None,
        };

        let opened = Pipe::open(&args).expect("opens");

        assert_eq!(read_all(opened), 2);
    }
}
