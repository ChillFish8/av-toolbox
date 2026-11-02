use std::fs::File;
use std::io;
use std::io::{BufReader, Read};
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;
use std::path::{Path, PathBuf};

use av_toolbox_core::{
    BoundInputs,
    DescriptorClaim,
    InputSignature,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
};
#[cfg(unix)]
use av_toolbox_core::{ClaimError, Claims};
use av_toolbox_gpu::{BuildContext, Capabilities, HostSide, OutputFormats, ResolvedFormats};
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

pub(crate) type Opened = Y4mReader<BufReader<PipeFile<'static>>>;

/// A file a pipe reads, holding its descriptor's claim until the file is closed.
#[derive(Debug)]
pub(crate) struct PipeFile<'a> {
    // Declared before the claim so the descriptor is closed before its claim is given back.
    file: File,
    _claim: Option<DescriptorClaim<'a>>,
}

impl PipeFile<'_> {
    fn unclaimed(file: File) -> Self {
        Self { file, _claim: None }
    }
}

impl Read for PipeFile<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        self.file.read(buffer)
    }
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
        "Reads a y4m stream from a FIFO, a file or an inherited file descriptor."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Pipe {
    type Side = HostSide;
    type Opened = Opened;

    /// Opens the descriptor or path.
    ///
    /// A pipe source can be built once, since its header is read here.
    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        let file = match (args.fd, &args.path) {
            (Some(fd), None) => file_from_fd(fd)?,
            (None, Some(path)) => {
                let opened = open_path(path)?;

                PipeFile::unclaimed(opened)
            },
            _ => return Err(StageError::msg("y4m/pipe takes exactly one of fd or path")),
        };

        let reader = BufReader::new(file);

        Y4mReader::open(reader)
    }

    fn reads_stream(args: &Self::Args) -> bool {
        match (args.fd, &args.path) {
            (Some(_), None) => true,
            (None, Some(path)) => path_is_fifo(path),
            _ => false,
        }
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = opened.format();
        let produced = produced_by(format);

        Ok(produced)
    }

    fn build(
        _context: &BuildContext,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        let source = Y4mSource::new(opened);

        Ok(Box::new(source))
    }
}

#[cfg(unix)]
fn file_from_fd(fd: u32) -> Result<PipeFile<'static>, StageError> {
    let claims = Claims::process();

    claimed_file(claims, fd)
}

/// Claims `fd` and takes ownership of it as a file, giving the claim back once the file is closed.
#[cfg(unix)]
fn claimed_file(claims: &Claims, fd: u32) -> Result<PipeFile<'_>, StageError> {
    let parsed = RawFd::try_from(fd);
    let raw = parsed.map_err(|_| {
        let message = format!("{fd} is not a valid file descriptor");

        StageError::msg(message)
    })?;

    let claimed = claims.claim_descriptor(raw);
    let claim = claimed.map_err(|error| claim_error(raw, error))?;

    // SAFETY: the pipeline was given this descriptor to read, and the claim has just made this
    // source its only owner, from here until the source drops.
    let file = unsafe { File::from_raw_fd(raw) };

    Ok(PipeFile {
        file,
        _claim: Some(claim),
    })
}

#[cfg(unix)]
fn claim_error(fd: RawFd, error: ClaimError) -> StageError {
    match error {
        ClaimError::StandardDescriptor(_) => {
            let message =
                format!("y4m/pipe cannot read descriptor {fd}, use y4m/stdin to read standard input");

            StageError::msg(message)
        },
        other => StageError::msg(other.to_string()),
    }
}

#[cfg(not(unix))]
fn file_from_fd(_fd: u32) -> Result<PipeFile<'static>, StageError> {
    Err(StageError::msg(
        "y4m/pipe can only read a file descriptor on Unix, use path instead",
    ))
}

/// Whether `path` is a FIFO, false for anything else, including a path that cannot be read.
#[cfg(unix)]
fn path_is_fifo(path: &Path) -> bool {
    let metadata = std::fs::metadata(path);
    let is_fifo = metadata.map(|metadata| metadata.file_type().is_fifo());

    is_fifo.unwrap_or(false)
}

#[cfg(not(unix))]
fn path_is_fifo(_path: &Path) -> bool {
    false
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
        let bytes = y4m_bytes(3);
        file.write_all(&bytes).expect("writes");
        let args = PipeArgs {
            fd: None,
            path: Some(file.path().to_path_buf()),
        };

        let opened = Pipe::open(&args).expect("opens");

        assert_eq!(read_all(opened), 3);
    }

    #[cfg(unix)]
    #[test]
    fn an_fd_from_a_real_pipe_reads_through_and_is_closed_afterwards() {
        use std::os::fd::IntoRawFd;

        let context = testing_context();
        let (reader, mut writer) = std::io::pipe().expect("a pipe");
        let bytes = y4m_bytes(2);
        writer.write_all(&bytes).expect("writes");
        let fd = reader.into_raw_fd() as u32;
        let args = PipeArgs {
            fd: Some(fd),
            path: None,
        };

        let opened = Pipe::open(&args).expect("opens");
        let mut source = Y4mSource::new(opened);
        let first = source.next_frame(&context).expect("reads");
        let second = source.next_frame(&context).expect("reads");
        drop(source);

        assert!(first.is_some());
        assert!(second.is_some());

        let written = writer.write_all(b"FRAME");
        let error = written.expect_err("the read end is closed");
        assert_eq!(error.kind(), std::io::ErrorKind::BrokenPipe);
    }

    #[cfg(unix)]
    #[test]
    fn a_second_claim_of_one_descriptor_is_refused() {
        use std::os::fd::IntoRawFd;

        let claims = Claims::new();
        let (reader, _writer) = std::io::pipe().expect("a pipe");
        let fd = reader.into_raw_fd() as u32;

        let first = claimed_file(&claims, fd);
        let second = claimed_file(&claims, fd);

        assert!(first.is_ok());

        let error = second.expect_err("the descriptor is already claimed");
        assert!(error.to_string().contains("already"));
    }

    #[cfg(unix)]
    #[test]
    fn a_closed_pipe_gives_its_descriptor_back() {
        use std::os::fd::IntoRawFd;

        let claims = Claims::new();
        let (reader, _writer) = std::io::pipe().expect("a pipe");
        let raw = reader.into_raw_fd();
        let fd = raw as u32;

        let file = claimed_file(&claims, fd).expect("claims the descriptor");
        drop(file);
        let again = claims.claim_descriptor(raw);

        assert!(again.is_ok());
    }

    #[cfg(unix)]
    #[test]
    fn the_standard_descriptors_are_refused() {
        let claims = Claims::new();

        for fd in 0..=2 {
            let error = claimed_file(&claims, fd).expect_err("a standard descriptor");

            assert!(error.to_string().contains("y4m/stdin"), "{fd}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn descriptor_zero_is_refused_by_the_stage() {
        let args = PipeArgs {
            fd: Some(0),
            path: None,
        };

        let error = Pipe::open(&args).err().expect("stdin is refused");

        assert!(error.to_string().contains("y4m/stdin"));
    }

    #[test]
    fn an_fd_reads_a_live_stream() {
        let args = PipeArgs {
            fd: Some(3),
            path: None,
        };

        assert!(Pipe::reads_stream(&args));
    }

    #[test]
    fn a_regular_file_path_does_not_read_a_live_stream() {
        let file = tempfile::NamedTempFile::new().expect("a temp file");
        let path = file.path().to_path_buf();
        let args = PipeArgs {
            fd: None,
            path: Some(path),
        };

        assert!(!Pipe::reads_stream(&args));
    }

    #[test]
    fn neither_fd_nor_path_does_not_read_a_live_stream() {
        let args = PipeArgs { fd: None, path: None };

        assert!(!Pipe::reads_stream(&args));
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_path_reads_a_live_stream() {
        let directory = tempfile::tempdir().expect("a temp directory");
        let fifo_path = directory.path().join("input.fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("runs mkfifo");
        assert!(status.success());

        let args = PipeArgs {
            fd: None,
            path: Some(fifo_path),
        };

        assert!(Pipe::reads_stream(&args));
    }
}
