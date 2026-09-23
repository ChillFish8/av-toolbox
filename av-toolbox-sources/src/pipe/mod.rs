#[cfg(unix)]
use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::{Path, PathBuf};
#[cfg(unix)]
use std::sync::{LazyLock, Mutex};

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

/// Every descriptor a `y4m/pipe` source has taken ownership of in this process.
#[cfg(unix)]
static CLAIMED_DESCRIPTORS: LazyLock<Mutex<HashSet<RawFd>>> = LazyLock::new(Mutex::default);

/// Guards each descriptor against being owned, and so closed, by more than one `y4m/pipe` source.
///
/// A claim is never given back, since a closed descriptor's number can be reused by an unrelated file.
#[cfg(unix)]
struct DescriptorGuard {
    claimed: &'static Mutex<HashSet<RawFd>>,
}

#[cfg(unix)]
impl DescriptorGuard {
    fn process() -> Self {
        Self {
            claimed: &CLAIMED_DESCRIPTORS,
        }
    }

    /// Claims `fd`, failing for standard input, output or error and for a descriptor already claimed.
    fn claim(&self, fd: RawFd) -> Result<(), StageError> {
        if (0..=2).contains(&fd) {
            let message =
                format!("y4m/pipe cannot read descriptor {fd}, use y4m/stdin to read standard input");

            return Err(StageError::msg(message));
        }

        let mut claimed = self
            .claimed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let newly_claimed = claimed.insert(fd);

        if !newly_claimed {
            let message = format!("descriptor {fd} is already read by another y4m/pipe source");

            return Err(StageError::msg(message));
        }

        Ok(())
    }
}

#[cfg(all(test, unix))]
impl DescriptorGuard {
    fn for_test() -> Self {
        let leaked = Box::leak(Box::default());

        Self { claimed: leaked }
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
    let guard = DescriptorGuard::process();

    claimed_file(&guard, fd)
}

/// Claims `fd` through `guard` and takes ownership of it as a file.
#[cfg(unix)]
fn claimed_file(guard: &DescriptorGuard, fd: u32) -> Result<File, StageError> {
    let parsed = RawFd::try_from(fd);
    let raw = parsed.map_err(|_| {
        let message = format!("{fd} is not a valid file descriptor");

        StageError::msg(message)
    })?;

    guard.claim(raw)?;

    // SAFETY: the pipeline was given this descriptor to read, and the guard has just made this
    // source its only owner, from here until the source drops.
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

        let guard = DescriptorGuard::for_test();
        let (reader, _writer) = std::io::pipe().expect("a pipe");
        let fd = reader.into_raw_fd() as u32;

        let first = claimed_file(&guard, fd);
        let second = claimed_file(&guard, fd);

        assert!(first.is_ok());

        let error = second.expect_err("the descriptor is already claimed");
        assert!(error.to_string().contains("already"));
    }

    #[cfg(unix)]
    #[test]
    fn the_standard_descriptors_are_refused() {
        let guard = DescriptorGuard::for_test();

        for fd in 0..=2 {
            let error = claimed_file(&guard, fd).expect_err("a standard descriptor");

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
}
