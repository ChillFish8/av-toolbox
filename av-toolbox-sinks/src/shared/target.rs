use std::fs::File;
use std::io;
use std::io::Write;
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::Path;

#[cfg(unix)]
use av_toolbox_core::{ClaimError, Claims};
use av_toolbox_core::{DescriptorClaim, StageError};

/// A file a sink writes, holding its descriptor's claim until the file is closed.
#[derive(Debug)]
pub(crate) struct TargetFile<'a> {
    // Declared before the claim so the descriptor is closed before its claim is given back.
    file: File,
    _claim: Option<DescriptorClaim<'a>>,
}

impl TargetFile<'_> {
    fn unclaimed(file: File) -> Self {
        Self { file, _claim: None }
    }
}

impl Write for TargetFile<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.file.write(buffer)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

/// Opens the file a sink writes, from exactly one of an inherited descriptor or a path.
///
/// Opening a FIFO waits until something opens it to read.
pub(crate) fn open_target(
    stage: &str,
    fd: Option<u32>,
    path: Option<&Path>,
) -> Result<TargetFile<'static>, StageError> {
    match (fd, path) {
        (Some(fd), None) => file_from_fd(stage, fd),
        (None, Some(path)) => {
            let created = create_path(path)?;

            Ok(TargetFile::unclaimed(created))
        },
        _ => {
            let message = format!("{stage} takes exactly one of fd or path");

            Err(StageError::msg(message))
        },
    }
}

#[cfg(unix)]
fn file_from_fd(stage: &str, fd: u32) -> Result<TargetFile<'static>, StageError> {
    let claims = Claims::process();

    claimed_file(claims, stage, fd)
}

/// Claims `fd` and takes ownership of it as a file, giving the claim back once the file is closed.
#[cfg(unix)]
fn claimed_file<'a>(claims: &'a Claims, stage: &str, fd: u32) -> Result<TargetFile<'a>, StageError> {
    let parsed = RawFd::try_from(fd);
    let raw = parsed.map_err(|_| {
        let message = format!("{fd} is not a valid file descriptor");

        StageError::msg(message)
    })?;

    let claimed = claims.claim_descriptor(raw);
    let claim = claimed.map_err(|error| claim_error(stage, raw, error))?;

    // SAFETY: the pipeline was given this descriptor to write, and the claim has just made this
    // sink its only owner, from here until the sink drops.
    let file = unsafe { File::from_raw_fd(raw) };

    Ok(TargetFile {
        file,
        _claim: Some(claim),
    })
}

#[cfg(unix)]
fn claim_error(stage: &str, fd: RawFd, error: ClaimError) -> StageError {
    match error {
        ClaimError::StandardDescriptor(_) => {
            let message =
                format!("{stage} cannot write descriptor {fd}, which is standard input, output or error");

            StageError::msg(message)
        },
        other => StageError::msg(other.to_string()),
    }
}

#[cfg(not(unix))]
fn file_from_fd(stage: &str, _fd: u32) -> Result<TargetFile<'static>, StageError> {
    let message = format!("{stage} can only write a file descriptor on Unix, use path instead");

    Err(StageError::msg(message))
}

fn create_path(path: &Path) -> Result<File, StageError> {
    let opened = File::options().write(true).create(true).truncate(true).open(path);

    opened.map_err(|error| {
        let display = path.display();
        let message = format!("{display} could not be opened for writing: {error}");

        StageError::msg(message)
    })
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    use super::*;

    #[cfg(unix)]
    #[test]
    fn the_standard_descriptors_are_refused() {
        let claims = Claims::new();

        for fd in 0..=2 {
            let error = claimed_file(&claims, "y4m/pipe", fd).expect_err("a standard descriptor");

            assert!(
                error.to_string().contains("standard input, output or error"),
                "{fd}"
            );
            assert!(error.to_string().contains("y4m/pipe"), "{fd}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_closed_target_gives_its_descriptor_back() {
        use std::os::fd::IntoRawFd;

        let claims = Claims::new();
        let (_reader, writer) = std::io::pipe().expect("a pipe");
        let raw = writer.into_raw_fd();
        let fd = raw as u32;

        let file = claimed_file(&claims, "y4m/pipe", fd).expect("claims the descriptor");
        drop(file);
        let again = claims.claim_descriptor(raw);

        assert!(again.is_ok());
    }
}
