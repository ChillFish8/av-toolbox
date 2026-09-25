use std::fs::File;
#[cfg(unix)]
use std::os::fd::{FromRawFd, RawFd};
use std::path::Path;

use av_toolbox_core::StageError;
#[cfg(unix)]
use av_toolbox_core::{ClaimError, Claims};

/// Opens the file a sink writes, from exactly one of an inherited descriptor or a path.
///
/// Opening a FIFO waits until something opens it to read.
pub(crate) fn open_target(stage: &str, fd: Option<u32>, path: Option<&Path>) -> Result<File, StageError> {
    match (fd, path) {
        (Some(fd), None) => file_from_fd(stage, fd),
        (None, Some(path)) => create_path(path),
        _ => {
            let message = format!("{stage} takes exactly one of fd or path");

            Err(StageError::msg(message))
        },
    }
}

#[cfg(unix)]
fn file_from_fd(stage: &str, fd: u32) -> Result<File, StageError> {
    let claims = Claims::process();

    claimed_file(claims, stage, fd)
}

/// Claims `fd` and takes ownership of it as a file.
#[cfg(unix)]
fn claimed_file(claims: &Claims, stage: &str, fd: u32) -> Result<File, StageError> {
    let parsed = RawFd::try_from(fd);
    let raw = parsed.map_err(|_| {
        let message = format!("{fd} is not a valid file descriptor");

        StageError::msg(message)
    })?;

    let claimed = claims.claim_descriptor(raw);
    claimed.map_err(|error| claim_error(stage, raw, error))?;

    // SAFETY: the pipeline was given this descriptor to write, and the claim has just made this
    // sink its only owner, from here until the sink drops.
    let file = unsafe { File::from_raw_fd(raw) };

    Ok(file)
}

#[cfg(unix)]
fn claim_error(stage: &str, fd: RawFd, error: ClaimError) -> StageError {
    match error {
        ClaimError::StandardDescriptor(_) => {
            let message = format!(
                "{stage} cannot write descriptor {fd}, standard output is written by its own sink such as y4m/stdout"
            );

            StageError::msg(message)
        },
        other => StageError::msg(other.to_string()),
    }
}

#[cfg(not(unix))]
fn file_from_fd(stage: &str, _fd: u32) -> Result<File, StageError> {
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

            assert!(error.to_string().contains("y4m/stdout"), "{fd}");
        }
    }
}
