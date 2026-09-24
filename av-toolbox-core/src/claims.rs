use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex};

static PROCESS_CLAIMS: LazyLock<Claims> = LazyLock::new(Claims::new);

/// The standard streams and descriptors stages have taken ownership of.
///
/// A claim is never given back, since a closed descriptor's number can be reused by an unrelated
/// file.
#[derive(Debug, Default)]
pub struct Claims {
    stdin: AtomicBool,
    stdout: AtomicBool,
    descriptors: Mutex<HashSet<i32>>,
}

/// Why a claim was refused.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ClaimError {
    StdinTaken,
    StdoutTaken,
    /// Descriptors `0..=2` are claimed as standard streams rather than as descriptors.
    StandardDescriptor(i32),
    DescriptorTaken(i32),
}

impl Claims {
    /// The claims shared by every stage in this process.
    pub fn process() -> &'static Claims {
        &PROCESS_CLAIMS
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn claim_stdin(&self) -> Result<(), ClaimError> {
        let already_taken = self.stdin.swap(true, Ordering::SeqCst);
        if already_taken {
            return Err(ClaimError::StdinTaken);
        }

        Ok(())
    }

    pub fn claim_stdout(&self) -> Result<(), ClaimError> {
        let already_taken = self.stdout.swap(true, Ordering::SeqCst);
        if already_taken {
            return Err(ClaimError::StdoutTaken);
        }

        Ok(())
    }

    pub fn claim_descriptor(&self, fd: i32) -> Result<(), ClaimError> {
        if (0..=2).contains(&fd) {
            return Err(ClaimError::StandardDescriptor(fd));
        }

        let mut claimed = self
            .descriptors
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let newly_claimed = claimed.insert(fd);

        if !newly_claimed {
            return Err(ClaimError::DescriptorTaken(fd));
        }

        Ok(())
    }
}

impl fmt::Display for ClaimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StdinTaken => formatter.write_str("standard input is already read by another stage"),
            Self::StdoutTaken => formatter.write_str("standard output is already written by another stage"),
            Self::StandardDescriptor(fd) => {
                write!(formatter, "descriptor {fd} is standard input, output or error")
            },
            Self::DescriptorTaken(fd) => {
                write!(formatter, "descriptor {fd} is already owned by another stage")
            },
        }
    }
}

impl std::error::Error for ClaimError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_descriptor_is_claimed_once() {
        let claims = Claims::new();

        let first = claims.claim_descriptor(7);
        let second = claims.claim_descriptor(7);

        assert_eq!(first, Ok(()));
        assert_eq!(second, Err(ClaimError::DescriptorTaken(7)));
    }

    #[test]
    fn the_standard_descriptors_cannot_be_claimed_as_descriptors() {
        let claims = Claims::new();

        for fd in 0..=2 {
            assert_eq!(
                claims.claim_descriptor(fd),
                Err(ClaimError::StandardDescriptor(fd))
            );
        }
    }

    #[test]
    fn stdin_and_stdout_are_each_claimed_once_and_apart() {
        let claims = Claims::new();

        assert_eq!(claims.claim_stdin(), Ok(()));
        assert_eq!(claims.claim_stdout(), Ok(()));
        assert_eq!(claims.claim_stdin(), Err(ClaimError::StdinTaken));
        assert_eq!(claims.claim_stdout(), Err(ClaimError::StdoutTaken));
    }

    #[test]
    fn a_taken_descriptor_names_itself() {
        let message = ClaimError::DescriptorTaken(9).to_string();

        assert!(message.contains('9'));
        assert!(message.contains("already"));
    }
}
