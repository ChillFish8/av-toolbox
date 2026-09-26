use std::collections::HashSet;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Mutex, MutexGuard, PoisonError};

static PROCESS_CLAIMS: LazyLock<Claims> = LazyLock::new(Claims::new);

/// The standard streams and descriptors stages have taken ownership of.
///
/// A standard stream stays claimed for the life of the process. A descriptor is claimed for as long
/// as its [DescriptorClaim](crate::DescriptorClaim) is held.
#[derive(Debug, Default)]
pub struct Claims {
    stdin: AtomicBool,
    stdout: AtomicBool,
    descriptors: Mutex<HashSet<i32>>,
}

/// A claimed descriptor, given back when this is dropped.
///
/// Drop it only after the descriptor is closed, so no other stage can claim it while it is still open.
#[derive(Debug)]
#[must_use = "the descriptor is given back as soon as the claim is dropped"]
pub struct DescriptorClaim<'a> {
    claims: &'a Claims,
    fd: i32,
}

impl DescriptorClaim<'_> {
    pub const fn fd(&self) -> i32 {
        self.fd
    }
}

impl Drop for DescriptorClaim<'_> {
    fn drop(&mut self) {
        let mut claimed = self.claims.claimed_descriptors();
        claimed.remove(&self.fd);
    }
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

    pub fn claim_descriptor(&self, fd: i32) -> Result<DescriptorClaim<'_>, ClaimError> {
        if (0..=2).contains(&fd) {
            return Err(ClaimError::StandardDescriptor(fd));
        }

        let mut claimed = self.claimed_descriptors();
        let newly_claimed = claimed.insert(fd);

        if !newly_claimed {
            return Err(ClaimError::DescriptorTaken(fd));
        }

        Ok(DescriptorClaim { claims: self, fd })
    }

    fn claimed_descriptors(&self) -> MutexGuard<'_, HashSet<i32>> {
        self.descriptors.lock().unwrap_or_else(PoisonError::into_inner)
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

        let held = first.expect("the first claim succeeds");
        assert_eq!(held.fd(), 7);
        assert_eq!(second.err(), Some(ClaimError::DescriptorTaken(7)));
    }

    #[test]
    fn a_dropped_claim_gives_its_descriptor_back() {
        let claims = Claims::new();

        let first = claims.claim_descriptor(7).expect("the first claim succeeds");
        drop(first);
        let second = claims.claim_descriptor(7);

        assert!(second.is_ok());
    }

    #[test]
    fn dropping_one_claim_leaves_the_others_held() {
        let claims = Claims::new();

        let seven = claims.claim_descriptor(7).expect("7 is free");
        let eight = claims.claim_descriptor(8).expect("8 is free");
        drop(seven);
        let again = claims.claim_descriptor(8);

        assert_eq!(again.err(), Some(ClaimError::DescriptorTaken(8)));
        drop(eight);
    }

    #[test]
    fn the_standard_descriptors_cannot_be_claimed_as_descriptors() {
        let claims = Claims::new();

        for fd in 0..=2 {
            let refused = claims.claim_descriptor(fd);

            assert_eq!(refused.err(), Some(ClaimError::StandardDescriptor(fd)));
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
