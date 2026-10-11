use std::ffi::{CStr, c_int};

use crate::sys;

const MESSAGE_BUFFER_LEN: usize = 1024;

/// A failure reported by Vship, or a call refused before reaching it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct VshipError {
    code: u32,
    message: String,
}

impl VshipError {
    /// An error found in Rust before calling Vship, with code `0`.
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            code: 0,
            message: message.into(),
        }
    }

    /// The `Vship_Exception` value of the failed call, or `0` for an error found in Rust.
    pub fn code(&self) -> u32 {
        self.code
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    fn last(code: sys::Vship_Exception) -> Self {
        let mut buffer = [0u8; MESSAGE_BUFFER_LEN];
        let buffer_len = c_int::try_from(buffer.len()).unwrap_or(c_int::MAX);
        // SAFETY: The buffer is writable for `buffer_len` bytes.
        unsafe { sys::Vship_GetDetailedLastError(buffer.as_mut_ptr().cast(), buffer_len) };

        Self::from_buffer(code, &buffer)
    }

    fn last_for_handler(handle: sys::Vship_Handle, code: sys::Vship_Exception) -> Self {
        let mut buffer = [0u8; MESSAGE_BUFFER_LEN];
        let buffer_len = c_int::try_from(buffer.len()).unwrap_or(c_int::MAX);
        // SAFETY: The buffer is writable for `buffer_len` bytes and `handle` is a live handler.
        let written =
            unsafe { sys::Vship_GetDetailedLastErrorHandler(handle, buffer.as_mut_ptr().cast(), buffer_len) };
        if written == 0 {
            return Self::last(code);
        }

        Self::from_buffer(code, &buffer)
    }

    fn from_buffer(code: sys::Vship_Exception, buffer: &[u8]) -> Self {
        let text = message_from_buffer(buffer);

        if text.is_empty() {
            return Self {
                code,
                message: format!("vship error {code}"),
            };
        }

        Self { code, message: text }
    }
}

/// Turns a Vship status into a result, reading Vship's last message on failure.
pub(crate) fn check(status: sys::Vship_Exception) -> Result<(), VshipError> {
    if status == sys::Vship_NoError {
        return Ok(());
    }

    Err(VshipError::last(status))
}

/// Turns a Vship status into a result, reading the message recorded on `handle` on failure.
///
/// Unlike [check], two handlers failing on different threads do not read each other's message.
pub(crate) fn check_handler(
    handle: sys::Vship_Handle,
    status: sys::Vship_Exception,
) -> Result<(), VshipError> {
    if status == sys::Vship_NoError {
        return Ok(());
    }

    Err(VshipError::last_for_handler(handle, status))
}

/// Reads a message up to the first NUL byte, replacing invalid UTF-8.
fn message_from_buffer(buffer: &[u8]) -> String {
    match CStr::from_bytes_until_nul(buffer) {
        Ok(text) => text.to_string_lossy().into_owned(),
        Err(_) => String::from_utf8_lossy(buffer).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_message_stops_at_the_first_nul() {
        let message = message_from_buffer(b"bad device\0junk");

        assert_eq!(message, "bad device");
    }

    #[test]
    fn a_buffer_with_no_nul_converts_whole() {
        let message = message_from_buffer(b"no terminator");

        assert_eq!(message, "no terminator");
    }

    #[test]
    fn invalid_utf8_converts_lossily() {
        let message = message_from_buffer(&[b'b', 0xff, b'd', 0]);

        assert_eq!(message, "b\u{fffd}d");
    }

    #[test]
    fn a_rust_side_error_has_code_zero() {
        let error = VshipError::new("refused");

        assert_eq!(error.code(), 0);
        assert_eq!(error.message(), "refused");
    }

    #[test]
    fn success_is_not_an_error() {
        let result = check(sys::Vship_NoError);

        assert!(result.is_ok());
    }
}
