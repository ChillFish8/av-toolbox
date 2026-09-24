use std::ffi::{CStr, c_char, c_int};
use std::{fmt, ptr};

use ffms2_sys::FFMS_ErrorInfo;

const BUFFER_SIZE: usize = 1024;

/// Room for FFMS2 to describe why a call failed.
pub(super) struct ErrorBuffer {
    message: [u8; BUFFER_SIZE],
    info: FFMS_ErrorInfo,
}

impl ErrorBuffer {
    pub(super) fn new() -> Self {
        let info = FFMS_ErrorInfo {
            ErrorType: 0,
            SubType: 0,
            BufferSize: 0,
            Buffer: ptr::null_mut(),
        };

        Self {
            message: [0; BUFFER_SIZE],
            info,
        }
    }

    /// The error info to pass to an FFMS2 call.
    ///
    /// The pointer is set on every call, so it is only valid until this buffer next moves.
    pub(super) fn info(&mut self) -> *mut FFMS_ErrorInfo {
        let buffer_size = self.message.len();

        self.info.BufferSize = buffer_size as c_int;
        self.info.Buffer = self.message.as_mut_ptr().cast::<c_char>();

        &raw mut self.info
    }

    pub(super) fn into_error(self) -> Ffms2Error {
        let message = message_from(&self.message);

        Ffms2Error {
            kind: self.info.ErrorType,
            sub_kind: self.info.SubType,
            message,
        }
    }

    /// Turns a status FFMS2 returned into a result, where `0` is success.
    pub(super) fn check(self, status: c_int) -> Result<(), Ffms2Error> {
        if status == 0 {
            return Ok(());
        }

        Err(self.into_error())
    }
}

/// The text FFMS2 wrote, up to its NUL or the end of the buffer.
fn message_from(bytes: &[u8]) -> String {
    let terminated = CStr::from_bytes_until_nul(bytes);
    let text = match terminated {
        Ok(text) => text.to_bytes(),
        Err(_) => bytes,
    };
    let message = String::from_utf8_lossy(text);

    message.into_owned()
}

/// Why an FFMS2 call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Ffms2Error {
    kind: i32,
    sub_kind: i32,
    message: String,
}

impl Ffms2Error {
    /// An error raised before a call reached FFMS2.
    pub(crate) fn local(message: String) -> Self {
        Self {
            kind: 0,
            sub_kind: 0,
            message,
        }
    }
}

impl fmt::Display for Ffms2Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.message.is_empty() {
            return write!(formatter, "ffms2 error {}/{}", self.kind, self.sub_kind);
        }

        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Ffms2Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_info_reports_the_whole_buffer_size() {
        let mut errors = ErrorBuffer::new();

        let _info = errors.info();

        assert_eq!(errors.info.BufferSize, 1024);
    }

    #[test]
    fn a_long_message_converts_whole() {
        let mut bytes = [0u8; 1024];
        let text = b"Can't open '/nonexistent/missing.mkv'";
        bytes[..text.len()].copy_from_slice(text);

        let message = message_from(&bytes);

        assert_eq!(message, "Can't open '/nonexistent/missing.mkv'");
    }

    #[test]
    fn a_buffer_with_no_nul_converts_whole() {
        let bytes = [b'a'; 1024];

        let message = message_from(&bytes);

        assert_eq!(message.len(), 1024);
    }

    #[test]
    fn invalid_utf8_converts_lossily() {
        let bytes = [b'o', 0x80, b'k', 0];

        let message = message_from(&bytes);

        assert_eq!(message, "o\u{FFFD}k");
    }

    #[test]
    fn an_empty_message_displays_its_kinds() {
        let error = Ffms2Error {
            kind: 7,
            sub_kind: 22,
            message: String::new(),
        };

        assert_eq!(error.to_string(), "ffms2 error 7/22");
    }

    #[test]
    fn a_message_displays_as_itself() {
        let error = Ffms2Error::local("no such file".to_string());

        assert_eq!(error.to_string(), "no such file");
    }
}
