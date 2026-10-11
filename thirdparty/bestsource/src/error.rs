use std::ffi::CStr;

/// The size of the buffer the shim writes an error message into.
pub(crate) const ERROR_BUFFER_LEN: usize = 1024;

/// A failure reported by BestSource, or a call refused before reaching it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct BestSourceError {
    message: String,
}

impl BestSourceError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Reads the message the shim wrote, up to the first NUL byte.
    ///
    /// Invalid UTF-8 is replaced and a buffer with no NUL byte is read whole.
    pub(crate) fn from_buffer(buffer: &[u8]) -> Self {
        let text = match CStr::from_bytes_until_nul(buffer) {
            Ok(text) => text.to_string_lossy().into_owned(),
            Err(_) => String::from_utf8_lossy(buffer).into_owned(),
        };

        if text.is_empty() {
            return Self::new("bestsource failed without a message");
        }

        Self::new(text)
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer_with(text: &[u8]) -> [u8; ERROR_BUFFER_LEN] {
        let mut buffer = [0u8; ERROR_BUFFER_LEN];
        buffer[..text.len()].copy_from_slice(text);

        buffer
    }

    #[test]
    fn a_long_message_converts_whole() {
        let text = "Couldn't open '/media/a/rather/long/path/to/some/clip.mkv'";
        let buffer = buffer_with(text.as_bytes());

        let error = BestSourceError::from_buffer(&buffer);

        assert_eq!(error.message(), text);
    }

    #[test]
    fn a_buffer_with_no_nul_converts_whole() {
        let buffer = [b'x'; ERROR_BUFFER_LEN];

        let error = BestSourceError::from_buffer(&buffer);

        assert_eq!(error.message().len(), ERROR_BUFFER_LEN);
    }

    #[test]
    fn invalid_utf8_converts_lossily() {
        let buffer = buffer_with(&[b'b', 0xff, b'd']);

        let error = BestSourceError::from_buffer(&buffer);

        assert_eq!(error.message(), "b\u{fffd}d");
    }

    #[test]
    fn an_empty_buffer_still_gives_a_message() {
        let buffer = [0u8; ERROR_BUFFER_LEN];

        let error = BestSourceError::from_buffer(&buffer);

        assert_eq!(error.message(), "bestsource failed without a message");
    }
}
