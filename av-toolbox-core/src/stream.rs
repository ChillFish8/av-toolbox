use std::fmt;

/// What a stage passes downstream.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum StreamType {
    Frames,
    Mask,
}

impl fmt::Display for StreamType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Frames => "frames",
            Self::Mask => "mask",
        };

        formatter.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stream_type_renders_as_a_word() {
        assert_eq!(StreamType::Frames.to_string(), "frames");
        assert_eq!(StreamType::Mask.to_string(), "mask");
    }
}
