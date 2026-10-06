use std::fmt;

use crate::AnyDemand;

/// How small a piece of the stream a stage may be given on its own.
///
/// A stage that can take a lone frame is equally happy given a whole chunk, or the whole
/// stream in order, so these order by how much a stage demands rather than what it prefers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum Granularity {
    Frame,
    Chunk,
    Stream,
}

impl fmt::Display for Granularity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Frame => "frame",
            Self::Chunk => "chunk",
            Self::Stream => "stream",
        };

        formatter.write_str(name)
    }
}

/// The smallest unit of a stream a stage can be given on its own.
#[derive(Debug, Clone, PartialEq)]
pub enum Parallelism {
    Frame,
    /// One chunk at a time, chunked by the layer its demand asks for.
    ///
    /// `None` chunks the stage without asking for any layer.
    Chunk(Option<AnyDemand>),
    Stream,
}

impl Parallelism {
    pub fn granularity(&self) -> Granularity {
        match self {
            Self::Frame => Granularity::Frame,
            Self::Chunk(_) => Granularity::Chunk,
            Self::Stream => Granularity::Stream,
        }
    }

    /// The demand a chunk stage is chunked by, if it names one.
    pub fn demand(&self) -> Option<&AnyDemand> {
        match self {
            Self::Chunk(demand) => demand.as_ref(),
            Self::Frame | Self::Stream => None,
        }
    }
}

impl fmt::Display for Parallelism {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.granularity().fmt(formatter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_granularity_renders_as_a_word() {
        let chunked = Parallelism::Chunk(None);

        assert_eq!(Parallelism::Frame.to_string(), "frame");
        assert_eq!(chunked.to_string(), "chunk");
        assert_eq!(Parallelism::Stream.to_string(), "stream");
    }
}
