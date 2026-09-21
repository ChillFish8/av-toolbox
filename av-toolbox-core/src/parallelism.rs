use std::fmt;

/// The smallest unit of a stream a stage can be given on its own.
///
/// A stage that can take a lone frame is equally happy given a whole chunk, or the whole
/// stream in order, so these order by how much a stage demands rather than what it prefers.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum Parallelism {
    Frame,
    Chunk,
    Stream,
}

impl fmt::Display for Parallelism {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Frame => "frame",
            Self::Chunk => "chunk",
            Self::Stream => "stream",
        };

        formatter.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_granularity_orders_by_how_much_it_demands() {
        assert!(Parallelism::Frame < Parallelism::Chunk);
        assert!(Parallelism::Chunk < Parallelism::Stream);
    }

    #[test]
    fn a_granularity_renders_as_a_word() {
        assert_eq!(Parallelism::Frame.to_string(), "frame");
        assert_eq!(Parallelism::Chunk.to_string(), "chunk");
        assert_eq!(Parallelism::Stream.to_string(), "stream");
    }
}
