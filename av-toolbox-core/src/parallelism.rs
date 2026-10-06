use std::fmt;

use av_toolbox_gpu::{ChunkKeyFn, ProvidedValue};

use crate::{AnyDemand, ChunkKey, DemandDefinition};

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
#[derive(Debug, Clone)]
pub enum Parallelism {
    Frame,
    /// One chunk at a time, chunked by the value its demand is served with.
    Chunk {
        demand: AnyDemand,
        key: ChunkKeyFn,
    },
    Stream,
}

/// Two chunk parallelisms are equal when they demand the same thing, as the demand fixes the key.
impl PartialEq for Parallelism {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Frame, Self::Frame) | (Self::Stream, Self::Stream) => true,
            (
                Self::Chunk { demand, .. },
                Self::Chunk {
                    demand: other_demand, ..
                },
            ) => demand == other_demand,
            _ => false,
        }
    }
}

/// Reads the chunk out of a value of `D`'s output, or 0 when the value is of another type.
fn key_of<D: DemandDefinition>(value: &dyn ProvidedValue) -> u64
where
    D::Output: ChunkKey,
{
    match value.as_any().downcast_ref::<D::Output>() {
        Some(output) => output.chunk(),
        None => 0,
    }
}

impl Parallelism {
    /// Chunk parallelism by `D`'s output for `request`.
    pub fn chunk<D: DemandDefinition>(request: D::Request) -> Self
    where
        D::Output: ChunkKey,
    {
        Self::Chunk {
            demand: AnyDemand::new::<D>(request),
            key: key_of::<D>,
        }
    }

    pub fn granularity(&self) -> Granularity {
        match self {
            Self::Frame => Granularity::Frame,
            Self::Chunk { .. } => Granularity::Chunk,
            Self::Stream => Granularity::Stream,
        }
    }

    /// The demand a chunk stage is chunked by.
    pub fn demand(&self) -> Option<&AnyDemand> {
        match self {
            Self::Chunk { demand, .. } => Some(demand),
            Self::Frame | Self::Stream => None,
        }
    }

    /// How a chunk stage reads the chunk out of the value its demand is served with.
    pub fn chunk_key(&self) -> Option<ChunkKeyFn> {
        match self {
            Self::Chunk { key, .. } => Some(*key),
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
    use crate::testing::FixtureCuts;

    #[test]
    fn a_granularity_renders_as_a_word() {
        let chunked = Parallelism::chunk::<FixtureCuts>(());

        assert_eq!(Parallelism::Frame.to_string(), "frame");
        assert_eq!(chunked.to_string(), "chunk");
        assert_eq!(Parallelism::Stream.to_string(), "stream");
    }
}
