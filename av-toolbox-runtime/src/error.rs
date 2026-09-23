use av_toolbox_core::StageError;
use av_toolbox_graph::NodeId;
use thiserror::Error;

/// A stage failure, naming the node it came from.
#[derive(Debug, Error)]
pub enum RunError {
    #[error("node {node:?} failed: {source}")]
    Stage {
        node: NodeId,
        #[source]
        source: StageError,
    },
    /// A node's feeds disagreed about how many items they supplied.
    ///
    /// One feed short of the others is not a guess to make, it means the graph does not mean
    /// what its author thought.
    #[error("node {node:?} has feeds that disagree on how many items they supply: {counts:?}")]
    MismatchedFeeds { node: NodeId, counts: Vec<usize> },
    /// A thread of the run panicked.
    ///
    /// The panic is caught on the thread it happened on and reported here, so the run stops and
    /// joins rather than the panic crossing into whoever called it.
    #[error("a thread of the run panicked: {message}")]
    Panicked { message: String },
    /// An exchange was given something it could not make sense of.
    ///
    /// Two inputs of a join carrying different frames, or a chunk id going backwards, are things
    /// the data did rather than bugs in the driver, so they fail the run instead of crashing it.
    #[error("an exchange refused what it was given: {source}")]
    Exchange {
        #[source]
        source: ExchangeError,
    },
    /// A source waited its whole deadline for room on its frame budget without any appearing.
    ///
    /// Nothing but a frame being dropped can free room, so a wait that long is one nothing is
    /// going to end. The run fails naming the limit and what was outstanding, rather than hanging.
    #[error(
        "a source waited for room under a {limit} byte frame budget with {outstanding} bytes outstanding"
    )]
    BudgetStalled { limit: usize, outstanding: usize },
    /// A run finished with items it never gave to a worker.
    ///
    /// Frames going nowhere is the hardest failure in a pipeline to notice afterwards, so a run
    /// that was not cancelled reports it rather than answering with a report nobody reads.
    #[error("the run finished with {dropped} items that never reached a worker")]
    DroppedItems { dropped: usize },
    /// A failure that happened while a worker was running one chunk.
    ///
    /// Which chunk is what localises a failure in a long encode, so a worker names it around
    /// whatever its stages reported.
    #[error("chunk {chunk}: {source}")]
    InChunk {
        chunk: u64,
        #[source]
        source: Box<RunError>,
    },
}

impl RunError {
    /// The node whose stage failed, where one can be named.
    pub fn node(&self) -> Option<NodeId> {
        match self {
            Self::Stage { node, .. } => Some(*node),
            Self::MismatchedFeeds { node, .. } => Some(*node),
            Self::InChunk { source, .. } => source.node(),
            Self::Panicked { .. }
            | Self::Exchange { .. }
            | Self::BudgetStalled { .. }
            | Self::DroppedItems { .. } => None,
        }
    }

    /// The chunk the failure came from, where one is known.
    pub fn chunk(&self) -> Option<u64> {
        match self {
            Self::InChunk { chunk, .. } => Some(*chunk),
            _ => None,
        }
    }

    /// This failure, said to have come from `chunk`.
    ///
    /// One already naming a chunk keeps the one it has, since that is the closer of the two to
    /// where it happened.
    pub fn in_chunk(self, chunk: u64) -> Self {
        if matches!(self, Self::InChunk { .. }) {
            return self;
        }

        Self::InChunk {
            chunk,
            source: Box::new(self),
        }
    }
}

/// Something an exchange could not make sense of.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ExchangeError {
    /// Two inputs of one stage disagreed about which frame they were carrying.
    #[error("input slot {slot} carried frame {found} while slot {expected_slot} carried frame {expected}")]
    OutOfStep {
        slot: usize,
        found: u64,
        expected_slot: usize,
        expected: u64,
    },
    /// A chunk id went backwards along a stream.
    #[error("chunk {found} followed chunk {previous}, and a chunk id never goes backwards")]
    ChunkWentBackwards { previous: u64, found: u64 },
    /// An aggregate was given something from a worker it was not reading.
    #[error("worker {found} answered while worker {expected} held unit {unit}")]
    WrongWorker {
        unit: u64,
        expected: usize,
        found: usize,
    },
    /// A slot was offered a second item before the bundle holding its first one completed.
    #[error("slot {slot} still held frame {held} when frame {offered} arrived for it")]
    SlotRanAhead { slot: usize, held: u64, offered: u64 },
    /// A slot index does not exist for this disperse.
    #[error("slot {slot} is out of range for a disperse with {slots} slots")]
    SlotOutOfRange { slot: usize, slots: usize },
    /// An item arrived for a slot the signature never bound.
    #[error("slot {slot} was offered an item but no input was ever bound to it")]
    UnboundSlot { slot: usize },
    /// A caller offered another item before placing the bundle a chunk boundary is holding.
    #[error("a completed bundle is still waiting to be placed with a worker")]
    PlacementPending,
    /// A caller offered an item after the disperse had already finished.
    #[error("an item arrived after the disperse had already finished")]
    AlreadyFinished,
    /// An aggregate was asked to accept something before it had any assignment to read against.
    #[error("worker {worker} answered but the aggregate has no assignment to read against")]
    NothingAssigned { worker: usize },
    /// A worker's `EndOfUnit` named a different unit than the assignment it is draining.
    #[error("worker {worker} closed unit {found} while draining unit {expected}")]
    UnitMismatch {
        worker: usize,
        expected: u64,
        found: u64,
    },
    /// A worker ended its stream while still owing the unit it was assigned.
    #[error("worker {worker} ended its stream while still owing unit {unit}")]
    WorkerEndedOwingUnit { worker: usize, unit: u64 },
}
