use thiserror::Error;

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
    #[error("worker {found} answered while worker {expected} held chunk {chunk}")]
    WrongWorker {
        chunk: u64,
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
    /// A worker's `EndOfChunk` named a different chunk than the assignment it is draining.
    #[error("worker {worker} closed chunk {found} while draining chunk {expected}")]
    ChunkMismatch {
        worker: usize,
        expected: u64,
        found: u64,
    },
    /// A worker ended its stream while still owing the chunk it was assigned.
    #[error("worker {worker} ended its stream while still owing chunk {chunk}")]
    WorkerEndedOwingChunk { worker: usize, chunk: u64 },
}
