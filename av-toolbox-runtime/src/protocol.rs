use av_toolbox_gpu::StreamItem;

/// What a disperse sends one of its workers.
#[derive(Debug)]
pub enum WorkItem {
    /// One slot per input of the region's head node, paired by frame index.
    ///
    /// A slot is empty where an optional input was never bound.
    Bundle(Vec<Option<StreamItem>>),
    /// Every item of this chunk has been sent.
    EndOfChunk(u64),
    /// Every chunk has been sent.
    EndOfStream,
}

/// What a worker sends its aggregate.
#[derive(Debug)]
pub enum WorkResult {
    Item(StreamItem),
    /// Everything this worker had for the chunk has been sent, which may be nothing.
    EndOfChunk(u64),
    EndOfStream,
}

/// Which worker a disperse gave a chunk to.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Assignment {
    pub chunk: u64,
    pub worker: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_assignment_names_a_chunk_and_the_worker_that_holds_it() {
        let assignment = Assignment { chunk: 4, worker: 2 };

        assert_eq!(assignment.chunk, 4);
        assert_eq!(assignment.worker, 2);
    }
}
