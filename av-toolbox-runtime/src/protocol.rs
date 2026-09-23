use av_toolbox_gpu::AnyItem;

/// What a disperse sends one of its workers.
#[derive(Debug)]
pub enum WorkItem {
    /// One slot per input of the region's head node, paired by frame index.
    ///
    /// A slot is empty where an optional input was never bound.
    Bundle(Vec<Option<AnyItem>>),
    /// A bundle that is a whole unit of work on its own, carrying the id its aggregate expects.
    ///
    /// Frame routing sends these. The worker runs the bundle and closes the unit without
    /// flushing its region, since a frame mode stage has nothing to carry between frames.
    Unit(u64, Vec<Option<AnyItem>>),
    /// Every item of this chunk has been sent.
    EndOfChunk(u64),
    /// Every chunk has been sent.
    EndOfStream,
}

/// What a worker sends its aggregate.
#[derive(Debug)]
pub enum WorkResult {
    Item(AnyItem),
    /// Everything this worker had for the unit has been sent, which may be nothing.
    EndOfUnit(u64),
    EndOfStream,
}

/// Which worker a disperse gave one unit of work to.
///
/// A unit is a chunk under chunk routing and a single bundle under frame routing.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Assignment {
    pub unit: u64,
    pub worker: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_assignment_names_a_unit_and_the_worker_that_holds_it() {
        let assignment = Assignment { unit: 4, worker: 2 };

        assert_eq!(assignment.unit, 4);
        assert_eq!(assignment.worker, 2);
    }
}
