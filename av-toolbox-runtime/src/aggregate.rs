use av_toolbox_gpu::AnyItem;

use crate::{Assignment, ExchangeError, WorkResult};

/// Merges workers' outputs back into one stream, in the order the disperse sent them out.
///
/// It reads from the specific worker holding the next unit, never whichever speaks first, and
/// renumbers every item it emits densely from zero as it goes.
pub struct Aggregate {
    assignments: Vec<Assignment>,
    position: usize,
    next_index: u64,
    finished: bool,
}

impl Aggregate {
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
            position: 0,
            next_index: 0,
            finished: false,
        }
    }

    /// Records which worker holds a unit, in the order the disperse assigned them.
    pub fn assign(&mut self, assignment: Assignment) {
        self.assignments.push(assignment);
    }

    /// Tells the aggregate no further assignments are coming.
    pub fn finish(&mut self) {
        self.finished = true;
    }

    /// Whether every assignment has drained and no more are coming.
    ///
    /// `next_worker` returning `None` is not enough on its own to mean this, since assignments can
    /// still lag behind the disperse that is making them.
    pub fn is_finished(&self) -> bool {
        self.finished && self.position >= self.assignments.len()
    }

    /// The worker the caller should take from next, or `None` once every assigned unit is drained.
    pub fn next_worker(&self) -> Option<usize> {
        self.assignments
            .get(self.position)
            .map(|assignment| assignment.worker)
    }

    /// Takes one message from `worker` and returns whatever is now ready to emit.
    ///
    /// Only the worker holding the current unit may answer with an item or close it. A unit
    /// that produced nothing still moves the aggregate on to the next assignment once its
    /// `EndOfUnit` arrives. A worker's `EndOfStream` means that worker has nothing left to send,
    /// not that the merged stream itself is over, so it neither emits anything nor moves the
    /// aggregate on to the next assignment, unless it is the worker holding the current, undrained
    /// unit, which means it ended without ever sending that unit's close.
    pub fn accept(&mut self, worker: usize, result: WorkResult) -> Result<Vec<AnyItem>, ExchangeError> {
        match result {
            WorkResult::Item(mut item) => {
                self.check_worker(worker)?;
                item.set_index(self.next_index);
                self.next_index += 1;
                Ok(vec![item])
            },
            WorkResult::EndOfUnit(unit) => {
                let current = self.check_worker(worker)?;

                if unit != current.unit {
                    return Err(ExchangeError::UnitMismatch {
                        worker,
                        expected: current.unit,
                        found: unit,
                    });
                }

                self.position += 1;
                Ok(Vec::new())
            },
            WorkResult::EndOfStream => {
                let owing = self
                    .assignments
                    .get(self.position)
                    .is_some_and(|current| current.worker == worker);

                if owing {
                    let unit = self.assignments[self.position].unit;
                    return Err(ExchangeError::WorkerEndedOwingUnit { worker, unit });
                }

                Ok(Vec::new())
            },
        }
    }

    fn check_worker(&self, worker: usize) -> Result<Assignment, ExchangeError> {
        let Some(current) = self.assignments.get(self.position) else {
            return Err(ExchangeError::NothingAssigned { worker });
        };

        if current.worker != worker {
            return Err(ExchangeError::WrongWorker {
                unit: current.unit,
                expected: current.worker,
                found: worker,
            });
        }

        Ok(*current)
    }
}

impl Default for Aggregate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_gpu::{
        ColourInfo,
        FrameFormat,
        FrameLayout,
        FrameMeta,
        SampleFormat,
        StreamItem,
        Subsampling,
        testing_context,
    };

    use super::*;

    fn format() -> FrameFormat {
        FrameFormat {
            width: 64,
            height: 64,
            layout: FrameLayout::Yuv {
                subsampling: Subsampling::Yuv420,
            },
            sample: SampleFormat::U8,
            colour: ColourInfo::default(),
            frame_rate: None,
            field_order: None,
        }
    }

    fn item(index: u64, chunk: u64) -> AnyItem {
        let context = testing_context();
        let meta = FrameMeta::at(index).in_chunk(chunk);
        StreamItem::Frame(context.alloc_frame(&format(), meta)).into()
    }

    #[test]
    fn it_reads_the_worker_the_assignment_names() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 2 });

        assert_eq!(aggregate.next_worker(), Some(2));
    }

    #[test]
    fn it_has_nothing_to_read_before_it_is_told_anything() {
        let aggregate = Aggregate::new();

        assert_eq!(aggregate.next_worker(), None);
    }

    #[test]
    fn output_is_renumbered_densely_from_zero() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        let first = aggregate
            .accept(0, WorkResult::Item(item(40, 7)))
            .expect("should accept");
        let second = aggregate
            .accept(0, WorkResult::Item(item(41, 7)))
            .expect("should accept");

        assert_eq!(first[0].meta().index, 0);
        assert_eq!(second[0].meta().index, 1);
    }

    #[test]
    fn renumbering_leaves_the_chunk_id_alone() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        let emitted = aggregate
            .accept(0, WorkResult::Item(item(40, 7)))
            .expect("should accept");

        assert_eq!(emitted[0].meta().chunk, 7);
    }

    #[test]
    fn a_unit_that_produced_nothing_still_moves_it_on() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });
        aggregate.assign(Assignment { unit: 8, worker: 1 });

        let emitted = aggregate
            .accept(0, WorkResult::EndOfUnit(7))
            .expect("should accept");

        assert!(emitted.is_empty());
        assert_eq!(aggregate.next_worker(), Some(1));
    }

    #[test]
    fn numbering_carries_across_a_unit_boundary() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });
        aggregate.assign(Assignment { unit: 8, worker: 1 });

        aggregate
            .accept(0, WorkResult::Item(item(40, 7)))
            .expect("should accept");
        aggregate
            .accept(0, WorkResult::EndOfUnit(7))
            .expect("should accept");
        let emitted = aggregate
            .accept(1, WorkResult::Item(item(90, 8)))
            .expect("should accept");

        assert_eq!(emitted[0].meta().index, 1);
    }

    #[test]
    fn a_worker_speaking_out_of_turn_is_an_error() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });
        aggregate.assign(Assignment { unit: 8, worker: 1 });

        let result = aggregate.accept(1, WorkResult::Item(item(90, 8)));

        let Err(error) = result else {
            panic!("the aggregate should have refused a worker speaking out of turn");
        };

        assert_eq!(
            error,
            ExchangeError::WrongWorker {
                unit: 7,
                expected: 0,
                found: 1,
            }
        );
    }

    #[test]
    fn it_runs_out_of_workers_to_read_once_every_unit_is_drained() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        aggregate
            .accept(0, WorkResult::EndOfUnit(7))
            .expect("should accept");

        assert_eq!(aggregate.next_worker(), None);
    }

    /// Worker 1 is not the worker holding the current, undrained unit, so its own end of stream
    /// is harmless and must not move the aggregate on or end the merge early.
    #[test]
    fn a_workers_end_of_stream_does_not_end_the_merged_stream_early() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });
        aggregate.assign(Assignment { unit: 8, worker: 1 });

        let emitted = aggregate
            .accept(1, WorkResult::EndOfStream)
            .expect("should accept");

        assert!(emitted.is_empty());
        assert_eq!(aggregate.next_worker(), Some(0));
    }

    #[test]
    fn a_worker_ending_while_it_still_owes_a_unit_is_an_error() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        let result = aggregate.accept(0, WorkResult::EndOfStream);

        let Err(error) = result else {
            panic!("the aggregate should have refused a worker that ended while owing its unit");
        };

        assert_eq!(error, ExchangeError::WorkerEndedOwingUnit { worker: 0, unit: 7 });
    }

    #[test]
    fn an_end_of_unit_naming_the_wrong_unit_is_an_error() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        let result = aggregate.accept(0, WorkResult::EndOfUnit(9));

        let Err(error) = result else {
            panic!("the aggregate should have refused an end of unit naming the wrong unit");
        };

        assert_eq!(
            error,
            ExchangeError::UnitMismatch {
                worker: 0,
                expected: 7,
                found: 9,
            }
        );
    }

    #[test]
    fn accepting_before_anything_is_assigned_is_an_error() {
        let mut aggregate = Aggregate::new();

        let result = aggregate.accept(0, WorkResult::Item(item(0, 0)));

        let Err(error) = result else {
            panic!("the aggregate should have refused an item with nothing assigned to read against");
        };

        assert_eq!(error, ExchangeError::NothingAssigned { worker: 0 });
    }

    #[test]
    fn a_finished_aggregate_says_so() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });
        aggregate
            .accept(0, WorkResult::EndOfUnit(7))
            .expect("should accept");

        aggregate.finish();

        assert!(aggregate.is_finished());
    }

    #[test]
    fn an_aggregate_merely_waiting_is_not_finished() {
        let aggregate = Aggregate::new();

        assert!(!aggregate.is_finished());
    }

    #[test]
    fn finishing_with_undrained_assignments_is_not_finished_yet() {
        let mut aggregate = Aggregate::new();
        aggregate.assign(Assignment { unit: 7, worker: 0 });

        aggregate.finish();

        assert!(!aggregate.is_finished());
    }
}
