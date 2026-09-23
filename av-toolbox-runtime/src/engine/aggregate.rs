use std::sync::mpsc::Receiver;

use super::channel::{AggregateSignal, DownstreamLink, forward, send_done};
use super::freedom::WorkerFreedom;
use super::region::AggregateSide;
use crate::{Cancel, ExchangeError, RunError, WorkResult};

/// Carries one worker's output straight downstream, for a region with no aggregate.
///
/// A region of one worker has nothing to merge back into order, so nothing here renumbers what
/// it produced. Marking the worker free as each unit closes is what lets its disperse hand it
/// the next one, and is the one job this shares with an aggregate.
pub(super) fn run_relay(
    result_receivers: Vec<Receiver<WorkResult>>,
    freedom: &WorkerFreedom,
    downstream: Vec<DownstreamLink>,
) -> Result<(), RunError> {
    let results = result_receivers
        .into_iter()
        .next()
        .expect("a region without an aggregate runs exactly one worker");

    while let Ok(result) = results.recv() {
        match result {
            WorkResult::Item(item) => forward(&downstream, vec![item]),
            WorkResult::EndOfUnit(_) => freedom.mark_free(0),
            WorkResult::EndOfStream => break,
        }
    }

    send_done(&downstream);
    Ok(())
}

/// Merges what a region's workers produced back into one stream, in assignment order.
///
/// A worker that ends still owing a chunk is only tolerated once the run has stopped, which means
/// something else has already failed and reported it. On a run that is still going it means the
/// exchange itself has lost a chunk, and that is reported rather than quietly finishing with
/// frames missing.
pub(crate) fn run_aggregate(
    side: AggregateSide,
    result_receivers: Vec<Receiver<WorkResult>>,
    freedom: &WorkerFreedom,
    cancel: &Cancel,
    downstream: Vec<DownstreamLink>,
) -> Result<(), RunError> {
    let AggregateSide {
        mut aggregate,
        assignment_receiver,
    } = side;

    loop {
        match aggregate.next_worker() {
            Some(worker) => {
                let Ok(result) = result_receivers[worker].recv() else {
                    break;
                };
                let is_end_of_unit = matches!(result, WorkResult::EndOfUnit(_));

                let accepted = aggregate.accept(worker, result);
                let emitted = match accepted {
                    Ok(emitted) => emitted,
                    // A worker that ends still owing a chunk has either failed or been stopped,
                    // and its own thread reports that. Nothing is left to merge in order, so this
                    // winds down rather than raising a second failure derived from the first.
                    Err(ExchangeError::WorkerEndedOwingUnit { .. }) if cancel.is_stopped() => break,
                    // The stop a failure raises is recorded a moment after the thread that failed
                    // has dropped its senders, so the arm above can lose that race and land here
                    // instead. Reporting rather than panicking leaves the real cause the first
                    // error of the run and this one behind it.
                    Err(error) => return Err(RunError::Exchange { source: error }),
                };

                if is_end_of_unit {
                    freedom.mark_free(worker);
                }

                forward(&downstream, emitted);
            },
            None => {
                if aggregate.is_finished() {
                    break;
                }

                match assignment_receiver.recv() {
                    Ok(AggregateSignal::Assignment(assignment)) => aggregate.assign(assignment),
                    Ok(AggregateSignal::Finished) | Err(_) => aggregate.finish(),
                }
            },
        }
    }

    send_done(&downstream);
    Ok(())
}
