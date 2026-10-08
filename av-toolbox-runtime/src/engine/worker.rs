use std::sync::mpsc::SyncSender;

use av_toolbox_gpu::{AnyItem, ChunkRoute};

use super::region::WorkerSlot;
use crate::chunks::chunk_in;
use crate::{RunError, WorkItem, WorkResult};

/// Runs one worker of an exchanged region, one unit at a time.
///
/// Every unit after the first is started by its own first bundle arriving rather than by the
/// unit before it closing, so a worker whose stream ends on a unit close never leaves its
/// stages holding a start they will never see the end of.
///
/// Nothing here reads the stop flag. Everything a worker waits on is a channel: its disperse
/// closes the work channel when it winds down, and its aggregate dropping the other end closes
/// the results one. Walking out part way through a unit instead would leave the aggregate
/// waiting on a close that never comes.
///
/// An aggregate that has already gone ends this the same way an `EndOfStream` does, by leaving
/// the loop, so the flush below still runs. Returning from inside the loop instead would leave
/// every stage in this worker holding a start it never sees the end of.
pub(crate) fn run_worker(worker: WorkerSlot) -> Result<(), RunError> {
    let WorkerSlot {
        mut runner,
        mut context,
        work_receiver,
        result_sender,
    } = worker;

    let route = runner.route();

    runner.start(&mut context)?;
    let mut started = true;
    let mut chunk_in_hand = None;

    loop {
        match work_receiver.recv() {
            Ok(WorkItem::Bundle(bundle)) => {
                if !started {
                    runner.start(&mut context)?;
                    started = true;
                }

                chunk_in_hand = bundle_chunk(&bundle, route);

                let run = runner.run(&mut context, bundle);
                let produced = run.map_err(|error| named_chunk(error, chunk_in_hand))?;
                if send_results(&result_sender, produced).is_err() {
                    break;
                }
            },
            Ok(WorkItem::Unit(unit, bundle)) => {
                if !started {
                    runner.start(&mut context)?;
                    started = true;
                }

                chunk_in_hand = bundle_chunk(&bundle, route);

                let run = runner.run(&mut context, bundle);
                let produced = run.map_err(|error| named_chunk(error, chunk_in_hand))?;
                if send_results(&result_sender, produced).is_err() {
                    break;
                }

                if result_sender.send(WorkResult::EndOfUnit(unit)).is_err() {
                    break;
                }
            },
            Ok(WorkItem::EndOfChunk(chunk)) => {
                chunk_in_hand = Some(chunk);

                if started {
                    let end = runner.end(&mut context, Some(chunk));
                    let flushed = end.map_err(|error| error.in_chunk(chunk))?;
                    started = false;

                    if send_results(&result_sender, flushed).is_err() {
                        break;
                    }
                }

                if result_sender.send(WorkResult::EndOfUnit(chunk)).is_err() {
                    break;
                }
            },
            Ok(WorkItem::EndOfStream) | Err(_) => break,
        }
    }

    if started {
        let end = runner.end(&mut context, chunk_in_hand);
        let flushed = end.map_err(|error| named_chunk(error, chunk_in_hand))?;
        send_results(&result_sender, flushed).ok();
    }

    result_sender.send(WorkResult::EndOfStream).ok();
    Ok(())
}

/// The chunk a bundle belongs to on `route`, taken from the first slot carrying an item.
fn bundle_chunk(bundle: &[Option<AnyItem>], route: Option<ChunkRoute>) -> Option<u64> {
    let item = bundle.iter().flatten().next()?;

    Some(chunk_in(item.meta(), route))
}

/// `error`, said to have come from `chunk` when there is one to name.
fn named_chunk(error: RunError, chunk: Option<u64>) -> RunError {
    match chunk {
        Some(chunk) => error.in_chunk(chunk),
        None => error,
    }
}

fn send_results(result_sender: &SyncSender<WorkResult>, items: Vec<AnyItem>) -> Result<(), ()> {
    for item in items {
        result_sender.send(WorkResult::Item(item)).map_err(|_| ())?;
    }

    Ok(())
}
