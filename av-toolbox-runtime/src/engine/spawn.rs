use std::any::Any;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use super::aggregate::{run_aggregate, run_relay};
use super::disperse::drive_disperse;
use super::region::ExchangedRegion;
use super::worker::run_worker;
use crate::{Cancel, RunError};

/// The first failure a run's threads reported, and the signal that stops the others.
///
/// The error is recorded before the stop goes out, so a thread the stop wakes cannot report a
/// failure of its own that beats the one which caused it.
pub(super) struct Outcome {
    pub(super) cancel: Arc<Cancel>,
    pub(super) error: Mutex<Option<RunError>>,
}

impl Outcome {
    pub(super) fn report(&self, error: RunError) {
        {
            let mut first = self.error.lock().expect("not poisoned");
            if first.is_none() {
                *first = Some(error);
            }
        }

        self.cancel.stop();
    }

    pub(super) fn take_error(&self) -> Option<RunError> {
        self.error.lock().expect("not poisoned").take()
    }
}

/// Spawns one of a run's threads, so whatever it ends in stops the rest of them.
///
/// A panic is caught here, on the thread it happened on, and reported as a failure of the run.
/// Letting it travel to the join instead would panic the caller part way through joining and
/// leave every thread after it unjoined.
pub(super) fn spawn<Body>(handles: &mut Vec<JoinHandle<()>>, outcome: &Arc<Outcome>, body: Body)
where
    Body: FnOnce() -> Result<(), RunError> + Send + 'static,
{
    let outcome = outcome.clone();

    handles.push(thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(body));

        match result {
            Ok(Ok(())) => {},
            Ok(Err(error)) => outcome.report(error),
            Err(payload) => outcome.report(RunError::Panicked {
                message: panic_message(payload),
            }),
        }
    }));
}

/// Whatever a panic was raised with, as text.
fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }

    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }

    "no message".to_string()
}

pub(super) fn spawn_exchanged(
    region: ExchangedRegion,
    handles: &mut Vec<JoinHandle<()>>,
    outcome: &Arc<Outcome>,
    cancel: &Arc<Cancel>,
    dropped: &Arc<AtomicUsize>,
) {
    let ExchangedRegion {
        disperse,
        freedom,
        aggregate,
        result_receivers,
        workers,
        downstream,
    } = region;

    let disperse_cancel = cancel.clone();
    let disperse_dropped = dropped.clone();
    spawn(handles, outcome, move || {
        drive_disperse(disperse, &disperse_cancel, &disperse_dropped)
    });

    let merge_cancel = cancel.clone();
    spawn(handles, outcome, move || match aggregate {
        Some(side) => run_aggregate(side, result_receivers, &freedom, &merge_cancel, downstream),
        None => run_relay(result_receivers, &freedom, downstream),
    });

    for worker in workers {
        spawn(handles, outcome, move || run_worker(worker));
    }
}
