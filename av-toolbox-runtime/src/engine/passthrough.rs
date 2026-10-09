use super::channel::{ExitMessage, forward, send_done};
use super::region::PassthroughRegion;
use crate::RunError;

/// Runs a region that only carries one upstream's output forward.
///
/// A record from upstream is held by the region and leaves ahead of the head's next output, or after
/// its end flush.
///
/// Nothing here reads the stop flag. Its one wait is a receive, which ends as soon as the region
/// feeding it stops and drops its sender, and walking out on the flag instead would throw away
/// whatever had already been handed over without anything accounting for it.
pub(super) fn run_passthrough(region: PassthroughRegion) -> Result<(), RunError> {
    let PassthroughRegion {
        mut runner,
        mut context,
        upstream,
        downstream,
    } = region;

    runner.start(&mut context)?;

    while let Ok((slot, message)) = upstream.recv() {
        match message {
            ExitMessage::Item(item) => {
                let mut bundle = vec![None; slot + 1];
                bundle[slot] = Some(item);

                let produced = runner.run(&mut context, bundle)?;
                forward(&downstream, produced);
            },
            ExitMessage::Record(record) => runner.offer_records(vec![record]),
            ExitMessage::Done => break,
        }
    }

    let flushed = runner.end(&mut context, None)?;
    forward(&downstream, flushed);
    send_done(&downstream);

    Ok(())
}
