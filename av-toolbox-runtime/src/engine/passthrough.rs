use super::channel::{ExitMessage, forward, send_done};
use super::region::PassthroughRegion;
use crate::RunError;

/// Runs a region that only carries one upstream's output forward.
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

    while let Ok((slot, ExitMessage::Item(item))) = upstream.recv() {
        let mut bundle = vec![None; slot + 1];
        bundle[slot] = Some(item);

        let produced = runner.run(&mut context, bundle)?;
        forward(&downstream, produced);
    }

    let flushed = runner.end(&mut context)?;
    forward(&downstream, flushed);
    send_done(&downstream);

    Ok(())
}
