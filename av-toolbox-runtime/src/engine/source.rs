use std::sync::atomic::Ordering;

use av_toolbox_gpu::FrameBudget;

use super::channel::{forward, send_done};
use super::region::SourceRegion;
use crate::{Cancel, Flow, RunError};

pub(super) fn run_source(region: SourceRegion, cancel: &Cancel) -> Result<(), RunError> {
    let SourceRegion {
        mut runner,
        mut context,
        budget,
        exhausted,
        downstream,
    } = region;

    runner.start(&mut context)?;

    loop {
        let flows = runner.run(&mut context, Vec::new())?;
        let produced = flows.into_iter().filter_map(Flow::into_item).collect();
        forward(&downstream, produced);
        check_budget(&budget)?;

        if exhausted.load(Ordering::SeqCst) || cancel.is_stopped() {
            break;
        }
    }

    let flows = runner.end(&mut context, None)?;
    let flushed = flows.into_iter().filter_map(Flow::into_item).collect();
    forward(&downstream, flushed);
    check_budget(&budget)?;
    send_done(&downstream);

    Ok(())
}

/// Fails the run when `budget` gave up waiting for room.
///
/// A source only ever gets room back when a frame it emitted earlier is dropped, so a wait that
/// lasted the whole deadline is one nothing was ever going to end. The run stops naming the limit
/// and what was outstanding, rather than sitting there.
fn check_budget(budget: &FrameBudget) -> Result<(), RunError> {
    match budget.stalled() {
        Some(outstanding) => Err(RunError::BudgetStalled {
            limit: budget.limit(),
            outstanding,
        }),
        None => Ok(()),
    }
}
