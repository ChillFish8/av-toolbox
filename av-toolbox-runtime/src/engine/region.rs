use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender, SyncSender};

use av_toolbox_gpu::{FrameBudget, FrameContext};

use super::channel::{AggregateSignal, DownstreamLink, Slotted};
use super::freedom::{FirstIdle, WorkerFreedom};
use crate::worker::RegionRunner;
use crate::{Aggregate, Disperse, WorkItem, WorkResult};

/// A region with no disperse and no aggregate, headed by a source.
///
/// Nothing feeds it, so it drives itself until [WatchedSource](crate::engine::wrap::WatchedSource)
/// reports its head is exhausted.
pub(super) struct SourceRegion {
    pub(super) runner: RegionRunner,
    pub(super) context: FrameContext,
    pub(super) budget: Arc<FrameBudget>,
    pub(super) exhausted: Arc<AtomicBool>,
    pub(super) downstream: Vec<DownstreamLink>,
}

/// A region with no disperse and no aggregate, fed by exactly one upstream region.
///
/// The upstream region's own exchange, if it has one, has already put its output back in order,
/// so this region only has to carry it forward. Its one input still arrives tagged with the
/// slot it fills, the same as every other consumer, so it can size its bundle from the tag
/// rather than needing to remember which slot that is.
pub(super) struct PassthroughRegion {
    pub(super) runner: RegionRunner,
    pub(super) context: FrameContext,
    pub(super) upstream: Receiver<Slotted>,
    pub(super) downstream: Vec<DownstreamLink>,
}

/// One worker of an [ExchangedRegion], before its thread is spawned.
pub(crate) struct WorkerSlot {
    pub(crate) runner: RegionRunner,
    pub(crate) context: FrameContext,
    pub(crate) work_receiver: Receiver<WorkItem>,
    pub(crate) result_sender: SyncSender<WorkResult>,
}

/// A region routed through a disperse, and merged back through an aggregate.
///
/// This covers every region wider than one worker, and a narrow region whose head pairs more
/// than one input. Either way the disperse decides how many workers matter, not the aggregate.
///
/// Every producer feeding this region's head shares one channel, each message tagged with the
/// slot it fills, so exactly one thread ever touches `disperse`. That alone only removes the
/// data race between producers: `Disperse` still accepts at most one pending item per slot
/// before raising `SlotRanAhead`, so the driving thread cannot simply offer each message the
/// instant it arrives. It keeps its own queue per slot in
/// [drive_disperse](crate::engine::disperse::drive_disperse), one item deep or more, and only offers
/// once every slot has something waiting. A slot fed by a faster producer then queues up rather
/// than panicking, and how far it may run ahead of a slower one is bounded by the frame budget
/// rather than capped at one.
pub(super) struct ExchangedRegion {
    pub(super) disperse: DisperseSide,
    pub(super) freedom: Arc<WorkerFreedom>,
    pub(super) aggregate: Option<AggregateSide>,
    pub(super) result_receivers: Vec<Receiver<WorkResult>>,
    pub(super) workers: Vec<WorkerSlot>,
    pub(super) downstream: Vec<DownstreamLink>,
}

/// Everything the one thread merging a region's workers back together owns.
///
/// Only a region above one worker has one. A region of one worker has nothing to merge, so its
/// worker's output goes straight downstream instead, keeping the item indices it already carries.
pub(crate) struct AggregateSide {
    pub(crate) aggregate: Aggregate,
    pub(crate) assignment_receiver: Receiver<AggregateSignal>,
}

/// Everything the one thread driving a region's disperse owns.
pub(super) struct DisperseSide {
    pub(super) disperse: Disperse<FirstIdle>,
    pub(super) upstream: Receiver<Slotted>,
    pub(super) slots: Vec<usize>,
    pub(super) freedom: Arc<WorkerFreedom>,
    pub(super) work_senders: Vec<Sender<WorkItem>>,
    pub(super) assignment_sender: Sender<AggregateSignal>,
}

pub(super) enum PreparedRegion {
    Source(SourceRegion),
    Passthrough(PassthroughRegion),
    Exchanged(ExchangedRegion),
}
