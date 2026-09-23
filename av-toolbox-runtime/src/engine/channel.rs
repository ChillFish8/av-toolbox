use std::sync::mpsc::Sender;

use av_toolbox_gpu::StreamItem;

use crate::{Assignment, Routed, WorkItem};

/// A message crossing the unbounded channel between two regions.
#[derive(Clone)]
pub(crate) enum ExitMessage {
    Item(StreamItem),
    Done,
}

/// One region's message, tagged with the input slot it fills at the consumer.
///
/// A consumer with more than one input is fed by more than one upstream region, each with its
/// own sender cloned onto this one channel, so every message says which of the consumer's
/// slots it belongs to. That is what lets one thread gather a whole bundle from one receiver
/// instead of several threads racing several receivers into the same disperse.
pub(crate) type Slotted = (usize, ExitMessage);

/// Where a region sends what one of its exit nodes produces: the slot it fills at whichever
/// consumer reads that edge, and the channel tagged messages for that slot go out on.
pub(crate) type DownstreamLink = (usize, Sender<Slotted>);

/// A message from a disperse to the aggregate reading its assignments.
pub(crate) enum AggregateSignal {
    Assignment(Assignment),
    Finished,
}

fn send_to_all(downstream: &[DownstreamLink], message: ExitMessage) {
    for (slot, sender) in downstream {
        sender.send((*slot, message.clone())).ok();
    }
}

pub(super) fn forward(downstream: &[DownstreamLink], items: Vec<StreamItem>) {
    for item in items {
        send_to_all(downstream, ExitMessage::Item(item));
    }
}

pub(super) fn send_done(downstream: &[DownstreamLink]) {
    send_to_all(downstream, ExitMessage::Done);
}

pub(super) fn send_routed(work_senders: &[Sender<WorkItem>], routed: Vec<Routed>) {
    for Routed { worker, item } in routed {
        work_senders[worker].send(item).ok();
    }
}
