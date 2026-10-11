use std::collections::HashSet;

use av_toolbox_core::StageKind;

use crate::graph::{CompiledGraph, NodeId};

/// The sources a probed input reads from.
pub(super) struct Reached {
    pub(super) sources: Vec<NodeId>,
    /// The first stage upstream of the probe that takes more than one input.
    pub(super) meeting: Option<NodeId>,
}

/// Every source upstream of `start`, following every input of every stage on the way.
pub(super) fn reached(graph: &CompiledGraph, start: NodeId) -> Reached {
    let mut sources = Vec::new();
    let mut meeting = None;
    let mut seen = HashSet::new();
    let mut pending = vec![start];

    while let Some(current) = pending.pop() {
        if !seen.insert(current) {
            continue;
        }

        let node = graph.node(current);
        if node.kind() == StageKind::Source {
            sources.push(current);
            continue;
        }

        if node.inputs().len() > 1 && meeting.is_none() {
            meeting = Some(current);
        }

        let upstream = node.inputs().iter().rev().map(|edge| edge.source());
        pending.extend(upstream);
    }

    sources.sort();
    Reached { sources, meeting }
}
