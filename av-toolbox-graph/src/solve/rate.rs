use av_toolbox_gpu::{FrameRate, StreamFormat};

use super::Solver;
use crate::diagnostic::stage_path;
use crate::graph::NodeId;
use crate::{Diagnostic, DiagnosticCode};

impl Solver<'_> {
    /// Whether every frames and mask input of `id` runs at one frame rate, reporting the node when not.
    ///
    /// Unbound slots are skipped. An unknown rate only matches another unknown rate.
    pub(super) fn rates_agree(&mut self, id: NodeId, inputs: &[Option<StreamFormat>]) -> bool {
        let rates = frame_rates(inputs);

        let Some((_, first)) = rates.first() else {
            return true;
        };

        let all_equal = rates.iter().all(|(_, rate)| rate == first);
        if all_equal {
            return true;
        }

        let listed = describe(&rates);
        let message =
            format!("{listed}, but every frames and mask input of a stage must share one frame rate");

        let graph = self.graph;
        let node = graph.node(id);
        let path = stage_path(node.kind(), node.name());
        let mut diagnostic = Diagnostic::error(DiagnosticCode::FrameRateMismatch, path, node.span(), message);

        for (slot, rate) in &rates {
            let edge_index = self.slots[id.index()][*slot].expect("a bound slot has an edge");
            let edge = &node.inputs()[edge_index];
            let source = edge.source();
            let producer = graph.node(source).name();
            let param = edge.param();
            let note = describe_input(param, producer, *rate);
            let span = edge.span();

            diagnostic = diagnostic.with_note(note, span);
        }

        self.diagnostics.push(diagnostic);

        false
    }
}

fn describe_input(param: Option<&str>, producer: &str, rate: Option<FrameRate>) -> String {
    let reads = match param {
        Some(param) => format!("{param} reads {producer}"),
        None => format!("this input reads {producer}"),
    };

    match rate {
        Some(rate) => format!("{reads} at {rate}"),
        None => format!("{reads} at an unknown rate"),
    }
}

/// The frame rate of every bound frames or mask input, beside its signature slot.
fn frame_rates(inputs: &[Option<StreamFormat>]) -> Vec<(usize, Option<FrameRate>)> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(slot, input)| match input {
            Some(StreamFormat::Frames(format)) => Some((slot, format.frame_rate)),
            Some(StreamFormat::Mask(format)) => Some((slot, format.frame_rate)),
            None => None,
        })
        .collect()
}

fn describe(rates: &[(usize, Option<FrameRate>)]) -> String {
    let described: Vec<String> = rates
        .iter()
        .map(|(slot, rate)| match rate {
            Some(rate) => format!("input {slot} runs at {rate}"),
            None => format!("input {slot} runs at an unknown rate"),
        })
        .collect();

    described.join(" and ")
}
