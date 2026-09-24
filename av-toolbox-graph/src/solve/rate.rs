use av_toolbox_gpu::{FrameRate, StreamFormat};

use super::Solver;
use crate::diagnostic::stage_path;
use crate::graph::NodeId;
use crate::{Diagnostic, DiagnosticCode};

impl Solver<'_> {
    /// Whether every frames input of `id` runs at one frame rate, reporting the node when not.
    ///
    /// Masks and unbound slots carry no rate and are skipped. An unknown rate only matches another
    /// unknown rate.
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
        let message = format!("{listed}, but every frames input of a stage must share one frame rate");

        let node = self.graph.node(id);
        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::FrameRateMismatch,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        ));

        false
    }
}

/// The frame rate of every bound frames input, beside its signature slot.
fn frame_rates(inputs: &[Option<StreamFormat>]) -> Vec<(usize, Option<FrameRate>)> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(slot, input)| match input {
            Some(StreamFormat::Frames(format)) => Some((slot, format.frame_rate)),
            _ => None,
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
