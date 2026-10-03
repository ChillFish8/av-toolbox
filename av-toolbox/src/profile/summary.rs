use std::collections::BTreeMap;
use std::time::Duration;

use av_toolbox_core::Parallelism;
use av_toolbox_graph::{NodeId, StageSpec};
use av_toolbox_runtime::{NodeTiming, RunReport};

use super::file::stage_key;
use super::model::{NodeCost, Width};
use super::pass::{NodeInfo, Pass};
use super::report::{Row, WorkersCell};

/// Each measured node's cost per frame, ready for the worker model.
pub(super) fn costs(pass: &Pass) -> Vec<NodeCost> {
    let mut costs = Vec::with_capacity(pass.nodes.len());

    for info in &pass.nodes {
        let Some(timing) = timing_of(&pass.report, info.id) else {
            continue;
        };

        let parallelism = model_parallelism(info);
        let host = per_frame(timing.host, timing.frames);
        let busy = per_frame(timing.busy, timing.frames);

        costs.push(NodeCost {
            key: info.key.clone(),
            parallelism,
            host,
            busy,
        });
    }

    costs
}

/// What the report's rows are built from.
pub(super) struct RowSources<'a> {
    /// The measure pass, whose timings fill each row.
    pub(super) measured: &'a Pass,
    /// The baseline pass, whose planned widths show on the left of the workers column.
    pub(super) baseline: &'a Pass,
    pub(super) tuned: &'a [Width],
    pub(super) written: &'a [StageSpec],
    pub(super) discarded: &'a [String],
}

/// One row per measured node, in the measure pass's order.
///
/// A replaced sink is marked `(null)`, and every widenable stage shows its baseline and suggested widths.
pub(super) fn rows(sources: &RowSources<'_>) -> Vec<Row> {
    let measured = sources.measured;
    let mut rows = Vec::with_capacity(measured.nodes.len());

    for info in &measured.nodes {
        let Some(timing) = timing_of(&measured.report, info.id) else {
            continue;
        };

        let label = row_label(info, sources.discarded);
        let host = per_frame(timing.host, timing.frames);
        let busy = per_frame(timing.busy, timing.frames);
        let device = busy.saturating_sub(host);
        let workers = workers_cell(info, sources);

        rows.push(Row {
            label,
            host,
            device,
            busy,
            workers,
        });
    }

    rows
}

/// The width the plan gave every widenable stage, inherited widths included.
pub(super) fn planned_workers(nodes: &[NodeInfo]) -> BTreeMap<String, usize> {
    nodes
        .iter()
        .filter(|info| info.widenable)
        .map(|info| (info.key.clone(), info.planned))
        .collect()
}

/// The counted wall time per frame that no node's own calls account for.
pub(super) fn overhead(report: &RunReport, counted: u64) -> Duration {
    let wall = per_frame(report.counted_wall, counted);
    let busy: Duration = report
        .timings
        .iter()
        .map(|timing| per_frame(timing.busy, timing.frames))
        .sum();

    wall.saturating_sub(busy)
}

/// The parallelism the model sees.
///
/// A widenable stage keeps the plan's and a `Stream` node keeps `Stream` so it can set the floor. Every other
/// node, such as a sink or a transfer, gets `None`.
fn model_parallelism(info: &NodeInfo) -> Option<Parallelism> {
    if info.widenable || info.parallelism == Some(Parallelism::Stream) {
        return info.parallelism;
    }

    None
}

fn row_label(info: &NodeInfo, discarded: &[String]) -> String {
    if discarded.contains(&info.key) {
        return format!("{} (null)", info.label);
    }

    info.label.clone()
}

fn workers_cell(info: &NodeInfo, sources: &RowSources<'_>) -> Option<WorkersCell> {
    if !info.widenable {
        return None;
    }

    let suggested = sources.tuned.iter().find(|width| width.key == info.key)?;
    let baseline = sources.baseline.nodes.iter().find(|node| node.key == info.key)?;
    let declared = declared_width(sources.written, &info.key);

    Some(WorkersCell {
        baseline: baseline.planned,
        declared,
        suggested: suggested.workers,
    })
}

fn declared_width(written: &[StageSpec], key: &str) -> Option<usize> {
    let spec = written.iter().find(|spec| stage_key(spec) == key)?;
    spec.workers
}

fn timing_of(report: &RunReport, id: NodeId) -> Option<&NodeTiming> {
    report.timings.iter().find(|timing| timing.node == id)
}

fn per_frame(total: Duration, frames: u64) -> Duration {
    if frames == 0 {
        return Duration::ZERO;
    }

    total.div_f64(frames as f64)
}

#[cfg(test)]
mod tests {
    use av_toolbox_graph::NodeId;

    use super::*;

    fn info(index: usize, key: &str, planned: usize, widenable: bool) -> NodeInfo {
        NodeInfo {
            id: NodeId::new(index),
            key: key.to_string(),
            label: key.to_string(),
            parallelism: Some(Parallelism::Frame),
            planned,
            widenable,
        }
    }

    #[test]
    fn planned_workers_keep_a_width_inherited_from_a_wide_input() {
        let nodes = vec![
            info(0, "source.clip", 1, false),
            info(1, "filter.wide", 4, true),
            info(2, "filter.after", 4, true),
            info(3, "sink.output", 1, false),
        ];

        let planned = planned_workers(&nodes);

        assert_eq!(planned.len(), 2);
        assert_eq!(planned["filter.wide"], 4);
        assert_eq!(planned["filter.after"], 4);
    }
}
