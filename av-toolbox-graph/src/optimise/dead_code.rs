use std::collections::{HashMap, HashSet};

use super::{Change, OptimisationPass, PassReport};
use crate::graph::{CompiledGraph, NodeId};

/// Drops every stage no sink transitively depends on.
///
/// A stage an artefact sink reads or another stage references is kept.
pub struct DeadStageElimination;

impl OptimisationPass for DeadStageElimination {
    fn name(&self) -> &'static str {
        "dead-stage-elimination"
    }

    fn run(&self, graph: &mut CompiledGraph) -> PassReport {
        let mut report = PassReport::new(self.name());
        let reachable = reachable_from_sinks(graph);

        let mut dropped = HashSet::new();
        for node in graph.nodes() {
            if reachable.contains(&node.id()) {
                continue;
            }

            dropped.insert(node.id());
            report.push(Change::DroppedStage {
                name: node.name().to_string(),
            });
        }

        if dropped.is_empty() {
            return report;
        }

        graph.retain_nodes(&HashMap::new(), &dropped);
        report
    }
}

fn reachable_from_sinks(graph: &CompiledGraph) -> HashSet<NodeId> {
    let mut reachable = HashSet::new();
    let producers = graph.artefact_producers();
    let mut pending: Vec<NodeId> = graph.sinks().collect();
    pending.extend(producers);

    while let Some(id) = pending.pop() {
        if !reachable.insert(id) {
            continue;
        }

        let node = graph.node(id);
        let inputs = node.inputs().iter().map(|edge| edge.source());
        let references = node.references().iter().map(|reference| reference.source);
        pending.extend(inputs.chain(references));
    }

    reachable
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::optimise::Change;
    use crate::{Spanned, StageSpec, build, testing};

    /// A source and a sink, plus a filter nothing reads.
    fn specs_with_a_spare_filter() -> Vec<StageSpec> {
        let source = testing::spec(StageKind::Source, "input", "test/file");

        let mut spare = testing::spec(StageKind::Filter, "spare", "test/blur");
        spare.inputs = vec![testing::positional("input")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        vec![source, spare, sink]
    }

    #[test]
    fn a_stage_no_sink_reads_is_dropped() {
        let registry = testing::registry();
        let specs = specs_with_a_spare_filter();
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        let report = DeadStageElimination.run(&mut graph);

        assert_eq!(
            report.changes(),
            &[Change::DroppedStage {
                name: String::from("spare"),
            }]
        );

        let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();
        assert_eq!(names, vec!["input", "out"]);
    }

    #[test]
    fn a_second_run_changes_nothing() {
        let registry = testing::registry();
        let specs = specs_with_a_spare_filter();
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        DeadStageElimination.run(&mut graph);
        let report = DeadStageElimination.run(&mut graph);

        assert!(report.is_empty());
    }

    #[test]
    fn dropping_a_stage_clears_a_stored_plan() {
        let registry = testing::registry();
        let mut specs = specs_with_a_spare_filter();
        specs[0].args = Spanned::unknown(json!({ "path": "a.mkv" }));
        specs[1].args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);
        assert!(graph.execution_plan().is_some());

        DeadStageElimination.run(&mut graph);

        assert!(graph.execution_plan().is_none());
    }

    #[test]
    fn the_surviving_edges_still_point_at_the_right_stages() {
        let registry = testing::registry();
        let specs = specs_with_a_spare_filter();
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        DeadStageElimination.run(&mut graph);

        let sink = graph.node(NodeId::new(1));
        assert_eq!(graph.node(sink.inputs()[0].source()).name(), "input");
    }
}
