use std::collections::{HashMap, HashSet};

use av_toolbox_core::StageKind;

use super::{Change, OptimisationPass, PassReport};
use crate::graph::{CompiledGraph, Node, NodeId};

/// What makes two stages interchangeable.
///
/// The kind is part of the key because the registry holds one map per kind, so the same
/// identifier can name a different stage in each. Arguments compare structurally through their
/// rendered JSON, so `1` and `1.0` are different stages.
#[derive(Debug, Eq, PartialEq, Hash)]
struct StageKey {
    kind: StageKind,
    identifier: String,
    args: String,
    inputs: Vec<(Option<String>, NodeId)>,
}

/// Merges stages that do the same work on the same inputs.
///
/// A stage an artefact sink reads is never merged, since each one's artefacts are written under its own name.
/// A deferred stage is never merged either, since its args are not known yet.
pub struct CommonStageElimination;

impl OptimisationPass for CommonStageElimination {
    fn name(&self) -> &'static str {
        "common-stage-elimination"
    }

    fn run(&self, graph: &mut CompiledGraph) -> PassReport {
        let mut report = PassReport::new(self.name());
        let mut seen: HashMap<StageKey, NodeId> = HashMap::new();
        let mut replacements: HashMap<NodeId, NodeId> = HashMap::new();

        let producers = graph.artefact_producers();

        for id in graph.order() {
            let node = graph.node(*id);
            if producers.contains(id) || node.is_deferred() {
                continue;
            }

            let key = key_for(node, &replacements);

            let Some(kept) = seen.get(&key).copied() else {
                seen.insert(key, *id);
                continue;
            };

            replacements.insert(*id, kept);
            report.push(Change::MergedStages {
                kept: graph.node(kept).name().to_string(),
                removed: node.name().to_string(),
            });
        }

        if replacements.is_empty() {
            return report;
        }

        let dropped: HashSet<NodeId> = replacements.keys().copied().collect();
        graph.retain_nodes(&replacements, &dropped);

        report
    }
}

/// Builds a node's key, reading its inputs through the merges made so far.
fn key_for(node: &Node, replacements: &HashMap<NodeId, NodeId>) -> StageKey {
    let inputs = node
        .inputs()
        .iter()
        .map(|edge| {
            let source = replacements.get(&edge.source()).copied();
            let resolved = source.unwrap_or_else(|| edge.source());

            (edge.name().map(str::to_string), resolved)
        })
        .collect();

    StageKey {
        kind: node.kind(),
        identifier: node.identifier().to_string(),
        args: node.args().to_string(),
        inputs,
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::optimise::Change;
    use crate::{Spanned, StageSpec, build, testing};

    /// Two identical sources, a filter on each, and a mask reading both filters.
    fn duplicated_branches() -> Vec<StageSpec> {
        let mut first_source = testing::spec(StageKind::Source, "first_source", "test/file");
        first_source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut second_source = testing::spec(StageKind::Source, "second_source", "test/file");
        second_source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut first_blur = testing::spec(StageKind::Filter, "first_blur", "test/blur");
        first_blur.inputs = vec![testing::positional("first_source")];
        first_blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut second_blur = testing::spec(StageKind::Filter, "second_blur", "test/blur");
        second_blur.inputs = vec![testing::positional("second_source")];
        second_blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![
            testing::positional("first_blur"),
            testing::positional("second_blur"),
        ];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("merge")];

        vec![first_source, second_source, first_blur, second_blur, merge, sink]
    }

    #[test]
    fn merging_a_shared_producer_cascades_into_its_consumers() {
        let registry = testing::registry();
        let mut graph = build(&registry, &duplicated_branches())
            .expect("should build")
            .into_graph();

        let report = CommonStageElimination.run(&mut graph);

        assert_eq!(
            report.changes(),
            &[
                Change::MergedStages {
                    kept: String::from("first_source"),
                    removed: String::from("second_source"),
                },
                Change::MergedStages {
                    kept: String::from("first_blur"),
                    removed: String::from("second_blur"),
                },
            ]
        );

        let names: Vec<&str> = graph.nodes().iter().map(|node| node.name()).collect();
        assert_eq!(names, vec!["first_source", "first_blur", "merge", "out"]);
    }

    #[test]
    fn both_of_the_masks_inputs_now_read_the_surviving_filter() {
        let registry = testing::registry();
        let mut graph = build(&registry, &duplicated_branches())
            .expect("should build")
            .into_graph();

        CommonStageElimination.run(&mut graph);

        let merge = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "merge")
            .expect("the mask survives");

        let sources: Vec<&str> = merge
            .inputs()
            .iter()
            .map(|edge| graph.node(edge.source()).name())
            .collect();

        assert_eq!(sources, vec!["first_blur", "first_blur"]);
    }

    #[test]
    fn stages_differing_only_in_arguments_are_left_alone() {
        let mut specs = duplicated_branches();
        specs[3].args = Spanned::unknown(json!({ "sigma": 2.0 }));

        let registry = testing::registry();
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        let report = CommonStageElimination.run(&mut graph);

        assert_eq!(
            report.changes(),
            &[Change::MergedStages {
                kept: String::from("first_source"),
                removed: String::from("second_source"),
            }]
        );
    }

    #[test]
    fn a_second_run_changes_nothing() {
        let registry = testing::registry();
        let mut graph = build(&registry, &duplicated_branches())
            .expect("should build")
            .into_graph();

        CommonStageElimination.run(&mut graph);
        let report = CommonStageElimination.run(&mut graph);

        assert!(report.is_empty());
    }
}
