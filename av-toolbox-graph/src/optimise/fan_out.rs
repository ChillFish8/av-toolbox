use super::{Change, OptimisationPass, PassReport};
use crate::graph::{CompiledGraph, NodeId, Sharing};

/// Records whether each consumer can take its producer's output outright.
///
/// A producer read exactly once can hand its output over. A producer read more than once, whether
/// by different stages or twice by the same stage, has to share it.
pub struct FanOutAnalysis;

impl OptimisationPass for FanOutAnalysis {
    fn name(&self) -> &'static str {
        "fan-out-analysis"
    }

    fn run(&self, graph: &mut CompiledGraph) -> PassReport {
        let mut report = PassReport::new(self.name());
        let decisions = decide(graph);

        for (consumer, index, sharing) in decisions {
            let (current, consumer_name, producer_name) = {
                let node = graph.node(consumer);
                let edge = &node.inputs()[index];
                let producer = graph.node(edge.source()).name().to_string();

                (edge.sharing(), node.name().to_string(), producer)
            };

            if current == sharing {
                continue;
            }

            report.push(Change::EdgeSharing {
                consumer: consumer_name,
                producer: producer_name,
                sharing,
            });

            graph.node_mut(consumer).inputs_mut()[index].set_sharing(sharing);
        }

        report
    }
}

/// Works out the sharing for every edge before any of them is written back.
fn decide(graph: &CompiledGraph) -> Vec<(NodeId, usize, Sharing)> {
    let mut decisions = Vec::new();

    for node in graph.nodes() {
        for (index, edge) in node.inputs().iter().enumerate() {
            let reads = graph.node(edge.source()).consumers().len();
            let sharing = if reads == 1 { Sharing::Move } else { Sharing::Share };

            decisions.push((node.id(), index, sharing));
        }
    }

    decisions
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::optimise::{Change, default_passes, optimise};
    use crate::{Sharing, Spanned, StageSpec, build, testing};

    /// One source read by a filter and a sink, so the source has two consumers.
    fn shared_source() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut first_sink = testing::spec(StageKind::Sink, "raw", "test/stdout");
        first_sink.inputs = vec![testing::positional("input")];

        let mut second_sink = testing::spec(StageKind::Sink, "blurred", "test/stdout");
        second_sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, first_sink, second_sink]
    }

    #[test]
    fn a_sole_consumer_takes_the_output_by_move() {
        let registry = testing::registry();
        let mut graph = build(&registry, &shared_source())
            .expect("should build")
            .into_graph();

        FanOutAnalysis.run(&mut graph);

        let blurred = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "blurred")
            .expect("the sink survives");

        assert_eq!(blurred.inputs()[0].sharing(), Sharing::Move);
    }

    #[test]
    fn a_producer_with_two_consumers_is_shared() {
        let registry = testing::registry();
        let mut graph = build(&registry, &shared_source())
            .expect("should build")
            .into_graph();

        FanOutAnalysis.run(&mut graph);

        let blur = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "blur")
            .expect("the filter survives");

        assert_eq!(blur.inputs()[0].sharing(), Sharing::Share);
    }

    #[test]
    fn only_the_edges_that_changed_are_reported() {
        let registry = testing::registry();
        let mut graph = build(&registry, &shared_source())
            .expect("should build")
            .into_graph();

        let report = FanOutAnalysis.run(&mut graph);

        assert_eq!(
            report.changes(),
            &[Change::EdgeSharing {
                consumer: String::from("blurred"),
                producer: String::from("blur"),
                sharing: Sharing::Move,
            }]
        );
    }

    #[test]
    fn the_default_set_changes_nothing_on_a_second_run() {
        let registry = testing::registry();
        let mut graph = build(&registry, &shared_source())
            .expect("should build")
            .into_graph();

        let passes = default_passes();
        optimise(&mut graph, &passes);
        let reports = optimise(&mut graph, &passes);

        assert!(reports.iter().all(|report| report.is_empty()));
    }
}
