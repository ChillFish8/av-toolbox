use std::collections::HashMap;

use crate::graph::{CompiledGraph, EdgeId, NodeId};

/// One position in a rank.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum Slot {
    Stage(NodeId),
    /// A point an edge passes through on its way down through a rank it does not stop in.
    Virtual(EdgeId),
}

/// A connection between a slot in one rank and a slot in the rank below it.
#[derive(Debug, Copy, Clone)]
struct Link {
    upper: Slot,
    lower: Slot,
}

/// Where every stage sits, without saying how wide anything is.
#[derive(Debug, Clone)]
pub struct Layout {
    ranks: Vec<Vec<Slot>>,
    links: Vec<Vec<Link>>,
    stage_ranks: Vec<usize>,
    crossings: usize,
}

impl Layout {
    pub fn build(graph: &CompiledGraph) -> Self {
        let stage_ranks = assign_ranks(graph);
        let depth = stage_ranks.iter().copied().max().map_or(0, |deepest| deepest + 1);

        let mut ranks: Vec<Vec<Slot>> = vec![Vec::new(); depth];
        for node in graph.nodes() {
            ranks[stage_ranks[node.id().index()]].push(Slot::Stage(node.id()));
        }

        let mut links: Vec<Vec<Link>> = vec![Vec::new(); depth.saturating_sub(1)];
        for node in graph.nodes() {
            for (index, edge) in node.inputs().iter().enumerate() {
                let id = EdgeId::new(node.id(), index);
                let from = stage_ranks[edge.source().index()];
                let to = stage_ranks[node.id().index()];

                let mut upper = Slot::Stage(edge.source());
                for rank in (from + 1)..to {
                    let hop = Slot::Virtual(id);
                    ranks[rank].push(hop);
                    links[rank - 1].push(Link { upper, lower: hop });
                    upper = hop;
                }

                links[to - 1].push(Link {
                    upper,
                    lower: Slot::Stage(node.id()),
                });
            }
        }

        let mut layout = Self {
            ranks,
            links,
            stage_ranks,
            crossings: 0,
        };

        layout.crossings = layout.count_crossings();
        layout
    }

    pub fn ranks(&self) -> &[Vec<Slot>] {
        &self.ranks
    }

    pub fn crossings(&self) -> usize {
        self.crossings
    }

    pub fn rank_of_stage(&self, node: NodeId) -> usize {
        self.stage_ranks[node.index()]
    }

    pub fn slot_position(&self, rank: usize, slot: Slot) -> Option<usize> {
        self.ranks.get(rank)?.iter().position(|entry| *entry == slot)
    }

    /// Counts how many pairs of links cross between one rank and the next.
    fn count_crossings(&self) -> usize {
        let positions: Vec<HashMap<Slot, usize>> = self.ranks.iter().map(|rank| positions_in(rank)).collect();
        let mut total = 0;

        for (gap, links) in self.links.iter().enumerate() {
            let placed: Vec<(usize, usize)> = links
                .iter()
                .map(|link| (positions[gap][&link.upper], positions[gap + 1][&link.lower]))
                .collect();

            for (index, first) in placed.iter().enumerate() {
                for second in placed.iter().skip(index + 1) {
                    let crosses = (first.0 < second.0 && first.1 > second.1)
                        || (first.0 > second.0 && first.1 < second.1);

                    if crosses {
                        total += 1;
                    }
                }
            }
        }

        total
    }
}

fn positions_in(rank: &[Slot]) -> HashMap<Slot, usize> {
    rank.iter()
        .enumerate()
        .map(|(index, slot)| (*slot, index))
        .collect()
}

/// Puts each stage one rank below the deepest stage it reads.
fn assign_ranks(graph: &CompiledGraph) -> Vec<usize> {
    let mut ranks = vec![0; graph.len()];

    for id in graph.order() {
        let node = graph.node(*id);
        let deepest = node
            .inputs()
            .iter()
            .map(|edge| ranks[edge.source().index()] + 1)
            .max();

        ranks[id.index()] = deepest.unwrap_or(0);
    }

    ranks
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::{Spanned, StageSpec, build, testing};

    /// A source read directly by a mask, and also through a filter, so one edge spans two ranks.
    fn skipping_edge() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("blur"), testing::positional("input")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("merge")];

        vec![source, blur, merge, sink]
    }

    fn layout_of(specs: &[StageSpec]) -> (crate::CompiledGraph, Layout) {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();
        let layout = Layout::build(&graph);

        (graph, layout)
    }

    #[test]
    fn a_stage_sits_one_rank_below_its_deepest_input() {
        let (graph, layout) = layout_of(&skipping_edge());

        let merge = graph
            .nodes()
            .iter()
            .find(|node| node.name() == "merge")
            .expect("the mask is in the graph");

        assert_eq!(layout.rank_of_stage(merge.id()), 2);
        assert_eq!(layout.ranks().len(), 4);
    }

    #[test]
    fn an_edge_spanning_two_ranks_gets_a_virtual_slot_in_between() {
        let (_, layout) = layout_of(&skipping_edge());

        let hops = layout.ranks()[1]
            .iter()
            .filter(|slot| matches!(slot, Slot::Virtual(_)))
            .count();

        assert_eq!(hops, 1);
    }

    #[test]
    fn a_rank_lists_its_stages_in_declaration_order() {
        let (graph, layout) = layout_of(&skipping_edge());
        let first = layout.ranks()[0][0];

        assert_eq!(first, Slot::Stage(graph.nodes()[0].id()));
    }

    #[test]
    fn a_graph_whose_edges_do_not_cross_reports_no_crossings() {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let (_, layout) = layout_of(&[source, sink]);

        assert_eq!(layout.crossings(), 0);
    }
}
