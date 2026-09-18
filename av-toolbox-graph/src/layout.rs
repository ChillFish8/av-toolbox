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
        layout.reduce_crossings();
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

    /// Reorders the ranks a few times, keeping whichever ordering crossed the least.
    fn reduce_crossings(&mut self) {
        let mut best = self.ranks.clone();
        let mut fewest = self.crossings;

        for _ in 0..SWEEPS {
            for rank in 1..self.ranks.len() {
                self.order_by_median(rank, rank - 1);
            }

            for rank in (0..self.ranks.len().saturating_sub(1)).rev() {
                self.order_by_median(rank, rank + 1);
            }

            let crossings = self.count_crossings();
            if crossings < fewest {
                fewest = crossings;
                best = self.ranks.clone();
            }
        }

        self.ranks = best;
        self.crossings = fewest;
    }

    /// Orders one rank by where each slot's neighbours sit in an adjacent rank.
    fn order_by_median(&mut self, rank: usize, neighbour_rank: usize) {
        let gap = rank.min(neighbour_rank);
        let reading_upwards = neighbour_rank > rank;
        let neighbours = positions_in(&self.ranks[neighbour_rank]);

        let mut keyed: Vec<(Slot, f64)> = self.ranks[rank]
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let found: Vec<usize> = self.links[gap]
                    .iter()
                    .filter_map(|link| {
                        let (near, far) = match reading_upwards {
                            true => (link.upper, link.lower),
                            false => (link.lower, link.upper),
                        };

                        if near != *slot {
                            return None;
                        }

                        neighbours.get(&far).copied()
                    })
                    .collect();

                (*slot, median(found, index))
            })
            .collect();

        keyed.sort_by(|first, second| first.1.total_cmp(&second.1));
        self.ranks[rank] = keyed.into_iter().map(|(slot, _)| slot).collect();
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

/// How many down and up passes to make. More than this rarely finds a better ordering.
const SWEEPS: usize = 4;

/// The middle of a slot's neighbours, or its current position when it has none.
fn median(mut positions: Vec<usize>, fallback: usize) -> f64 {
    if positions.is_empty() {
        return fallback as f64;
    }

    positions.sort_unstable();
    let middle = positions.len() / 2;

    if positions.len() % 2 == 1 {
        return positions[middle] as f64;
    }

    (positions[middle - 1] + positions[middle]) as f64 / 2.0
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

    /// Two sources whose consumers are declared in the opposite order, which crosses the edges.
    fn crossed_branches() -> Vec<StageSpec> {
        let mut first_source = testing::spec(StageKind::Source, "first_source", "test/file");
        first_source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut second_source = testing::spec(StageKind::Source, "second_source", "test/file");
        second_source.args = Spanned::unknown(json!({ "path": "b.mkv" }));

        let mut first_blur = testing::spec(StageKind::Filter, "first_blur", "test/blur");
        first_blur.inputs = vec![testing::positional("second_source")];
        first_blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut second_blur = testing::spec(StageKind::Filter, "second_blur", "test/blur");
        second_blur.inputs = vec![testing::positional("first_source")];
        second_blur.args = Spanned::unknown(json!({ "sigma": 2.0 }));

        let mut first_sink = testing::spec(StageKind::Sink, "first_out", "test/stdout");
        first_sink.inputs = vec![testing::positional("first_blur")];

        let mut second_sink = testing::spec(StageKind::Sink, "second_out", "test/stdout");
        second_sink.inputs = vec![testing::positional("second_blur")];

        vec![
            first_source,
            second_source,
            first_blur,
            second_blur,
            first_sink,
            second_sink,
        ]
    }

    #[test]
    fn reordering_a_rank_removes_the_crossings_declaration_order_created() {
        let (_, layout) = layout_of(&crossed_branches());

        assert_eq!(layout.crossings(), 0);
    }

    #[test]
    fn the_same_graph_always_lays_out_the_same_way() {
        let (_, first) = layout_of(&crossed_branches());
        let (_, second) = layout_of(&crossed_branches());

        assert_eq!(first.ranks(), second.ranks());
    }
}
