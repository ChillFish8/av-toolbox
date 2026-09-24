use std::collections::{HashMap, VecDeque};

/// One position in a rank.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum Slot {
    Node(usize),
    /// A point an edge passes through on its way across a rank it does not stop in.
    Virtual(usize),
}

/// A connection between a slot in one rank and a slot in the next.
#[derive(Debug, Copy, Clone)]
struct Link {
    from: Slot,
    to: Slot,
}

/// Where every node sits, without saying how big anything is.
#[derive(Debug, Clone)]
pub(crate) struct Layout {
    ranks: Vec<Vec<Slot>>,
    links: Vec<Vec<Link>>,
    node_ranks: Vec<usize>,
    edges: Vec<(usize, usize)>,
    crossings: usize,
}

impl Layout {
    /// Lays out `count` nodes joined by `edges`, each naming the node it leaves and the node it enters.
    pub(crate) fn build(count: usize, edges: &[(usize, usize)]) -> Self {
        let node_ranks = longest_path_ranks(count, edges);
        Self::with_ranks(node_ranks, edges)
    }

    /// Lays out nodes whose ranks are already decided.
    ///
    /// Every edge must run from a lower rank to a higher one.
    pub(crate) fn with_ranks(node_ranks: Vec<usize>, edges: &[(usize, usize)]) -> Self {
        let depth = node_ranks.iter().copied().max().map_or(0, |deepest| deepest + 1);

        let mut ranks: Vec<Vec<Slot>> = vec![Vec::new(); depth];
        for (node, rank) in node_ranks.iter().enumerate() {
            ranks[*rank].push(Slot::Node(node));
        }

        let mut links: Vec<Vec<Link>> = vec![Vec::new(); depth.saturating_sub(1)];
        for (index, &(from, to)) in edges.iter().enumerate() {
            let first = node_ranks[from];
            let last = node_ranks[to];

            let mut previous = Slot::Node(from);
            for rank in (first + 1)..last {
                let hop = Slot::Virtual(index);
                ranks[rank].push(hop);
                links[rank - 1].push(Link {
                    from: previous,
                    to: hop,
                });
                previous = hop;
            }

            links[last - 1].push(Link {
                from: previous,
                to: Slot::Node(to),
            });
        }

        let mut layout = Self {
            ranks,
            links,
            node_ranks,
            edges: edges.to_vec(),
            crossings: 0,
        };

        layout.crossings = layout.count_crossings();
        layout.reduce_crossings();
        layout
    }

    pub(crate) fn ranks(&self) -> &[Vec<Slot>] {
        &self.ranks
    }

    #[cfg(test)]
    pub(crate) fn crossings(&self) -> usize {
        self.crossings
    }

    pub(crate) fn rank_of(&self, node: usize) -> usize {
        self.node_ranks[node]
    }

    /// The slots one edge passes through, from the node it leaves to the node it enters.
    pub(crate) fn chain(&self, edge: usize) -> Vec<(usize, Slot)> {
        let (from, to) = self.edges[edge];
        let first = self.node_ranks[from];
        let last = self.node_ranks[to];

        let mut chain = vec![(first, Slot::Node(from))];
        for rank in (first + 1)..last {
            chain.push((rank, Slot::Virtual(edge)));
        }

        chain.push((last, Slot::Node(to)));
        chain
    }

    /// Counts how many pairs of links cross between one rank and the next.
    fn count_crossings(&self) -> usize {
        let positions: Vec<HashMap<Slot, usize>> = self.ranks.iter().map(|rank| positions_in(rank)).collect();
        let mut total = 0;

        for (gap, links) in self.links.iter().enumerate() {
            let placed: Vec<(usize, usize)> = links
                .iter()
                .map(|link| (positions[gap][&link.from], positions[gap + 1][&link.to]))
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
        let reading_forwards = neighbour_rank > rank;
        let neighbours = positions_in(&self.ranks[neighbour_rank]);

        let mut keyed: Vec<(Slot, f64)> = self.ranks[rank]
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                let found: Vec<usize> = self.links[gap]
                    .iter()
                    .filter_map(|link| {
                        let (near, far) = match reading_forwards {
                            true => (link.from, link.to),
                            false => (link.to, link.from),
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

/// Puts each node one rank after the deepest node it reads.
pub(crate) fn longest_path_ranks(count: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut waiting = vec![0; count];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); count];

    for &(from, to) in edges {
        waiting[to] += 1;
        consumers[from].push(to);
    }

    let mut ready: VecDeque<usize> = (0..count).filter(|node| waiting[*node] == 0).collect();
    let mut ranks = vec![0; count];

    while let Some(node) = ready.pop_front() {
        for &consumer in &consumers[node] {
            ranks[consumer] = ranks[consumer].max(ranks[node] + 1);
            waiting[consumer] -= 1;

            if waiting[consumer] == 0 {
                ready.push_back(consumer);
            }
        }
    }

    ranks
}

/// How many forward and backward passes to make. More than this rarely finds a better ordering.
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
    use super::*;

    /// A source read directly by a mask, and also through a filter, so one edge spans two ranks.
    ///
    /// Nodes are the source, the filter, the mask and a sink, in that order.
    const SKIPPING_EDGE: [(usize, usize); 4] = [(0, 1), (1, 2), (0, 2), (2, 3)];

    #[test]
    fn a_node_sits_one_rank_after_its_deepest_input() {
        let layout = Layout::build(4, &SKIPPING_EDGE);

        assert_eq!(layout.rank_of(2), 2);
        assert_eq!(layout.ranks().len(), 4);
    }

    #[test]
    fn an_edge_spanning_two_ranks_gets_a_virtual_slot_in_between() {
        let layout = Layout::build(4, &SKIPPING_EDGE);

        let hops = layout.ranks()[1]
            .iter()
            .filter(|slot| matches!(slot, Slot::Virtual(_)))
            .count();

        assert_eq!(hops, 1);
    }

    #[test]
    fn a_rank_lists_its_nodes_in_the_order_given() {
        let layout = Layout::build(4, &SKIPPING_EDGE);

        assert_eq!(layout.ranks()[0][0], Slot::Node(0));
    }

    #[test]
    fn a_graph_whose_edges_do_not_cross_reports_no_crossings() {
        let layout = Layout::build(2, &[(0, 1)]);

        assert_eq!(layout.crossings(), 0);
    }

    /// Two sources whose consumers are listed in the opposite order, which crosses the edges.
    const CROSSED_BRANCHES: [(usize, usize); 4] = [(1, 2), (0, 3), (2, 4), (3, 5)];

    #[test]
    fn reordering_a_rank_removes_the_crossings_the_given_order_created() {
        let layout = Layout::build(6, &CROSSED_BRANCHES);

        assert_eq!(layout.crossings(), 0);
    }

    #[test]
    fn the_same_graph_always_lays_out_the_same_way() {
        let first = Layout::build(6, &CROSSED_BRANCHES);
        let second = Layout::build(6, &CROSSED_BRANCHES);

        assert_eq!(first.ranks(), second.ranks());
    }

    #[test]
    fn longest_path_ranks_put_a_join_after_its_longest_branch() {
        let ranks = longest_path_ranks(4, &[(0, 1), (1, 2), (0, 3), (2, 3)]);

        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_layout_with_given_ranks_keeps_them() {
        let layout = Layout::with_ranks(vec![0, 1, 3], &[(0, 1), (0, 2)]);

        assert_eq!(layout.rank_of(2), 3);
        assert_eq!(layout.ranks().len(), 4);
    }

    #[test]
    fn a_chain_passes_a_virtual_slot_in_every_rank_it_skips() {
        let layout = Layout::with_ranks(vec![0, 1, 3], &[(0, 1), (0, 2)]);

        let chain = layout.chain(1);

        assert_eq!(
            chain,
            vec![
                (0, Slot::Node(0)),
                (1, Slot::Virtual(1)),
                (2, Slot::Virtual(1)),
                (3, Slot::Node(2)),
            ]
        );
    }
}
