use std::collections::HashMap;

use crate::layout::{Layout, Slot};

/// The smallest gap between two columns.
const COLUMN_GAP: usize = 4;

/// Rows between two things stacked in one column.
const ROW_GAP: usize = 1;

/// A rectangle of cells on the grid.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub(super) struct Rect {
    pub(super) x: usize,
    pub(super) y: usize,
    pub(super) width: usize,
    pub(super) height: usize,
}

impl Rect {
    pub(super) fn right(&self) -> usize {
        self.x + self.width - 1
    }

    pub(super) fn bottom(&self) -> usize {
        self.y + self.height - 1
    }

    pub(super) fn shifted(&self, x: usize, y: usize) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
            ..*self
        }
    }
}

/// A line's run between two neighbouring columns, from the row it leaves by to the row it enters on.
#[derive(Debug, Copy, Clone)]
pub(super) struct Run {
    pub(super) rank: usize,
    pub(super) from: usize,
    pub(super) to: usize,
}

/// Every row lines leave a column by, with the rows those lines enter the next column on.
struct Turning {
    rank: usize,
    from: usize,
    entries: Vec<usize>,
}

impl Turning {
    /// Lines heading down turn first, lowest leaving row first, then lines heading both ways, then
    /// lines heading up, highest leaving row first. This keeps nearby lines from crossing.
    fn sort_key(&self) -> (usize, usize, isize) {
        let lowest = self.entries.iter().copied().min().unwrap_or(self.from);
        let highest = self.entries.iter().copied().max().unwrap_or(self.from);
        let from = self.from as isize;

        match (highest > self.from, lowest < self.from) {
            (true, false) => (self.rank, 0, -from),
            (true, true) => (self.rank, 1, from),
            _ => (self.rank, 2, from),
        }
    }

    /// Whether this turn has to wait for another, because a line leaves by a row this one enters on.
    fn waits_for(&self, other: &Self) -> bool {
        other.rank == self.rank && other.from != self.from && self.entries.contains(&other.from)
    }
}

/// Where every node and virtual slot of one layout ended up.
pub(super) struct Placed {
    pub(super) nodes: Vec<Rect>,
    pub(super) virtuals: HashMap<(usize, usize), Rect>,
    columns: Vec<(usize, usize)>,
    pub(super) width: usize,
    pub(super) height: usize,
}

impl Placed {
    pub(super) fn slot(&self, rank: usize, slot: Slot) -> Rect {
        match slot {
            Slot::Node(node) => self.nodes[node],
            Slot::Virtual(edge) => self.virtuals[&(rank, edge)],
        }
    }

    /// The first cell past the right edge of a column.
    pub(super) fn column_end(&self, rank: usize) -> usize {
        let (x, width) = self.columns[rank];
        x + width
    }

    pub(super) fn shifted(&self, x: usize, y: usize) -> Self {
        let nodes = self.nodes.iter().map(|rect| rect.shifted(x, y)).collect();
        let virtuals = self
            .virtuals
            .iter()
            .map(|(key, rect)| (*key, rect.shifted(x, y)))
            .collect();
        let columns = self
            .columns
            .iter()
            .map(|(column_x, width)| (column_x + x, *width))
            .collect();

        Self {
            nodes,
            virtuals,
            columns,
            width: self.width,
            height: self.height,
        }
    }
}

/// Lays each rank out as a column, left to right, with its slots stacked top to bottom.
///
/// `sizes` holds each node's width and height, and `gaps` the gap before each column. Slots are
/// aligned to the left of their column, and each column is centred against the tallest one.
pub(super) fn place(layout: &Layout, sizes: &[(usize, usize)], gaps: &[usize]) -> Placed {
    let ranks = layout.ranks();

    let mut columns = Vec::with_capacity(ranks.len());
    let mut next = 0;
    for (rank, slots) in ranks.iter().enumerate() {
        if rank > 0 {
            next += gaps[rank];
        }

        let width = column_width(slots, sizes);
        columns.push((next, width));
        next += width;
    }

    let heights: Vec<Vec<usize>> = ranks
        .iter()
        .map(|slots| slots.iter().map(|slot| height_of(*slot, sizes)).collect())
        .collect();
    let tops = stack(&heights);
    let tallest = heights
        .iter()
        .map(|column| column_height(column))
        .max()
        .unwrap_or(0);

    let mut nodes = vec![Rect::default(); sizes.len()];
    let mut virtuals = HashMap::new();

    for (rank, slots) in ranks.iter().enumerate() {
        let (x, width) = columns[rank];

        for (position, slot) in slots.iter().enumerate() {
            let y = tops[rank][position];

            match slot {
                Slot::Node(node) => {
                    let (node_width, node_height) = sizes[*node];
                    nodes[*node] = Rect {
                        x,
                        y,
                        width: node_width,
                        height: node_height,
                    };
                },
                Slot::Virtual(edge) => {
                    let lane = Rect {
                        x,
                        y,
                        width,
                        height: 1,
                    };
                    virtuals.insert((rank, *edge), lane);
                },
            }
        }
    }

    Placed {
        nodes,
        virtuals,
        columns,
        width: next,
        height: tallest,
    }
}

/// The gap before each column, widened to fit the longest label on an edge ending in it.
pub(super) fn gaps(layout: &Layout, edges: &[(usize, usize)], labels: &[usize]) -> Vec<usize> {
    let mut gaps = vec![COLUMN_GAP; layout.ranks().len()];

    for (index, &(_, to)) in edges.iter().enumerate() {
        let rank = layout.rank_of(to);
        let wanted = labels[index] + COLUMN_GAP;
        gaps[rank] = gaps[rank].max(wanted);
    }

    gaps
}

/// Gives every row that lines turn from its own column in the gap after their rank.
///
/// Lines leaving by one row carry one output, so they share a turn. A row that another row's lines
/// enter on turns first, so no line runs along another's row into its turn. Each turn is counted
/// in columns past the first one in the gap.
pub(super) fn turn_columns(runs: &[Run]) -> HashMap<(usize, usize), usize> {
    let mut waiting: Vec<Turning> = Vec::new();
    for run in runs {
        let found = waiting
            .iter_mut()
            .find(|turning| turning.rank == run.rank && turning.from == run.from);

        match found {
            Some(turning) => turning.entries.push(run.to),
            None => waiting.push(Turning {
                rank: run.rank,
                from: run.from,
                entries: vec![run.to],
            }),
        }
    }

    waiting.retain(|turning| turning.entries.iter().any(|to| *to != turning.from));
    waiting.sort_by_key(Turning::sort_key);

    let mut offsets = HashMap::new();
    let mut used: HashMap<usize, usize> = HashMap::new();

    while !waiting.is_empty() {
        let ready = waiting
            .iter()
            .position(|turning| !waiting.iter().any(|other| turning.waits_for(other)));
        let turning = waiting.remove(ready.unwrap_or(0));

        let next = used.entry(turning.rank).or_insert(0);
        offsets.insert((turning.rank, turning.from), *next);
        *next += 1;
    }

    offsets
}

/// Widens the gap after each rank by one column for every turn past the first in it.
pub(super) fn widen(gaps: &[usize], turns: &HashMap<(usize, usize), usize>) -> Vec<usize> {
    let mut widened = gaps.to_vec();

    for (&(rank, _), &offset) in turns {
        let gap = rank + 1;
        widened[gap] = widened[gap].max(gaps[gap] + offset);
    }

    widened
}

fn column_width(slots: &[Slot], sizes: &[(usize, usize)]) -> usize {
    slots
        .iter()
        .map(|slot| match slot {
            Slot::Node(node) => sizes[*node].0,
            Slot::Virtual(_) => 1,
        })
        .max()
        .unwrap_or(1)
}

fn height_of(slot: Slot, sizes: &[(usize, usize)]) -> usize {
    match slot {
        Slot::Node(node) => sizes[node].1,
        Slot::Virtual(_) => 1,
    }
}

fn column_height(heights: &[usize]) -> usize {
    let total: usize = heights.iter().sum();
    total + ROW_GAP * heights.len().saturating_sub(1)
}

/// Stacks each column's slots top to bottom, then centres the column against the tallest one.
fn stack(heights: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let totals: Vec<usize> = heights.iter().map(|column| column_height(column)).collect();
    let tallest = totals.iter().copied().max().unwrap_or(0);

    let mut placed = Vec::with_capacity(heights.len());
    for (column, total) in heights.iter().zip(totals.iter()) {
        let mut next = (tallest - total) / 2;
        let mut tops = Vec::with_capacity(column.len());

        for height in column {
            tops.push(next);
            next += height + ROW_GAP;
        }

        placed.push(tops);
    }

    placed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_run_left_to_right_with_the_gap_between_them() {
        let layout = Layout::build(2, &[(0, 1)]);
        let placed = place(&layout, &[(9, 3), (7, 3)], &[4, 4]);

        assert_eq!(
            placed.nodes[0],
            Rect {
                x: 0,
                y: 0,
                width: 9,
                height: 3
            }
        );
        assert_eq!(
            placed.nodes[1],
            Rect {
                x: 13,
                y: 0,
                width: 7,
                height: 3
            }
        );
        assert_eq!(placed.width, 20);
        assert_eq!(placed.height, 3);
    }

    #[test]
    fn a_shorter_column_is_centred_against_the_tallest() {
        let layout = Layout::build(3, &[(0, 1), (0, 2)]);
        let placed = place(&layout, &[(5, 3), (5, 3), (5, 3)], &[4, 4]);

        assert_eq!(placed.nodes[0].y, 2);
        assert_eq!(placed.nodes[1].y, 0);
        assert_eq!(placed.nodes[2].y, 4);
    }

    #[test]
    fn a_label_widens_the_gap_before_the_column_it_ends_in() {
        let layout = Layout::build(2, &[(0, 1)]);
        let widened = gaps(&layout, &[(0, 1)], &[9]);

        assert_eq!(widened, vec![4, 13]);
    }

    #[test]
    fn lines_from_one_row_share_a_turn_and_other_rows_get_their_own() {
        let runs = [
            Run {
                rank: 0,
                from: 2,
                to: 0,
            },
            Run {
                rank: 0,
                from: 2,
                to: 4,
            },
            Run {
                rank: 0,
                from: 6,
                to: 8,
            },
            Run {
                rank: 0,
                from: 9,
                to: 9,
            },
        ];

        let turns = turn_columns(&runs);

        assert_eq!(turns.len(), 2);
        assert_ne!(turns[&(0, 2)], turns[&(0, 6)]);
        assert!(!turns.contains_key(&(0, 9)));
    }

    #[test]
    fn a_row_another_line_enters_on_turns_before_that_line() {
        let runs = [
            Run {
                rank: 0,
                from: 1,
                to: 5,
            },
            Run {
                rank: 0,
                from: 5,
                to: 0,
            },
        ];

        let turns = turn_columns(&runs);

        assert!(turns[&(0, 5)] < turns[&(0, 1)]);
    }

    #[test]
    fn every_extra_turn_widens_its_gap_by_one() {
        let runs = [
            Run {
                rank: 0,
                from: 1,
                to: 5,
            },
            Run {
                rank: 0,
                from: 3,
                to: 0,
            },
        ];
        let turns = turn_columns(&runs);

        let widened = widen(&[4, 4], &turns);

        assert_eq!(widened, vec![4, 5]);
    }

    #[test]
    fn shifting_moves_every_rectangle_and_column() {
        let layout = Layout::build(2, &[(0, 1)]);
        let placed = place(&layout, &[(9, 3), (7, 3)], &[4, 4]);
        let shifted = placed.shifted(2, 5);

        assert_eq!(shifted.nodes[1].x, 15);
        assert_eq!(shifted.nodes[1].y, 5);
        assert_eq!(shifted.column_end(0), 11);
    }
}
