use std::cmp::Reverse;
use std::collections::{HashMap, HashSet, VecDeque};

/// Where one item sits in the grid.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
pub(super) struct Cell {
    pub(super) column: usize,
    pub(super) row: usize,
}

/// What placement needs to know, with items numbered from zero in declaration order.
pub(super) struct Shape<'shape> {
    pub(super) count: usize,
    /// Each link as the items it runs from and to.
    pub(super) edges: &'shape [(usize, usize)],
    /// Each region's items in declaration order. Every item sits in exactly one region.
    pub(super) regions: &'shape [Vec<usize>],
}

/// Every item's cell, and how many columns and rows the grid has.
#[derive(Debug)]
pub(super) struct Cells {
    pub(super) cells: Vec<Cell>,
    pub(super) columns: usize,
    pub(super) rows: usize,
}

/// The columns and rows one region covers.
#[derive(Debug, Copy, Clone)]
struct Rectangle {
    first_column: usize,
    last_column: usize,
    top: isize,
    bottom: isize,
}

impl Rectangle {
    fn overlaps(&self, other: &Self) -> bool {
        let columns = self.first_column <= other.last_column && other.first_column <= self.last_column;
        let rows = self.top <= other.bottom && other.top <= self.bottom;

        columns && rows
    }
}

/// Places every item in the grid.
///
/// The spine goes on row 0 and the regions holding it sit side by side. Every other region is
/// placed whole, on the nearest rows to what feeds it where it overlaps nothing already placed.
pub(super) fn place_cells(shape: &Shape) -> Cells {
    let mut columns = longest_path_ranks(shape.count, shape.edges);
    let everything: Vec<usize> = (0..shape.count).collect();
    let spine = longest_chain(&everything, &columns, shape.edges);
    let on_spine = spine_regions(shape, &spine);
    separate(&mut columns, shape, &on_spine);

    let mut rows: Vec<Option<isize>> = vec![None; shape.count];
    let mut placed = Vec::new();

    for &region in &on_spine {
        let items = &shape.regions[region];
        let local = region_rows(items, &spine, &columns, shape.edges);
        let rectangle = rectangle_of(&local, &columns, 0);

        apply(&mut rows, &local, 0);
        placed.push(rectangle);
    }

    let mut others: Vec<usize> = (0..shape.regions.len())
        .filter(|region| !on_spine.contains(region))
        .collect();
    others.sort_by_key(|region| first_column(&shape.regions[*region], &columns));

    for region in others {
        let items = &shape.regions[region];
        let local = region_rows(items, &[], &columns, shape.edges);
        let base = feeding_row(items, shape.edges, &rows);
        let offset = free_offset(base, &local, &columns, &placed);
        let rectangle = rectangle_of(&local, &columns, offset);

        apply(&mut rows, &local, offset);
        placed.push(rectangle);
    }

    normalise(&columns, &rows)
}

/// The longest chain through `items`, following links between them only.
///
/// It ends at the deepest item and walks back through the deepest producer each time, taking the
/// earliest declared item whenever two are equally deep.
fn longest_chain(items: &[usize], columns: &[usize], edges: &[(usize, usize)]) -> Vec<usize> {
    let inside: HashSet<usize> = items.iter().copied().collect();
    let Some(mut current) = deepest(items.iter().copied(), columns) else {
        return Vec::new();
    };

    let mut chain = vec![current];
    loop {
        let producers = edges
            .iter()
            .filter(|(from, to)| *to == current && inside.contains(from))
            .map(|(from, _)| *from);
        let Some(previous) = deepest(producers, columns) else {
            break;
        };

        chain.push(previous);
        current = previous;
    }

    chain.reverse();
    chain
}

fn deepest(items: impl Iterator<Item = usize>, columns: &[usize]) -> Option<usize> {
    items.max_by_key(|item| (columns[*item], Reverse(*item)))
}

/// The regions holding spine items, in the order the spine reaches them.
fn spine_regions(shape: &Shape, spine: &[usize]) -> Vec<usize> {
    let region_of: HashMap<usize, usize> = shape
        .regions
        .iter()
        .enumerate()
        .flat_map(|(region, items)| items.iter().map(move |item| (*item, region)))
        .collect();

    let mut regions = Vec::new();
    for item in spine {
        let region = region_of
            .get(item)
            .copied()
            .expect("every spine item sits in a region");

        if !regions.contains(&region) {
            regions.push(region);
        }
    }

    regions
}

/// Shifts each spine region right until it starts after the one before it ends.
///
/// Everything reachable from a shifted region moves with it, so every link still runs to a later
/// column.
fn separate(columns: &mut [usize], shape: &Shape, on_spine: &[usize]) {
    for pair in on_spine.windows(2) {
        let previous = &shape.regions[pair[0]];
        let next = &shape.regions[pair[1]];
        let last = last_column(previous, columns);
        let first = first_column(next, columns);

        if first > last {
            continue;
        }

        let shift = last + 1 - first;
        let moved = reachable(next, shape.edges);

        for item in moved {
            columns[item] += shift;
        }
    }
}

fn reachable(start: &[usize], edges: &[(usize, usize)]) -> HashSet<usize> {
    let mut seen: HashSet<usize> = start.iter().copied().collect();
    let mut waiting: VecDeque<usize> = start.iter().copied().collect();

    while let Some(item) = waiting.pop_front() {
        for (from, to) in edges {
            if *from == item && seen.insert(*to) {
                waiting.push_back(*to);
            }
        }
    }

    seen
}

fn first_column(items: &[usize], columns: &[usize]) -> usize {
    items.iter().map(|item| columns[*item]).min().unwrap_or(0)
}

fn last_column(items: &[usize], columns: &[usize]) -> usize {
    items.iter().map(|item| columns[*item]).max().unwrap_or(0)
}

/// Gives each of a region's items a row relative to the region's own row 0.
///
/// The region's share of the spine, or its own longest chain when it holds none, sits on row 0.
/// Every other item takes the free row nearest the item feeding it.
fn region_rows(
    items: &[usize],
    spine: &[usize],
    columns: &[usize],
    edges: &[(usize, usize)],
) -> Vec<(usize, isize)> {
    let inside: HashSet<usize> = items.iter().copied().collect();
    let spine_part: Vec<usize> = spine
        .iter()
        .copied()
        .filter(|item| inside.contains(item))
        .collect();
    let chain = match spine_part.is_empty() {
        true => longest_chain(items, columns, edges),
        false => spine_part,
    };

    let mut rows: HashMap<usize, isize> = chain.iter().map(|item| (*item, 0)).collect();
    let mut taken: HashSet<(usize, isize)> = chain.iter().map(|item| (columns[*item], 0)).collect();

    let mut rest: Vec<usize> = items
        .iter()
        .copied()
        .filter(|item| !rows.contains_key(item))
        .collect();
    rest.sort_by_key(|item| (columns[*item], *item));

    for item in rest {
        let fed_from = edges
            .iter()
            .filter(|(_, to)| *to == item)
            .find_map(|(from, _)| rows.get(from).copied());
        let base = fed_from.unwrap_or(0);
        let column = columns[item];

        let mut candidates = nearest_rows(base);
        let row = candidates
            .find(|row| !taken.contains(&(column, *row)))
            .expect("some row is always free");

        rows.insert(item, row);
        taken.insert((column, row));
    }

    items.iter().map(|item| (*item, rows[item])).collect()
}

/// Rows to try, nearest first.
///
/// The base row, then one below, one above, two below, and so on.
fn nearest_rows(base: isize) -> impl Iterator<Item = isize> {
    (0..).flat_map(move |distance: isize| {
        let below = base + distance;
        let above = base - distance;
        let tried = match distance {
            0 => vec![below],
            _ => vec![below, above],
        };

        tried.into_iter()
    })
}

/// The row of the first already placed item feeding the region, or row 0 when none is placed.
fn feeding_row(items: &[usize], edges: &[(usize, usize)], rows: &[Option<isize>]) -> isize {
    let inside: HashSet<usize> = items.iter().copied().collect();
    let found = edges
        .iter()
        .filter(|(from, to)| inside.contains(to) && !inside.contains(from))
        .find_map(|(from, _)| rows[*from]);

    found.unwrap_or(0)
}

/// The nearest offset to `base` at which the region overlaps nothing already placed.
fn free_offset(base: isize, local: &[(usize, isize)], columns: &[usize], placed: &[Rectangle]) -> isize {
    let mut candidates = nearest_rows(base);
    let found = candidates.find(|offset| {
        let rectangle = rectangle_of(local, columns, *offset);
        !placed.iter().any(|other| other.overlaps(&rectangle))
    });

    found.expect("some offset is always free")
}

fn rectangle_of(local: &[(usize, isize)], columns: &[usize], offset: isize) -> Rectangle {
    let items: Vec<usize> = local.iter().map(|(item, _)| *item).collect();
    let top = local.iter().map(|(_, row)| row + offset).min().unwrap_or(offset);
    let bottom = local.iter().map(|(_, row)| row + offset).max().unwrap_or(offset);

    Rectangle {
        first_column: first_column(&items, columns),
        last_column: last_column(&items, columns),
        top,
        bottom,
    }
}

fn apply(rows: &mut [Option<isize>], local: &[(usize, isize)], offset: isize) {
    for (item, row) in local {
        rows[*item] = Some(row + offset);
    }
}

/// Shifts every row so the topmost is row 0.
fn normalise(columns: &[usize], rows: &[Option<isize>]) -> Cells {
    let rows: Vec<isize> = rows
        .iter()
        .map(|row| row.expect("every item sits in a region"))
        .collect();
    let top = rows.iter().copied().min().unwrap_or(0);
    let bottom = rows.iter().copied().max().unwrap_or(0);

    let cells = columns
        .iter()
        .zip(&rows)
        .map(|(column, row)| {
            let shifted = row - top;
            let row = usize::try_from(shifted).expect("a shifted row is never negative");

            Cell { column: *column, row }
        })
        .collect();

    let column_count = columns.iter().max().map_or(0, |last| last + 1);
    let row_span = bottom - top + 1;
    let row_count = match columns.is_empty() {
        true => 0,
        false => usize::try_from(row_span).expect("the span of rows is never negative"),
    };

    Cells {
        cells,
        columns: column_count,
        rows: row_count,
    }
}

/// Puts each node one rank after the deepest node it reads.
fn longest_path_ranks(count: usize, edges: &[(usize, usize)]) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn singletons(count: usize) -> Vec<Vec<usize>> {
        (0..count).map(|item| vec![item]).collect()
    }

    fn placed(count: usize, edges: &[(usize, usize)], regions: &[Vec<usize>]) -> Cells {
        let shape = Shape {
            count,
            edges,
            regions,
        };
        place_cells(&shape)
    }

    #[test]
    fn a_chain_lies_on_row_zero() {
        let regions = singletons(3);
        let cells = placed(3, &[(0, 1), (1, 2)], &regions);

        let rows: Vec<usize> = cells.cells.iter().map(|cell| cell.row).collect();
        let columns: Vec<usize> = cells.cells.iter().map(|cell| cell.column).collect();

        assert_eq!(rows, vec![0, 0, 0]);
        assert_eq!(columns, vec![0, 1, 2]);
        assert_eq!((cells.columns, cells.rows), (3, 1));
    }

    /// Region `[3, 4]` starts in the column region `[0, 1, 2]` ends in, and both hold the spine.
    #[test]
    fn every_link_runs_to_a_later_column_after_spine_regions_separate() {
        let edges = [(0, 1), (1, 2), (1, 3), (3, 4)];
        let regions = vec![vec![0, 1, 2], vec![3, 4]];
        let cells = placed(5, &edges, &regions);

        assert_eq!(cells.cells[3].column, 3);

        for (from, to) in edges {
            assert!(cells.cells[from].column < cells.cells[to].column);
        }
    }

    #[test]
    fn a_branch_takes_the_row_below_before_the_row_above() {
        let edges = [(0, 1), (1, 2), (0, 3), (0, 4)];
        let regions = vec![vec![0, 1, 2, 3, 4]];
        let cells = placed(5, &edges, &regions);
        let spine_row = cells.cells[1].row;

        assert_eq!(cells.cells[3].row, spine_row + 1);
        assert_eq!(cells.cells[4].row + 1, spine_row);
    }

    #[test]
    fn a_branch_continues_along_its_own_row() {
        let edges = [(0, 1), (1, 2), (0, 3), (3, 4)];
        let regions = vec![vec![0, 1, 2, 3, 4]];
        let cells = placed(5, &edges, &regions);

        assert_eq!(cells.cells[3].row, cells.cells[4].row);
        assert_ne!(cells.cells[3].row, cells.cells[1].row);
    }

    #[test]
    fn an_off_spine_region_moves_as_one_band() {
        let edges = [(0, 1), (1, 2), (0, 3), (3, 4)];
        let regions = vec![vec![0, 1, 2], vec![3, 4]];
        let cells = placed(5, &edges, &regions);

        assert_eq!(cells.cells[3].row, cells.cells[4].row);
        assert_ne!(cells.cells[3].row, cells.cells[0].row);
    }

    #[test]
    fn an_unplanned_branch_takes_the_next_free_row() {
        let regions = singletons(4);
        let cells = placed(4, &[(0, 1), (1, 2), (0, 3)], &regions);

        assert_eq!(cells.cells[3].column, 1);
        assert_ne!(cells.cells[3].row, cells.cells[1].row);
        assert_eq!(cells.rows, 2);
    }

    #[test]
    fn longest_path_puts_a_join_after_its_longest_branch() {
        let ranks = longest_path_ranks(4, &[(0, 1), (1, 2), (0, 3), (2, 3)]);

        assert_eq!(ranks, vec![0, 1, 2, 3]);
    }
}
