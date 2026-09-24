mod draw;
mod frame;
mod grid;
mod place;

use std::collections::HashMap;

use self::draw::{Entry, Exit, box_entry, box_port, box_size, draw_box, draw_label, draw_lane, route};
use self::frame::{FRAME_INSET, draw_frame, draw_marker, frame_size, marker_size};
use self::grid::Grid;
use self::place::{Placed, Rect, Run, gaps, place, turn_columns, widen};
use super::RenderOptions;
use super::diagram::{Block, Diagram, Endpoint, Exchange};
use crate::graph::CompiledGraph;
use crate::layout::{Layout, Slot, longest_path_ranks};

/// Something drawn inside a block.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Item {
    Endpoint(Endpoint),
    /// Where a line leaving a box crosses the block's last column, so it stays clear of the
    /// boxes between it and the frame.
    Port(Endpoint),
}

/// One item with everything needed to size and draw it.
struct Placeable {
    item: Item,
    lines: Vec<String>,
    stacked: bool,
    dashed: bool,
}

/// One block laid out on its own, before it is placed among the others.
struct BlockPlan {
    items: Vec<Placeable>,
    edge_links: Vec<Option<usize>>,
    layout: Layout,
    /// How many lines from inside the block enter each item.
    inner_arrivals: Vec<usize>,
    /// How many lines from anywhere enter each item.
    arrivals: Vec<usize>,
    placed: Placed,
    size: (usize, usize),
}

/// One run of a line between two neighbouring columns.
#[derive(Debug, Copy, Clone)]
struct Hop {
    rank: usize,
    exit: Exit,
    entry: Entry,
}

/// A label waiting to be written once every line is down, so no line can cut through it.
struct PendingLabel {
    end: usize,
    y: usize,
    text: &'static str,
}

/// Every block laid out and placed, ready to draw.
struct Arrangement<'render> {
    diagram: &'render Diagram,
    labels: Vec<Option<&'static str>>,
    plans: Vec<BlockPlan>,
    outer_links: Vec<usize>,
    outer: Layout,
    placed: Placed,
    /// Each block's items, placed where the block sits on the grid.
    absolute: Vec<Placed>,
}

/// Draws the graph left to right as boxes joined by lines.
///
/// A planned graph frames each region, stacks a box standing for several workers, and marks
/// where a region's work is dispersed to its workers and aggregated back.
pub fn render_ascii(graph: &CompiledGraph, options: &RenderOptions) -> String {
    if graph.is_empty() {
        return String::new();
    }

    let diagram = Diagram::of(graph);
    let arranged = arrange(&diagram, options);

    let mut grid = Grid::new(arranged.placed.width, arranged.placed.height);
    let mut pending = Vec::new();

    if diagram.planned {
        for (block, rect) in diagram.blocks.iter().zip(&arranged.placed.nodes) {
            let heading = block.heading();
            draw_frame(&mut grid, *rect, &heading);
        }
    }

    for (plan, inner) in arranged.plans.iter().zip(&arranged.absolute) {
        draw_items(&mut grid, plan, inner);
        draw_inner_links(&mut grid, plan, inner, &arranged.labels, &mut pending);
    }

    for lane in arranged.placed.virtuals.values() {
        draw_lane(&mut grid, *lane);
    }

    draw_outer_links(&mut grid, &arranged, &mut pending);

    for label in pending {
        draw_label(&mut grid, label.end, label.y, label.text);
    }

    grid.render()
}

/// Lays out every block, then places the blocks among each other.
///
/// Rows never depend on the gaps between columns, so a first placement finds where every line
/// turns, and the second widens each gap to give every turn its own column.
fn arrange<'render>(diagram: &'render Diagram, options: &RenderOptions) -> Arrangement<'render> {
    let labels: Vec<Option<&'static str>> = diagram
        .links
        .iter()
        .map(|link| link.param.filter(|_| options.show_param_names))
        .collect();

    let plans: Vec<BlockPlan> = diagram
        .blocks
        .iter()
        .map(|block| plan_block(diagram, block, options, &labels))
        .collect();

    let outer_links: Vec<usize> = (0..diagram.links.len())
        .filter(|index| crosses_blocks(diagram, *index))
        .collect();
    let outer_edges: Vec<(usize, usize)> = outer_links
        .iter()
        .map(|index| block_edge(diagram, *index))
        .collect();
    let outer_labels: Vec<usize> = outer_links
        .iter()
        .map(|index| label_width(labels[*index]))
        .collect();

    let outer = Layout::build(plans.len(), &outer_edges);
    let base_gaps = gaps(&outer, &outer_edges, &outer_labels);
    let block_sizes: Vec<(usize, usize)> = plans.iter().map(|plan| plan.size).collect();

    let inset = match diagram.planned {
        true => FRAME_INSET,
        false => 0,
    };

    let rough = place(&outer, &block_sizes, &base_gaps);
    let rough_absolute = absolute_items(&plans, &rough, inset);
    let links = OuterLinks {
        diagram,
        plans: &plans,
        links: &outer_links,
        layout: &outer,
    };
    let rough_hops = links.hops(&rough, &rough_absolute);
    let turns = turns_of(&rough_hops);
    let outer_gaps = widen(&base_gaps, &turns);

    let placed = place(&outer, &block_sizes, &outer_gaps);
    let absolute = absolute_items(&plans, &placed, inset);

    Arrangement {
        diagram,
        labels,
        plans,
        outer_links,
        outer,
        placed,
        absolute,
    }
}

/// Moves each block's items to where the block sits, inside its frame.
fn absolute_items(plans: &[BlockPlan], placed: &Placed, inset: usize) -> Vec<Placed> {
    plans
        .iter()
        .zip(&placed.nodes)
        .map(|(plan, rect)| plan.placed.shifted(rect.x + inset, rect.y + inset))
        .collect()
}

fn crosses_blocks(diagram: &Diagram, index: usize) -> bool {
    let link = &diagram.links[index];
    diagram.block_of(link.from) != diagram.block_of(link.to)
}

fn block_edge(diagram: &Diagram, index: usize) -> (usize, usize) {
    let link = &diagram.links[index];
    let from = diagram.block_of(link.from);
    let to = diagram.block_of(link.to);

    (from, to)
}

fn label_width(label: Option<&str>) -> usize {
    label.map_or(0, |label| label.chars().count())
}

/// Lays out one block's items in their own columns.
///
/// A disperse takes the first column and an aggregate the last. When a box read from outside the
/// block sits before the last box column, every box read from outside gets a port in the last
/// column, so each line leaves the block on its own row.
fn plan_block(
    diagram: &Diagram,
    block: &Block,
    options: &RenderOptions,
    labels: &[Option<&'static str>],
) -> BlockPlan {
    let mut items = placeables(diagram, block, options);
    let (mut edges, mut edge_links) = inner_edges(diagram, block, &items);

    let mut ranks = longest_path_ranks(items.len(), &edges);
    let deepest_box = items
        .iter()
        .zip(&ranks)
        .filter(|(placeable, _)| matches!(placeable.item, Item::Endpoint(Endpoint::Box(_))))
        .map(|(_, rank)| *rank)
        .max()
        .unwrap_or(0);
    let last = deepest_box + 1;

    for (position, placeable) in items.iter().enumerate() {
        if matches!(placeable.item, Item::Endpoint(Endpoint::Aggregate(_))) {
            ranks[position] = last;
        }
    }

    let leaving = leaving_boxes(diagram, block);
    let needs_ports = leaving.iter().any(|source| {
        let position = position_of(&items, Item::Endpoint(*source));
        ranks[position] < deepest_box
    });

    if needs_ports {
        for source in leaving {
            let from = position_of(&items, Item::Endpoint(source));

            items.push(Placeable {
                item: Item::Port(source),
                lines: Vec::new(),
                stacked: false,
                dashed: false,
            });
            ranks.push(last);
            edges.push((from, items.len() - 1));
            edge_links.push(None);
        }
    }

    let mut inner_arrivals = vec![0; items.len()];
    for (_, to) in &edges {
        inner_arrivals[*to] += 1;
    }

    let arrivals: Vec<usize> = items
        .iter()
        .zip(&inner_arrivals)
        .map(|(placeable, inner)| inner + outer_arrivals(diagram, placeable.item))
        .collect();
    let sizes: Vec<(usize, usize)> = items
        .iter()
        .zip(&arrivals)
        .map(|(placeable, count)| size_of(placeable, *count))
        .collect();

    let layout = Layout::with_ranks(ranks, &edges);
    let edge_labels: Vec<usize> = edge_links
        .iter()
        .map(|link| {
            let label = link.and_then(|index| labels[index]);
            label_width(label)
        })
        .collect();
    let base_gaps = gaps(&layout, &edges, &edge_labels);
    let rough = place(&layout, &sizes, &base_gaps);

    let mut plan = BlockPlan {
        items,
        edge_links,
        layout,
        inner_arrivals,
        arrivals,
        placed: rough,
        size: (0, 0),
    };

    let rough_hops = plan.hops(&plan.placed);
    let turns = turns_of(&rough_hops);
    let column_gaps = widen(&base_gaps, &turns);
    plan.placed = place(&plan.layout, &sizes, &column_gaps);

    let inner = (plan.placed.width, plan.placed.height);
    plan.size = match diagram.planned {
        true => {
            let heading = block.heading();
            frame_size(inner, &heading)
        },
        false => inner,
    };

    plan
}

/// Every box in the block that a link leaving the block starts at, in link order.
fn leaving_boxes(diagram: &Diagram, block: &Block) -> Vec<Endpoint> {
    let mut sources = Vec::new();

    for link in &diagram.links {
        let leaves = diagram.block_of(link.from) == block.id && diagram.block_of(link.to) != block.id;
        let is_box = matches!(link.from, Endpoint::Box(_));

        if leaves && is_box && !sources.contains(&link.from) {
            sources.push(link.from);
        }
    }

    sources
}

/// How many links from other blocks end at an item.
fn outer_arrivals(diagram: &Diagram, item: Item) -> usize {
    let Item::Endpoint(endpoint) = item else {
        return 0;
    };

    diagram
        .links
        .iter()
        .filter(|link| link.to == endpoint && diagram.block_of(link.from) != diagram.block_of(endpoint))
        .count()
}

fn placeables(diagram: &Diagram, block: &Block, options: &RenderOptions) -> Vec<Placeable> {
    let mut items = Vec::new();

    let disperse = diagram.exchange(Endpoint::Disperse(block.id));
    if let Some(exchange) = disperse {
        items.push(marker(exchange));
    }

    for node in &block.boxes {
        let entry = &diagram.boxes[node.index()];

        items.push(Placeable {
            item: Item::Endpoint(Endpoint::Box(*node)),
            lines: entry.lines(options),
            stacked: entry.width > 1,
            dashed: entry.crossing.is_some(),
        });
    }

    let aggregate = diagram.exchange(Endpoint::Aggregate(block.id));
    if let Some(exchange) = aggregate {
        items.push(marker(exchange));
    }

    items
}

fn marker(exchange: &Exchange) -> Placeable {
    let lines = exchange.lines();

    Placeable {
        item: Item::Endpoint(exchange.endpoint()),
        lines: lines.to_vec(),
        stacked: false,
        dashed: false,
    }
}

/// Every link with both ends in this block, as item positions, alongside the link it came from.
fn inner_edges(
    diagram: &Diagram,
    block: &Block,
    items: &[Placeable],
) -> (Vec<(usize, usize)>, Vec<Option<usize>>) {
    let mut edges = Vec::new();
    let mut edge_links = Vec::new();

    for (index, link) in diagram.links.iter().enumerate() {
        let inside = diagram.block_of(link.from) == block.id && diagram.block_of(link.to) == block.id;
        if !inside {
            continue;
        }

        let from = position_of(items, Item::Endpoint(link.from));
        let to = position_of(items, Item::Endpoint(link.to));
        edges.push((from, to));
        edge_links.push(Some(index));
    }

    (edges, edge_links)
}

fn position_of(items: &[Placeable], item: Item) -> usize {
    let found = items.iter().position(|placeable| placeable.item == item);
    found.expect("every endpoint in a block is one of its items")
}

/// The size an item needs, grown so every line entering it has a row of its own.
fn size_of(placeable: &Placeable, arrivals: usize) -> (usize, usize) {
    let rows = placeable.lines.len();
    let extra = arrivals.saturating_sub(rows);

    match placeable.item {
        Item::Endpoint(Endpoint::Box(_)) => {
            let (width, height) = box_size(&placeable.lines, placeable.stacked);
            (width, height + extra)
        },
        Item::Endpoint(_) => {
            let (width, height) = marker_size(&placeable.lines);
            (width, height + extra)
        },
        Item::Port(_) => (1, 1),
    }
}

/// Numbers the lines arriving at each target from the top down, by the row each arrives from.
fn arrival_order<Target: Eq>(targets: &[Target], rows: &[usize]) -> Vec<usize> {
    (0..targets.len())
        .map(|line| {
            let key = (rows[line], line);

            (0..targets.len())
                .filter(|other| targets[*other] == targets[line] && (rows[*other], *other) < key)
                .count()
        })
        .collect()
}

fn turns_of(hops: &[Vec<Hop>]) -> HashMap<(usize, usize), usize> {
    let runs: Vec<Run> = hops
        .iter()
        .flatten()
        .map(|hop| Run {
            rank: hop.rank,
            from: hop.exit.y,
            to: hop.entry.y,
        })
        .collect();

    turn_columns(&runs)
}

/// Draws one line's runs, each turning in the column its gap gave its row.
fn draw_hops(grid: &mut Grid, placed: &Placed, turns: &HashMap<(usize, usize), usize>, hops: &[Hop]) {
    for hop in hops {
        let offset = turns.get(&(hop.rank, hop.exit.y)).copied().unwrap_or(0);
        let turn = placed.column_end(hop.rank) + 1 + offset;

        route(grid, hop.exit, turn, hop.entry);
    }
}

fn draw_items(grid: &mut Grid, plan: &BlockPlan, placed: &Placed) {
    for (position, placeable) in plan.items.iter().enumerate() {
        let rect = placed.nodes[position];

        match placeable.item {
            Item::Endpoint(Endpoint::Box(_)) => {
                draw_box(grid, rect, &placeable.lines, placeable.dashed, placeable.stacked);
            },
            Item::Endpoint(_) => draw_marker(grid, rect, &placeable.lines),
            Item::Port(_) => {},
        }
    }

    for lane in placed.virtuals.values() {
        draw_lane(grid, *lane);
    }
}

impl BlockPlan {
    /// Every inner link's runs, each entering its target on a row of its own.
    fn hops(&self, placed: &Placed) -> Vec<Vec<Hop>> {
        let chains: Vec<Vec<(usize, Slot)>> = (0..self.edge_links.len())
            .map(|edge| self.layout.chain(edge))
            .collect();
        let targets: Vec<Slot> = chains.iter().map(|chain| chain[chain.len() - 1].1).collect();
        let rows: Vec<usize> = chains
            .iter()
            .map(|chain| {
                let (rank, slot) = chain[chain.len() - 2];
                self.exit_at(placed, rank, slot).y
            })
            .collect();
        let order = arrival_order(&targets, &rows);

        chains
            .iter()
            .zip(&order)
            .map(|(chain, index)| {
                chain
                    .windows(2)
                    .map(|pair| Hop {
                        rank: pair[0].0,
                        exit: self.exit_at(placed, pair[0].0, pair[0].1),
                        entry: self.entry_at(placed, pair[1].0, pair[1].1, *index),
                    })
                    .collect()
            })
            .collect()
    }

    fn exit_at(&self, placed: &Placed, rank: usize, slot: Slot) -> Exit {
        match slot {
            Slot::Node(position) => self.exit_of(placed, position),
            Slot::Virtual(_) => {
                let lane = placed.slot(rank, slot);
                lane_exit(lane)
            },
        }
    }

    fn entry_at(&self, placed: &Placed, rank: usize, slot: Slot, index: usize) -> Entry {
        match slot {
            Slot::Node(position) => self.entry_of(placed, position, index),
            Slot::Virtual(_) => {
                let lane = placed.slot(rank, slot);
                lane_entry(lane)
            },
        }
    }

    fn exit_of(&self, placed: &Placed, position: usize) -> Exit {
        let rect = placed.nodes[position];
        let placeable = &self.items[position];

        match placeable.item {
            Item::Endpoint(Endpoint::Box(_)) => Exit {
                x: rect.right(),
                y: box_port(rect, placeable.stacked),
                tee: true,
            },
            Item::Endpoint(_) => Exit {
                x: rect.right(),
                y: rect.y,
                tee: false,
            },
            Item::Port(_) => Exit {
                x: rect.x,
                y: rect.y,
                tee: false,
            },
        }
    }

    /// Where the line at `index` of those entering an item arrives.
    ///
    /// Lines enter a marker on its rows from the top down, and the marker is sized to have a row
    /// for each of them.
    fn entry_of(&self, placed: &Placed, position: usize, index: usize) -> Entry {
        let rect = placed.nodes[position];
        let placeable = &self.items[position];

        match placeable.item {
            Item::Endpoint(Endpoint::Box(_)) => {
                let count = self.arrivals[position];

                Entry {
                    x: rect.x,
                    y: box_entry(rect, placeable.stacked, count, index),
                    arrow: true,
                }
            },
            Item::Endpoint(_) => Entry {
                x: rect.x,
                y: rect.y + index,
                arrow: true,
            },
            Item::Port(_) => Entry {
                x: rect.x,
                y: rect.y,
                arrow: false,
            },
        }
    }
}

fn lane_exit(rect: Rect) -> Exit {
    Exit {
        x: rect.right(),
        y: rect.y,
        tee: false,
    }
}

fn lane_entry(rect: Rect) -> Entry {
    Entry {
        x: rect.x,
        y: rect.y,
        arrow: false,
    }
}

fn draw_inner_links(
    grid: &mut Grid,
    plan: &BlockPlan,
    placed: &Placed,
    labels: &[Option<&'static str>],
    pending: &mut Vec<PendingLabel>,
) {
    let hops = plan.hops(placed);
    let turns = turns_of(&hops);

    for (link_hops, link) in hops.iter().zip(&plan.edge_links) {
        draw_hops(grid, placed, &turns, link_hops);

        let label = link.and_then(|index| labels[index]);
        let last = link_hops.last();

        if let (Some(text), Some(hop)) = (label, last) {
            pending.push(PendingLabel {
                end: hop.entry.x,
                y: hop.entry.y,
                text,
            });
        }
    }
}

fn draw_outer_links(grid: &mut Grid, arranged: &Arrangement, pending: &mut Vec<PendingLabel>) {
    let links = OuterLinks {
        diagram: arranged.diagram,
        plans: &arranged.plans,
        links: &arranged.outer_links,
        layout: &arranged.outer,
    };
    let hops = links.hops(&arranged.placed, &arranged.absolute);
    let turns = turns_of(&hops);

    for (link_hops, index) in hops.iter().zip(&arranged.outer_links) {
        draw_hops(grid, &arranged.placed, &turns, link_hops);

        let label = arranged.labels[*index];
        let last = link_hops.last();

        if let (Some(text), Some(hop)) = (label, last) {
            let link = &arranged.diagram.links[*index];
            let target = arranged.diagram.block_of(link.to);

            pending.push(PendingLabel {
                end: arranged.placed.nodes[target].x,
                y: hop.entry.y,
                text,
            });
        }
    }
}

/// Everything needed to route the links that run between blocks.
struct OuterLinks<'render> {
    diagram: &'render Diagram,
    plans: &'render [BlockPlan],
    links: &'render [usize],
    layout: &'render Layout,
}

impl OuterLinks<'_> {
    /// Every link's runs between blocks, each entering its target on a row of its own.
    fn hops(&self, placed: &Placed, absolute: &[Placed]) -> Vec<Vec<Hop>> {
        let chains: Vec<Vec<(usize, Slot)>> = (0..self.links.len())
            .map(|edge| self.layout.chain(edge))
            .collect();
        let starts: Vec<Exit> = self
            .links
            .iter()
            .map(|index| {
                let link = &self.diagram.links[*index];
                self.leaving(absolute, link.from)
            })
            .collect();
        let targets: Vec<Endpoint> = self
            .links
            .iter()
            .map(|index| self.diagram.links[*index].to)
            .collect();
        let rows: Vec<usize> = chains
            .iter()
            .zip(&starts)
            .map(|(chain, start)| {
                let (rank, slot) = chain[chain.len() - 2];

                match slot {
                    Slot::Node(_) => start.y,
                    Slot::Virtual(_) => placed.slot(rank, slot).y,
                }
            })
            .collect();
        let order = arrival_order(&targets, &rows);

        let mut hops = Vec::with_capacity(chains.len());
        for (edge, chain) in chains.iter().enumerate() {
            let end = self.entering(absolute, targets[edge], order[edge]);
            let link_hops = chain
                .windows(2)
                .map(|pair| {
                    let (from_rank, from_slot) = pair[0];
                    let (to_rank, to_slot) = pair[1];

                    let exit = match from_slot {
                        Slot::Node(_) => starts[edge],
                        Slot::Virtual(_) => {
                            let lane = placed.slot(from_rank, from_slot);
                            lane_exit(lane)
                        },
                    };
                    let entry = match to_slot {
                        Slot::Node(_) => end,
                        Slot::Virtual(_) => {
                            let lane = placed.slot(to_rank, to_slot);
                            lane_entry(lane)
                        },
                    };

                    Hop {
                        rank: from_rank,
                        exit,
                        entry,
                    }
                })
                .collect();

            hops.push(link_hops);
        }

        hops
    }

    /// Where a link leaves its block, through the box's port when it has one.
    fn leaving(&self, absolute: &[Placed], endpoint: Endpoint) -> Exit {
        let block = self.diagram.block_of(endpoint);
        let plan = &self.plans[block];
        let port = plan
            .items
            .iter()
            .position(|placeable| placeable.item == Item::Port(endpoint));

        let position = match port {
            Some(position) => position,
            None => position_of(&plan.items, Item::Endpoint(endpoint)),
        };

        plan.exit_of(&absolute[block], position)
    }

    /// Where the link at `index` of those entering an endpoint from other blocks arrives.
    ///
    /// These lines take the rows after any line from inside the block.
    fn entering(&self, absolute: &[Placed], endpoint: Endpoint, index: usize) -> Entry {
        let block = self.diagram.block_of(endpoint);
        let plan = &self.plans[block];
        let position = position_of(&plan.items, Item::Endpoint(endpoint));
        let offset = plan.inner_arrivals[position];

        plan.entry_of(&absolute[block], position, offset + index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::fixtures;

    fn bare() -> RenderOptions {
        RenderOptions {
            show_identifier: false,
            show_param_names: false,
        }
    }

    #[test]
    fn a_source_feeding_a_sink_draws_two_boxes_side_by_side() {
        let specs = fixtures::source_to_sink();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "┌───────┐    ┌─────┐\n",
                "│ input ├────▶ out │\n",
                "└───────┘    └─────┘\n",
            )
        );
    }

    #[test]
    fn an_edge_skipping_a_column_follows_a_lane_through_it() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "             ┌──────┐    ┌───────┐\n",
                "┌───────┐ ┌──▶ blur ├────▶ merge │    ┌─────┐\n",
                "│ input ├─┤  └──────┘ ┌──▶       ├────▶ out │\n",
                "└───────┘ │           │  └───────┘    └─────┘\n",
                "          └───────────┘\n",
            )
        );
    }

    #[test]
    fn a_wide_region_draws_exactly() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "                         ┌ region 1 ×4 · chunk · by chunk ──────────────────────┐\n",
                "┌ region 0 · stream ┐    │                                                      │    ┌ region 2 · stream (granularity) ┐\n",
                "│                   │    │                        ┌─────────┐                   │    │                                 │\n",
                "│ ┌───────┐         │ ┌──┼─▶ ◁ 1:4 ────────────┐  │ denoise │┐ ┌──▶ ▷ 4:1 ──────┼─┐  │ ┌─────┐                         │\n",
                "│ │ input ├─────────┼─┘  │   disperse slot 0   └──▶   ×4    │├─┘    aggregate   │ └──┼─▶ out │                         │\n",
                "│ └───────┘         │    │   unbounded            └─────────┘│      bounded     │    │ └─────┘                         │\n",
                "│                   │    │                         └─────────┘                  │    │                                 │\n",
                "└───────────────────┘    │                                                      │    └─────────────────────────────────┘\n",
                "                         └──────────────────────────────────────────────────────┘\n",
            )
        );
    }

    #[test]
    fn a_crossing_is_drawn_dashed_with_its_direction() {
        let graph = fixtures::placed_round_trip();
        let rendered = render_ascii(&graph, &RenderOptions::default());

        assert!(rendered.contains("upload(input)"));
        assert!(rendered.contains("host→gpu"));
        assert!(rendered.contains('┆'));
    }

    #[test]
    fn param_names_label_the_edges_only_when_asked_for() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let shown = RenderOptions {
            show_identifier: false,
            show_param_names: true,
        };

        let labelled = render_ascii(&graph, &shown);
        let bare_render = render_ascii(&graph, &bare());

        assert!(labelled.contains("reference"));
        assert!(!bare_render.contains("reference"));
    }

    #[test]
    fn the_identifier_is_printed_only_when_asked_for() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);

        let shown = render_ascii(&graph, &RenderOptions::default());
        let hidden = render_ascii(&graph, &bare());

        assert!(shown.contains("test/file"));
        assert!(!hidden.contains("test/file"));
    }

    #[test]
    fn an_unplanned_graph_draws_no_frames() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert!(!rendered.contains("region"));
    }

    #[test]
    fn an_empty_graph_renders_nothing() {
        let graph = crate::graph::CompiledGraph::new(Vec::new());
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(rendered, "");
    }

    #[test]
    fn crossed_branches_draw_every_stage() {
        let specs = fixtures::crossed_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        for name in [
            "first_source",
            "second_source",
            "first_blur",
            "second_blur",
            "first_out",
            "second_out",
        ] {
            assert!(rendered.contains(name), "{name} should be drawn");
        }
    }

    #[test]
    fn a_two_digit_width_keeps_its_markers_and_badge() {
        let specs = fixtures::wide_pipeline(16);
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert!(rendered.contains("×16"));
        assert!(rendered.contains("◁ 1:16"));
        assert!(rendered.contains("▷ 16:1"));
    }

    /// `first` is read from outside its region while it sits in the region's first column, so
    /// its outgoing line has to pass the boxes after it without cutting through them.
    #[test]
    fn a_line_leaving_a_middle_column_passes_clear_of_later_boxes() {
        let specs = fixtures::middle_exit_pipeline();
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "┌ region 0 · stream ────────────────────┐\n",
                "│                                       │\n",
                "│              ┌──────┐    ┌─────┐      │\n",
                "│ ┌───────┐ ┌──▶ blur ├────▶ out │      │\n",
                "│ │ first ├─┤  └──────┘    └─────┘ ┌────┼─┐\n",
                "│ └───────┘ │                      │    │ │   ┌ region 2 · stream ─────────────────────────────────┐\n",
                "│           └──────────────────────┘    │ │   │                                                    │\n",
                "│                                       │ └───┼─▶ ◁ 1:1 ────────────┐  ┌───────┐    ┌────────────┐ │\n",
                "└───────────────────────────────────────┘  ┌──┼─▶ disperse slot 0   └──▶ merge ├────▶ merged_out │ │\n",
                "                                           │  │   unbounded            └───────┘    └────────────┘ │\n",
                "┌ region 1 · stream ┐                      │  │                                                    │\n",
                "│                   │                      │  └────────────────────────────────────────────────────┘\n",
                "│ ┌────────┐        │                      │\n",
                "│ │ second ├────────┼──────────────────────┘\n",
                "│ └────────┘        │\n",
                "│                   │\n",
                "└───────────────────┘\n",
            )
        );
    }

    #[test]
    fn each_named_input_enters_on_its_own_row_under_its_own_label() {
        let specs = fixtures::named_inputs_pipeline();
        let graph = fixtures::built(&specs);
        let labelled = RenderOptions {
            show_identifier: false,
            show_param_names: true,
        };
        let rendered = render_ascii(&graph, &labelled);

        assert_eq!(
            rendered,
            concat!(
                "┌───────┐\n",
                "│ first ├──┐          ┌─────────┐\n",
                "└───────┘  └──────base▶ matched │         ┌─────┐\n",
                "            ┌──overlay▶         ├────input▶ out │\n",
                "┌────────┐  │         └─────────┘         └─────┘\n",
                "│ second ├──┘\n",
                "└────────┘\n",
            )
        );
    }

    #[test]
    fn lines_leaving_one_region_for_two_others_never_merge() {
        let specs = fixtures::mixed_inputs_pipeline();
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "                                     ┌ region 1 ×4 · chunk · by chunk ───────────────────┐\n",
                "┌ region 0 · stream ───────────┐     │                                                   │    ┌ region 2 · frame ────────────────────────────────────┐\n",
                "│                              │     │                        ┌──────┐                   │    │                                                      │\n",
                "│              ┌────────┐      │  ┌──┼─▶ ◁ 1:4 ────────────┐  │ wide │┐ ┌──▶ ▷ 4:1 ──────┼────┼─▶ ◁ 1:1 ────────────┐  ┌────────────────┐    ┌─────┐ │\n",
                "│ ┌───────┐ ┌──▶ narrow ├──────┼─┐│  │   disperse slot 0   └──▶  ×4  │├─┘    aggregate   │ ┌──┼─▶ disperse slot 0   │  │     merge      │ ┌──▶ out │ │\n",
                "│ │ input ├─┤  └────────┘      │ ││  │   unbounded            └──────┘│      bounded     │ │  │   unbounded         └──▶ (mixed inputs) ├─┘  └─────┘ │\n",
                "│ └───────┘ │             ┌────┼─┼┘  │                         └──────┘                  │ │  │                        └────────────────┘            │\n",
                "│           └─────────────┘    │ │   │                                                   │ │  │                                                      │\n",
                "│                              │ │   └───────────────────────────────────────────────────┘ │  └──────────────────────────────────────────────────────┘\n",
                "└──────────────────────────────┘ │                                                         │\n",
                "                                 └─────────────────────────────────────────────────────────┘\n",
            )
        );
    }

    fn multi_region_graphs() -> Vec<CompiledGraph> {
        let all_specs = [
            fixtures::two_exit_pipeline(),
            fixtures::mixed_inputs_pipeline(),
            fixtures::middle_exit_pipeline(),
            fixtures::many_region_pipeline(),
        ];

        all_specs.iter().map(|specs| fixtures::planned(specs)).collect()
    }

    fn overlaps(first: Rect, second: Rect) -> bool {
        let across = first.x <= second.right() && second.x <= first.right();
        let down = first.y <= second.bottom() && second.y <= first.bottom();

        across && down
    }

    #[test]
    fn a_planned_graphs_blocks_never_overlap() {
        for graph in multi_region_graphs() {
            let diagram = Diagram::of(&graph);
            let arranged = arrange(&diagram, &RenderOptions::default());
            let blocks = &arranged.placed.nodes;

            assert!(blocks.len() > 1);

            for (index, first) in blocks.iter().enumerate() {
                for second in blocks.iter().skip(index + 1) {
                    assert!(!overlaps(*first, *second), "{first:?} overlaps {second:?}");
                }
            }
        }
    }

    #[test]
    fn every_box_sits_inside_its_block() {
        for graph in multi_region_graphs() {
            let diagram = Diagram::of(&graph);
            let arranged = arrange(&diagram, &RenderOptions::default());

            for (block, items) in arranged.placed.nodes.iter().zip(&arranged.absolute) {
                for item in &items.nodes {
                    let inside_across = item.x > block.x && item.right() < block.right();
                    let inside_down = item.y > block.y && item.bottom() < block.bottom();

                    assert!(inside_across && inside_down, "{item:?} is outside {block:?}");
                }
            }
        }
    }
}
