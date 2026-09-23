mod ascii;
mod execution;
mod grid;
mod mermaid;
mod svg;

use std::collections::HashMap;

pub use self::ascii::render_ascii;
pub use self::mermaid::render_mermaid;
pub use self::svg::render_svg;
use crate::graph::{CompiledGraph, Node, NodeId};
use crate::layout::{Layout, Slot};
use crate::plan::{NodePlan, Routing};

/// Which graph a renderer draws.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub enum RenderView {
    /// The stages the pipeline declared.
    #[default]
    Stages,
    /// How those stages will be run.
    Execution,
}

/// What a rendered graph shows.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Print each stage's `package/stage` identifier under its name.
    pub show_identifier: bool,
    /// Label each edge with the input param it fills.
    pub show_param_names: bool,
    /// Which graph to draw.
    pub view: RenderView,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            show_identifier: true,
            show_param_names: true,
            view: RenderView::default(),
        }
    }
}

/// The lines of text a node's box holds.
///
/// When the node carries a plan, a final line names how it will run, its width and routing mode
/// included whenever that width is above one.
pub(crate) fn labels_for(node: &Node, plan: Option<&NodePlan>, options: &RenderOptions) -> Vec<String> {
    let mut lines = vec![node.name().to_string()];

    if options.show_identifier {
        lines.push(node.identifier().to_string());
    }

    if let Some(residency) = node.residency() {
        lines.push(residency.to_string());
    }

    if let Some(plan) = plan {
        let routing = routing_text(plan.routing());
        let label = match plan.width() {
            1 => plan.parallelism().to_string(),
            width => format!("{} x{width} {routing}", plan.parallelism()),
        };

        lines.push(label);
    }

    lines
}

fn routing_text(routing: Routing) -> &'static str {
    match routing {
        Routing::ByFrame => "by frame",
        Routing::ByChunk => "by chunk",
    }
}

/// Lays each rank out left to right, then centres it against the widest rank.
pub(crate) fn place_ranks(widths: &[Vec<usize>], gap: usize) -> Vec<Vec<usize>> {
    let totals: Vec<usize> = widths.iter().map(|rank| total_width(rank, gap)).collect();
    let widest = totals.iter().copied().max().unwrap_or(0);

    let mut placed = Vec::with_capacity(widths.len());
    for (rank, total) in widths.iter().zip(totals.iter()) {
        let mut next = (widest - total) / 2;
        let mut positions = Vec::with_capacity(rank.len());

        for width in rank {
            positions.push(next);
            next += width + gap;
        }

        placed.push(positions);
    }

    placed
}

fn total_width(rank: &[usize], gap: usize) -> usize {
    let boxes: usize = rank.iter().sum();
    boxes + gap * rank.len().saturating_sub(1)
}

/// Where one slot ended up, in whatever unit the backend is working in.
#[derive(Debug, Copy, Clone)]
pub(crate) struct Placement {
    pub(crate) x: usize,
    pub(crate) width: usize,
    pub(crate) top: usize,
    pub(crate) height: usize,
}

impl Placement {
    pub(crate) fn centre(&self) -> usize {
        self.x + self.width / 2
    }

    pub(crate) fn bottom(&self) -> usize {
        self.top + self.height - 1
    }
}

/// The spacing a backend wants between the things it draws.
pub(crate) struct Metrics {
    pub(crate) column_gap: usize,
    pub(crate) rank_gap: usize,
    pub(crate) virtual_width: usize,
}

/// Every slot's placement, and the canvas they need.
pub(crate) struct Placements {
    slots: HashMap<(usize, Slot), Placement>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Placements {
    pub(crate) fn get(&self, rank: usize, slot: Slot) -> Placement {
        self.slots[&(rank, slot)]
    }
}

/// Turns the layout into coordinates, given the size each stage's box wants to be.
pub(crate) fn place_slots(
    layout: &Layout,
    sizes: &HashMap<NodeId, (usize, usize)>,
    metrics: &Metrics,
) -> Placements {
    let widths: Vec<Vec<usize>> = layout
        .ranks()
        .iter()
        .map(|rank| {
            rank.iter()
                .map(|slot| match slot {
                    Slot::Stage(id) => sizes[id].0,
                    Slot::Virtual(_) => metrics.virtual_width,
                })
                .collect()
        })
        .collect();

    let columns = place_ranks(&widths, metrics.column_gap);
    let heights = rank_heights(layout, sizes, metrics);

    let mut tops = Vec::with_capacity(heights.len());
    let mut next = 0;
    for height in &heights {
        tops.push(next);
        next += height + metrics.rank_gap;
    }

    let mut slots = HashMap::new();
    let mut width = 0;

    for (rank, entries) in layout.ranks().iter().enumerate() {
        for (position, slot) in entries.iter().enumerate() {
            let height = match slot {
                Slot::Stage(id) => sizes[id].1,
                Slot::Virtual(_) => heights[rank],
            };

            let placement = Placement {
                x: columns[rank][position],
                width: widths[rank][position],
                top: tops[rank],
                height,
            };

            width = width.max(placement.x + placement.width);
            slots.insert((rank, *slot), placement);
        }
    }

    let last_top = tops.last().copied().unwrap_or(0);
    let last_height = heights.last().copied().unwrap_or(0);

    Placements {
        slots,
        width,
        height: last_top + last_height,
    }
}

/// A rank is as tall as its tallest box, and a rank of only virtual slots is one unit tall.
fn rank_heights(layout: &Layout, sizes: &HashMap<NodeId, (usize, usize)>, metrics: &Metrics) -> Vec<usize> {
    layout
        .ranks()
        .iter()
        .map(|rank| {
            rank.iter()
                .filter_map(|slot| match slot {
                    Slot::Stage(id) => Some(sizes[id].1),
                    Slot::Virtual(_) => None,
                })
                .max()
                .unwrap_or(metrics.virtual_width)
        })
        .collect()
}

/// The slots one edge passes through, from the stage it reads down to the stage that reads it.
pub(crate) fn chain_of(
    layout: &Layout,
    graph: &CompiledGraph,
    node: NodeId,
    index: usize,
) -> Vec<(usize, Slot)> {
    let edge = &graph.node(node).inputs()[index];
    let from = layout.rank_of_stage(edge.source());
    let to = layout.rank_of_stage(node);

    let id = crate::graph::EdgeId::new(node, index);
    let mut chain = vec![(from, Slot::Stage(edge.source()))];

    for rank in (from + 1)..to {
        chain.push((rank, Slot::Virtual(id)));
    }

    chain.push((to, Slot::Stage(node)));
    chain
}
