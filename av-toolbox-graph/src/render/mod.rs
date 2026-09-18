mod grid;
mod mermaid;

pub use self::mermaid::render_mermaid;
use crate::graph::Node;

/// What a rendered graph shows.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Print each stage's `package/stage` identifier under its name.
    pub show_identifier: bool,
    /// Label each edge with the input param it fills.
    pub show_param_names: bool,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            show_identifier: true,
            show_param_names: true,
        }
    }
}

/// The lines of text a node's box holds.
pub(crate) fn labels_for(node: &Node, options: &RenderOptions) -> Vec<String> {
    let mut lines = vec![node.name().to_string()];

    if options.show_identifier {
        lines.push(node.identifier().to_string());
    }

    lines
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
