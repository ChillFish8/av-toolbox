mod cells;
mod draw;
mod frame;
mod grid;
#[cfg(test)]
mod tests;
mod tracks;

use std::collections::{HashMap, HashSet};

use self::cells::{Cell, Cells, Shape, place_cells};
use self::draw::{box_size, draw_box, draw_label, draw_route};
use self::frame::{LEGEND, draw_frame, draw_frame_text, draw_marker, marker_size};
use self::grid::Grid;
use self::tracks::{FrameCells, Geometry, Rect, Size, Wire, lay_out};
use super::RenderOptions;
use super::diagram::{Diagram, Endpoint};
use crate::graph::CompiledGraph;

/// What one item looks like, for sizing and drawing.
#[derive(Debug, Clone)]
enum Look {
    Box {
        lines: Vec<String>,
        stacked: bool,
        dashed: bool,
    },
    Marker {
        ratio: String,
    },
}

/// One box or marker, with the region it belongs to.
#[derive(Debug, Clone)]
struct Item {
    endpoint: Endpoint,
    region: usize,
    look: Look,
}

/// Everything placed and routed, ready to draw.
struct Arrangement {
    items: Vec<Item>,
    wires: Vec<Wire>,
    geometry: Geometry,
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
    draw_diagram(&diagram, options)
}

/// Draws a diagram on one grid, so every column shares a width and every row a port line.
///
/// Frame text and labels go down after every line, so no line cuts through them.
pub(crate) fn draw_diagram(diagram: &Diagram, options: &RenderOptions) -> String {
    if diagram.boxes.is_empty() {
        return String::new();
    }

    let arranged = arrange(diagram, options);
    let geometry = &arranged.geometry;
    let mut grid = Grid::new(geometry.width, geometry.height);

    for rect in &geometry.frames {
        draw_frame(&mut grid, *rect);
    }

    let fronts = box_fronts(&arranged);

    for (item, front) in arranged.items.iter().zip(&fronts) {
        match &item.look {
            Look::Box {
                lines,
                stacked,
                dashed,
            } => draw_box(&mut grid, *front, lines, *dashed, *stacked),
            Look::Marker { ratio } => draw_marker(&mut grid, *front, ratio),
        }
    }

    for (wire, points) in arranged.wires.iter().zip(&geometry.routes) {
        let source = &arranged.items[wire.from];
        let front = fronts[wire.from];
        let cell = geometry.items[wire.from];
        let boxed = matches!(source.look, Look::Box { .. });
        let narrowed = front.width < cell.width;

        draw_route(&mut grid, points, boxed && !narrowed);

        if boxed && narrowed {
            let (_, exit_y) = points[0];
            grid.draw(front.right(), exit_y, '├');
        }
    }

    for (block, rect) in diagram.blocks.iter().zip(&geometry.frames) {
        let heading = block.short_heading();
        let detail = block.detail();
        draw_frame_text(&mut grid, *rect, &heading, detail.as_deref());
    }

    for (link, spot) in diagram.links.iter().zip(&geometry.labels) {
        let param = link.param.filter(|_| options.show_param_names);
        let (Some(param), Some((end, y))) = (param, spot) else {
            continue;
        };

        draw_label(&mut grid, *end, *y, param);
    }

    let mut rendered = grid.render();
    while rendered.ends_with("\n\n") {
        rendered.pop();
    }

    if diagram.planned {
        rendered.push('\n');
        rendered.push_str(LEGEND);
        rendered.push('\n');
    }

    rendered
}

/// The rectangle each item is drawn in.
///
/// A plain box sharing a column with a stacked box is drawn one cell narrower, so its right
/// border lines up with the stacked box's front rather than its shadow. Its line still leaves
/// from the column's last cell.
fn box_fronts(arranged: &Arrangement) -> Vec<Rect> {
    let stacked_columns: HashSet<usize> = arranged
        .items
        .iter()
        .zip(&arranged.geometry.items)
        .filter(|(item, _)| matches!(item.look, Look::Box { stacked: true, .. }))
        .map(|(_, rect)| rect.x)
        .collect();

    arranged
        .items
        .iter()
        .zip(&arranged.geometry.items)
        .map(|(item, rect)| {
            let plain = matches!(item.look, Look::Box { stacked: false, .. });

            match plain && stacked_columns.contains(&rect.x) {
                true => Rect {
                    width: rect.width - 1,
                    ..*rect
                },
                false => *rect,
            }
        })
        .collect()
}

/// Places and routes every item, with wires in the same order as the diagram's links.
fn arrange(diagram: &Diagram, options: &RenderOptions) -> Arrangement {
    let items = items_of(diagram, options);
    let index: HashMap<Endpoint, usize> = items
        .iter()
        .enumerate()
        .map(|(position, item)| (item.endpoint, position))
        .collect();
    let edges: Vec<(usize, usize)> = diagram
        .links
        .iter()
        .map(|link| (index[&link.from], index[&link.to]))
        .collect();
    let regions = regions_of(&items, diagram.blocks.len());

    let shape = Shape {
        count: items.len(),
        edges: &edges,
        regions: &regions,
    };
    let cells = place_cells(&shape);

    let entries = entry_indices(&edges, &cells);
    let arrivals = arrival_counts(&edges, items.len());
    let sizes: Vec<Size> = items
        .iter()
        .zip(&arrivals)
        .map(|(item, count)| size_of(item, *count))
        .collect();

    let mut wires = Vec::with_capacity(edges.len());
    for ((edge, entry), link) in edges.iter().zip(&entries).zip(&diagram.links) {
        let label = link
            .param
            .filter(|_| options.show_param_names)
            .map_or(0, |param| param.chars().count());

        let shared_region = items[edge.0].region == items[edge.1].region;
        let region = match diagram.planned && shared_region {
            true => Some(items[edge.0].region),
            false => None,
        };

        wires.push(Wire {
            from: edge.0,
            to: edge.1,
            entry: *entry,
            label,
            region,
        });
    }

    let frames = match diagram.planned {
        true => frame_cells(diagram, &regions, &cells),
        false => Vec::new(),
    };
    let geometry = lay_out(&cells, &sizes, &wires, &frames);

    Arrangement {
        items,
        wires,
        geometry,
    }
}

/// Every block's items in order, its disperse first and its aggregate last.
fn items_of(diagram: &Diagram, options: &RenderOptions) -> Vec<Item> {
    let mut items = Vec::new();

    for block in &diagram.blocks {
        let disperse = diagram.exchange(Endpoint::Disperse(block.id));
        if let Some(exchange) = disperse {
            items.push(Item {
                endpoint: exchange.endpoint(),
                region: block.id,
                look: Look::Marker {
                    ratio: exchange.ratio(),
                },
            });
        }

        for node in &block.boxes {
            let entry = &diagram.boxes[node.index()];

            items.push(Item {
                endpoint: Endpoint::Box(*node),
                region: block.id,
                look: Look::Box {
                    lines: entry.lines(options),
                    stacked: entry.width > 1,
                    dashed: entry.crossing.is_some() || entry.provider.is_some(),
                },
            });
        }

        let aggregate = diagram.exchange(Endpoint::Aggregate(block.id));
        if let Some(exchange) = aggregate {
            items.push(Item {
                endpoint: exchange.endpoint(),
                region: block.id,
                look: Look::Marker {
                    ratio: exchange.ratio(),
                },
            });
        }
    }

    items
}

fn regions_of(items: &[Item], count: usize) -> Vec<Vec<usize>> {
    let mut regions = vec![Vec::new(); count];

    for (position, item) in items.iter().enumerate() {
        regions[item.region].push(position);
    }

    regions
}

/// Each link's entry index at its target, ordered by the row it arrives from, top first.
fn entry_indices(edges: &[(usize, usize)], cells: &Cells) -> Vec<usize> {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by_key(|index| {
        let (from, to) = edges[*index];
        let (arrival, track) = arrival_of(cells.cells[from], cells.cells[to]);

        (to, arrival, track, *index)
    });

    let mut entries = vec![0; edges.len()];
    let mut previous = None;
    let mut next = 0;

    for index in order {
        let target = edges[index].1;

        if previous != Some(target) {
            previous = Some(target);
            next = 0;
        }

        entries[index] = next;
        next += 1;
    }

    entries
}

/// Where a link arrives from, counting rows and channels together, with a tie break for links
/// sharing a channel.
///
/// A link to the next column arrives from its source's row. Any other link arrives from the
/// channel it runs along, just above or below its source's row. Links sharing a channel take
/// entries in the order of their tracks, top first.
fn arrival_of(source: Cell, target: Cell) -> (usize, isize) {
    let row_position = source.row * 2 + 1;
    let column = source.column as isize;

    if target.column == source.column + 1 {
        return (row_position, column);
    }

    match target.row < source.row {
        true => (row_position - 1, column),
        false => (row_position + 1, -column),
    }
}

fn arrival_counts(edges: &[(usize, usize)], count: usize) -> Vec<usize> {
    let mut arrivals = vec![0; count];

    for (_, to) in edges {
        arrivals[*to] += 1;
    }

    arrivals
}

fn size_of(item: &Item, arrivals: usize) -> Size {
    match &item.look {
        Look::Box { lines, stacked, .. } => {
            let (width, height) = box_size(lines, *stacked, arrivals);
            let shadow = usize::from(*stacked);

            Size {
                width,
                height,
                port: (height - shadow) / 2,
                stacked: *stacked,
            }
        },
        Look::Marker { ratio } => {
            let (width, height) = marker_size(ratio, arrivals);

            Size {
                width,
                height,
                port: 0,
                stacked: false,
            }
        },
    }
}

fn frame_cells(diagram: &Diagram, regions: &[Vec<usize>], cells: &Cells) -> Vec<FrameCells> {
    diagram
        .blocks
        .iter()
        .zip(regions)
        .map(|(block, items)| {
            let placed: Vec<Cell> = items.iter().map(|item| cells.cells[*item]).collect();
            let first_column = placed.iter().map(|cell| cell.column).min().unwrap_or(0);
            let last_column = placed.iter().map(|cell| cell.column).max().unwrap_or(0);
            let top_row = placed.iter().map(|cell| cell.row).min().unwrap_or(0);
            let bottom_row = placed.iter().map(|cell| cell.row).max().unwrap_or(0);
            let detail = block.detail();

            FrameCells {
                first_column,
                last_column,
                top_row,
                bottom_row,
                detail: detail.is_some(),
            }
        })
        .collect()
}
