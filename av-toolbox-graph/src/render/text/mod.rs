mod cells;
mod draw;
mod frame;
mod grid;
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
                    dashed: entry.crossing.is_some(),
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

#[cfg(test)]
mod tests {
    use av_toolbox_core::{Granularity, StageKind};

    use super::draw::route_cells;
    use super::frame::frame_text;
    use super::*;
    use crate::graph::NodeId;
    use crate::render::diagram::{Block, DiagramBox, Link};
    use crate::render::fixtures;

    fn bare() -> RenderOptions {
        RenderOptions {
            show_identifier: false,
            show_param_names: false,
        }
    }

    fn labelled() -> RenderOptions {
        RenderOptions {
            show_identifier: false,
            show_param_names: true,
        }
    }

    /// A diagram of one box, framed as a one wide stream region when `planned`.
    fn lone_box(name: &str, planned: bool) -> Diagram {
        let granularity = planned.then_some(Granularity::Stream);
        let block = Block {
            id: 0,
            boxes: vec![NodeId::new(0)],
            width: 1,
            granularity,
            routing: None,
            reason: None,
        };
        let only = DiagramBox {
            node: NodeId::new(0),
            block: 0,
            name: name.to_string(),
            identifier: String::from("test/file"),
            kind: StageKind::Source,
            residency: None,
            crossing: None,
            width: 1,
            reason: None,
        };

        Diagram {
            blocks: vec![block],
            boxes: vec![only],
            exchanges: Vec::new(),
            links: Vec::new(),
            planned,
        }
    }

    fn property_diagrams() -> Vec<Diagram> {
        let pipelines = [
            fixtures::two_exit_pipeline(),
            fixtures::mixed_inputs_pipeline(),
            fixtures::middle_exit_pipeline(),
            fixtures::many_region_pipeline(),
            fixtures::wide_pipeline(4),
        ];

        let mut diagrams: Vec<Diagram> = pipelines
            .iter()
            .map(|specs| {
                let graph = fixtures::planned(specs);
                Diagram::of(&graph)
            })
            .collect();

        let unplanned = [
            fixtures::skipping_pipeline(),
            fixtures::named_inputs_pipeline(),
            fixtures::crossed_pipeline(),
        ];

        for specs in &unplanned {
            let graph = fixtures::built(specs);
            let diagram = Diagram::of(&graph);
            diagrams.push(diagram);
        }

        let example = fixtures::example_diagram();
        diagrams.push(example);
        diagrams
    }

    /// A diagram built by hand, with boxes numbered across the regions in order.
    ///
    /// Each region is its box names and how many workers run it. Links name the boxes they join.
    fn hand_built(regions: &[(&[&str], usize)], links: &[(usize, usize)], planned: bool) -> Diagram {
        let mut blocks = Vec::new();
        let mut boxes = Vec::new();

        for (id, (names, width)) in regions.iter().enumerate() {
            let first = boxes.len();
            let granularity = planned.then_some(Granularity::Stream);

            for (offset, name) in names.iter().enumerate() {
                boxes.push(DiagramBox {
                    node: NodeId::new(first + offset),
                    block: id,
                    name: name.to_string(),
                    identifier: String::from("test/file"),
                    kind: StageKind::Filter,
                    residency: None,
                    crossing: None,
                    width: *width,
                    reason: None,
                });
            }

            let members = (first..boxes.len()).map(NodeId::new).collect();
            blocks.push(Block {
                id,
                boxes: members,
                width: *width,
                granularity,
                routing: None,
                reason: None,
            });
        }

        let links = links
            .iter()
            .map(|(from, to)| Link {
                from: Endpoint::Box(NodeId::new(*from)),
                to: Endpoint::Box(NodeId::new(*to)),
                param: None,
                bounded: None,
            })
            .collect();

        Diagram {
            blocks,
            boxes,
            exchanges: Vec::new(),
            links,
            planned,
        }
    }

    /// Every frame heading or detail that a line has written over, as the cell it starts in.
    fn covered_frame_text(diagram: &Diagram) -> Vec<(usize, usize)> {
        let arranged = arrange(diagram, &labelled());
        let rendered = draw_diagram(diagram, &labelled());
        let rows: Vec<Vec<char>> = rendered.lines().map(|line| line.chars().collect()).collect();
        let mut covered = Vec::new();

        for (block, rect) in diagram.blocks.iter().zip(&arranged.geometry.frames) {
            let heading = block.short_heading();
            let detail = block.detail();

            for piece in frame_text(*rect, &heading, detail.as_deref()) {
                let row = &rows[piece.y];
                let shown: String = row
                    .iter()
                    .skip(piece.x)
                    .take(piece.text.chars().count())
                    .collect();

                if shown.trim_end() != piece.text.trim_end() {
                    covered.push((piece.x, piece.y));
                }
            }
        }

        covered
    }

    fn overlaps(first: Rect, second: Rect) -> bool {
        let across = first.x <= second.right() && second.x <= first.right();
        let down = first.y <= second.bottom() && second.y <= first.bottom();

        across && down
    }

    fn strictly_inside(inner: Rect, outer: Rect) -> bool {
        let across = inner.x > outer.x && inner.right() < outer.right();
        let down = inner.y > outer.y && inner.bottom() < outer.bottom();

        across && down
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

        let shown = render_ascii(&graph, &labelled());
        let hidden = render_ascii(&graph, &bare());

        assert!(shown.contains("reference"));
        assert!(!hidden.contains("reference"));
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
    fn an_unplanned_graph_draws_no_frames_and_no_legend() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert!(!rendered.contains("region"));
        assert!(!rendered.contains("◁ disperse"));
    }

    #[test]
    fn an_empty_graph_renders_nothing() {
        let graph = CompiledGraph::new(Vec::new());
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

    #[test]
    fn a_planned_diagram_ends_with_the_legend() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(rendered.lines().last(), Some(LEGEND));
    }

    #[test]
    fn a_long_heading_is_cut_rather_than_widening_its_frame() {
        let diagram = lone_box("a", true);
        let heading = diagram.blocks[0].short_heading();
        let arranged = arrange(&diagram, &bare());
        let frame = arranged.geometry.frames[0];
        let only = arranged.geometry.items[0];

        assert!(heading.chars().count() > only.width);

        let rendered = draw_diagram(&diagram, &bare());
        let top_border = rendered.lines().next().expect("the frame has a top border");

        assert!(top_border.contains('…'));
        assert!(frame.width <= only.width + 4);
    }

    #[test]
    fn a_lone_source_draws_one_box() {
        let diagram = lone_box("input", false);
        let rendered = draw_diagram(&diagram, &bare());

        assert_eq!(rendered, concat!("┌───────┐\n", "│ input │\n", "└───────┘\n"));
    }

    #[test]
    fn frames_never_overlap() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());
            let frames = &arranged.geometry.frames;

            for (index, first) in frames.iter().enumerate() {
                for second in frames.iter().skip(index + 1) {
                    assert!(!overlaps(*first, *second), "{first:?} overlaps {second:?}");
                }
            }
        }
    }

    #[test]
    fn every_item_sits_inside_its_frame() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());

            if !diagram.planned {
                continue;
            }

            for (item, rect) in arranged.items.iter().zip(&arranged.geometry.items) {
                let frame = arranged.geometry.frames[item.region];
                assert!(strictly_inside(*rect, frame), "{rect:?} is not inside {frame:?}");
            }
        }
    }

    #[test]
    fn lines_from_different_exits_never_run_along_each_other() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());
            let mut owners: HashMap<(usize, usize, bool), usize> = HashMap::new();

            for (wire, points) in arranged.wires.iter().zip(&arranged.geometry.routes) {
                let passed = route_cells(points);
                let interior = passed.iter().skip(1).take(passed.len().saturating_sub(2));

                for (x, y, sides) in interior {
                    let orientations = [(sides.left || sides.right, false), (sides.up || sides.down, true)];

                    for (present, vertical) in orientations {
                        if !present {
                            continue;
                        }

                        let owner = owners.entry((*x, *y, vertical)).or_insert(wire.from);
                        assert_eq!(*owner, wire.from, "two exits share the run through ({x}, {y})");
                    }
                }
            }
        }
    }

    #[test]
    fn every_route_runs_from_its_sources_right_edge_to_inside_its_targets_left_edge() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());
            let items = &arranged.geometry.items;

            for (wire, points) in arranged.wires.iter().zip(&arranged.geometry.routes) {
                let source = items[wire.from];
                let target = items[wire.to];
                let (start_x, start_y) = *points.first().expect("a route has a start");
                let (end_x, end_y) = *points.last().expect("a route has an end");

                assert_eq!(start_x, source.right());
                assert!(start_y >= source.y && start_y <= source.bottom());
                assert_eq!(end_x, target.x);
                assert!(end_y >= target.y && end_y <= target.bottom());
            }
        }
    }

    #[test]
    fn labels_overlap_neither_each_other_nor_an_item() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());
            let mut spans: Vec<Rect> = Vec::new();

            for (wire, spot) in arranged.wires.iter().zip(&arranged.geometry.labels) {
                let Some((end, y)) = spot else {
                    continue;
                };

                let span = Rect {
                    x: end - wire.label,
                    y: *y,
                    width: wire.label,
                    height: 1,
                };

                for item in &arranged.geometry.items {
                    assert!(!overlaps(span, *item), "a label covers {item:?}");
                }

                for other in &spans {
                    assert!(!overlaps(span, *other), "two labels overlap at {span:?}");
                }

                spans.push(span);
            }
        }
    }

    #[test]
    fn no_line_covers_a_frames_heading_or_detail() {
        for diagram in property_diagrams() {
            let covered = covered_frame_text(&diagram);
            assert!(covered.is_empty(), "frame text is covered at {covered:?}");
        }
    }

    #[test]
    fn a_line_turning_down_through_a_frames_top_leaves_its_heading_whole() {
        let regions: [(&[&str], usize); 2] = [(&["x", "y", "z"], 1), (&["p"], 1)];
        let diagram = hand_built(&regions, &[(0, 1), (0, 2), (3, 1)], true);
        let arranged = arrange(&diagram, &bare());
        let frame = arranged.geometry.frames[0];
        let crossing = arranged.geometry.routes[2]
            .iter()
            .any(|(x, y)| *y < frame.y && *x > frame.x && *x < frame.right());
        let covered = covered_frame_text(&diagram);

        assert!(crossing, "the line from p should reach region 0 from above");
        assert!(covered.is_empty(), "frame text is covered at {covered:?}");
    }

    #[test]
    fn no_line_passes_through_an_item() {
        for diagram in property_diagrams() {
            let arranged = arrange(&diagram, &labelled());

            for points in &arranged.geometry.routes {
                let passed = route_cells(points);
                let interior = passed.iter().skip(1).take(passed.len().saturating_sub(2));

                for (x, y, _) in interior {
                    for item in &arranged.geometry.items {
                        let inside =
                            *x >= item.x && *x <= item.right() && *y >= item.y && *y <= item.bottom();
                        assert!(!inside, "a line passes through {item:?} at ({x}, {y})");
                    }
                }
            }
        }
    }

    #[test]
    fn a_plain_box_lines_up_with_a_stacked_box_in_its_column() {
        let regions: [(&[&str], usize); 3] = [(&["plain"], 1), (&["wide"], 2), (&["out"], 1)];
        let diagram = hand_built(&regions, &[(0, 2), (1, 2)], false);
        let rendered = draw_diagram(&diagram, &bare());
        let names = ["plain", "wide", "out", "×2"];

        for line in rendered.lines() {
            for name in names {
                let Some(start) = line.find(name) else {
                    continue;
                };

                let before = line[..start].chars().last();
                let after = line[start + name.len()..].chars().next();
                assert_eq!(
                    (before, after),
                    (Some(' '), Some(' ')),
                    "{name} touches its border in {line:?}"
                );
            }
        }

        assert_eq!(
            rendered,
            concat!(
                "             ┌─────┐\n",
                "┌───────┐    │ out │\n",
                "│ plain ├────▶     │\n",
                "└───────┘  ┌─▶     │\n",
                "           │ └─────┘\n",
                "           │\n",
                "┌───────┐  │\n",
                "│ wide  │┐ │\n",
                "│  ×2   │├─┘\n",
                "└───────┘│\n",
                " └───────┘\n",
            )
        );
    }

    #[test]
    fn a_join_takes_the_spine_on_its_port_line_ahead_of_a_skipping_link() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let diagram = Diagram::of(&graph);
        let arranged = arrange(&diagram, &bare());
        let blur = fixtures::node_named(&graph, "blur");
        let merge = fixtures::node_named(&graph, "merge");

        let position_of = |node: NodeId| {
            let endpoint = Endpoint::Box(node);
            arranged.items.iter().position(|item| item.endpoint == endpoint)
        };
        let from = position_of(blur);
        let to = position_of(merge);
        let spine_link = arranged
            .wires
            .iter()
            .position(|wire| Some(wire.from) == from && Some(wire.to) == to)
            .expect("blur feeds merge");

        assert_eq!(arranged.geometry.routes[spine_link].len(), 2);
    }

    #[test]
    fn a_source_feeding_a_sink_draws_two_boxes_side_by_side() {
        let specs = fixtures::source_to_sink();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "┌───────┐  ┌─────┐\n",
                "│ input ├──▶ out │\n",
                "└───────┘  └─────┘\n",
            )
        );
    }

    #[test]
    fn an_edge_skipping_a_column_runs_in_the_channel_below() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "                       ┌───────┐\n",
                "┌───────┐   ┌──────┐   │ merge │  ┌─────┐\n",
                "│ input ├─┬─▶ blur ├───▶       ├──▶ out │\n",
                "└───────┘ │ └──────┘ ┌─▶       │  └─────┘\n",
                "          │          │ └───────┘\n",
                "          │          │\n",
                "          └──────────┘\n",
            )
        );
    }

    #[test]
    fn each_named_input_enters_on_its_own_row_under_its_own_label() {
        let specs = fixtures::named_inputs_pipeline();
        let graph = fixtures::built(&specs);
        let rendered = render_ascii(&graph, &labelled());

        assert_eq!(
            rendered,
            concat!(
                "                    ┌─────────┐\n",
                "┌────────┐          │ matched │       ┌─────┐\n",
                "│ first  ├──────base▶         ├──input▶ out │\n",
                "└────────┘ ┌─overlay▶         │       └─────┘\n",
                "           │        └─────────┘\n",
                "           │\n",
                "┌────────┐ │\n",
                "│ second ├─┘\n",
                "└────────┘\n",
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
                "┌ region … ─┐  ┌ region 1 · ×4 chunk ───────────────┐  ┌ regio… ─┐\n",
                "│           │  │ by chunk                           │  │ (granu… │\n",
                "│           │  │                                    │  │         │\n",
                "│           │  │            ┌─────────┐             │  │         │\n",
                "│ ┌───────┐ │  │            │ denoise │┐            │  │ ┌─────┐ │\n",
                "│ │ input ├─┼──┼─▶ ◁ 1:4 ───▶   ×4    │├──▶ ▷ 4:1 ──┼──┼─▶ out │ │\n",
                "│ └───────┘ │  │            └─────────┘│            │  │ └─────┘ │\n",
                "│           │  │             └─────────┘            │  │         │\n",
                "│           │  │                                    │  │         │\n",
                "└───────────┘  └────────────────────────────────────┘  └─────────┘\n",
                "\n",
                "◁ disperse, unbounded   ▷ aggregate, bounded   ┄ host and gpu crossing   ×N workers\n",
            )
        );
    }

    #[test]
    fn a_planned_mixed_inputs_graph_draws_exactly() {
        let specs = fixtures::mixed_inputs_pipeline();
        let graph = fixtures::planned(&specs);
        let rendered = render_ascii(&graph, &bare());

        assert_eq!(
            rendered,
            concat!(
                "┌ region … ─┐   ┌ region 1 · ×4 chunk ────────────────┐   ┌ region 3 · frame ──────────────────────┐\n",
                "│           │   │ by chunk                            │   │                                        │\n",
                "│           │   │                                     │   │                                        │\n",
                "│           │   │                ┌──────┐             │   │            ┌────────────────┐          │\n",
                "│ ┌───────┐ │   │                │ wide │┐            │   │            │     merge      │  ┌─────┐ │\n",
                "│ │ input ├─┼─┬─┼─▶ ◁ 1:4 ───────▶  ×4  │├──▶ ▷ 4:1 ──┼───┼─▶ ◁ 1:1 ───▶ (mixed inputs) ├──▶ out │ │\n",
                "│ └───────┘ │ │ │                └──────┘│            │ ┌─┼─▶          └────────────────┘  └─────┘ │\n",
                "│           │ │ │                 └──────┘            │ │ │                                        │\n",
                "│           │ │ │                                     │ │ │                                        │\n",
                "└───────────┘ │ └─────────────────────────────────────┘ │ └────────────────────────────────────────┘\n",
                "              │                                         │\n",
                "              │                ┌────────────────────────┘\n",
                "              │                │\n",
                "              │ ┌ region 2… ─┐ │\n",
                "              │ │            │ │\n",
                "              │ │ ┌────────┐ │ │\n",
                "              └─┼─▶ narrow ├─┼─┘\n",
                "                │ └────────┘ │\n",
                "                │            │\n",
                "                └────────────┘\n",
                "\n",
                "◁ disperse, unbounded   ▷ aggregate, bounded   ┄ host and gpu crossing   ×N workers\n",
            )
        );
    }

    #[test]
    fn the_example_pipeline_draws_exactly() {
        let diagram = fixtures::example_diagram();
        let rendered = draw_diagram(&diagram, &labelled());

        assert_eq!(
            rendered,
            concat!(
                "┌ region 0 · stream ──────────────────────────────┐      ┌ region 1 · ×8 chunk ───────────────────────────────────────────────────────────────────────────┐             ┌ region 2… ─┐      ┌ region 3 · ×6 chunk ───────────────────┐       ┌ region… ─┐\n",
                "│                                                 │      │ by chunk                                                                                       │             │ (granular… │      │ by chunk                               │       │ (granul… │\n",
                "│                                                 │      │                                                                                                │             │            │      │                                        │       │          │\n",
                "│                                                 │      │                                                     ┌────────┐                                 │             │            │      │                                        │       │          │\n",
                "│ ┌────────┐       ┌────────┐  ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐ │      │            ┌─────────┐        ┌────────┐            │ merge  │┐  ┌┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┐             │             │ ┌────────┐ │      │                ┌─────────┐             │       │ ┌──────┐ │\n",
                "│ │ source │       │ scenes │  ┆ upload(scenes) ┆ │      │            │ denoise │┐       │ deband │┐           │ gpu ×8 ││  ┆ unload(merge) ┆┐            │             │ │ chunks │ │      │                │ encode  │┐            │       │ │ mux  │ │\n",
                "│ │  host  ├─┬─clip▶  host  ├──▶      gpu       ├─┼──clip┼─▶ ◁ 1:8 ───▶ gpu ×8  │├─┬─clip▶ gpu ×8 │├───────base▶        │├──▶    host ×8    ┆├──▶ ▷ 8:1 ──┼──┬──────clip┼─▶  host  ├─┼──clip┼─▶ ◁ 1:6 ───────▶ host ×6 │├──▶ ▷ 6:1 ──┼──video┼─▶ host │ │\n",
                "│ └────────┘ │     └────────┘  ┆    host→gpu    ┆ │      │            └─────────┘│ │     └────────┘│ ┌──overlay▶        ││  ┆   gpu→host    ┆┆            │  │          │ └────────┘ │      │                └─────────┘│            │       │ └──────┘ │\n",
                "│            │                 └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘ │      │             └─────────┘ │      └────────┘ │┌────mask▶        ││  └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘┆            │  │          │            │      │                 └─────────┘            │       │          │\n",
                "│            │                                    │      │                         │                 ││        └────────┘│   └┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┘            │  │          │            │      │                                        │       │          │\n",
                "│            │                                    │      │                         │                 ││         └────────┘                                │  │          │            │      │                                        │       │          │\n",
                "│            │                                    │      │                         │                 ││                                                   │  │          │            │      │                                        │       │          │\n",
                "└────────────┼────────────────────────────────────┘      │                         │                 ││                                                   │  │          └────────────┘      └────────────────────────────────────────┘       └──────────┘\n",
                "             │                                           │                         │                 ││                                                   │  │\n",
                "             │                                           │                         ├─────────────────┘│                                                   │  │\n",
                "             └───────────────────────────────────────────┼─────────────────────────┼──────────────────┼───────────────────────────────────────────────────┼─┐│\n",
                "                                                         │                         │                  │                                                   │ ││\n",
                "                                                         │                         │                  │                                                   │ ││          ┌ region 5 · stream ──────────────┐\n",
                "                                                         │                         │                  │                                                   │ ││          │ (granularity)                   │\n",
                "                                                         │                         │                  │                                                   │ ││          │                                 │\n",
                "                                                         │                         │     ┌────────┐   │                                                   │ ││          │                     ┌─────────┐ │\n",
                "                                                         │                         │     │ edges  │┐  │                                                   │ ││          │                     │ quality │ │\n",
                "                                                         │                         └─clip▶ gpu ×8 │├──┘                                                   │ │└─distorted┼─▶ ◁ 1:1 ────────────▶  host   │ │\n",
                "                                                         │                               └────────┘│                                                      │ └──reference┼─▶                   └─────────┘ │\n",
                "                                                         │                                └────────┘                                                      │             │                                 │\n",
                "                                                         │                                                                                                │             │                                 │\n",
                "                                                         └────────────────────────────────────────────────────────────────────────────────────────────────┘             └─────────────────────────────────┘\n",
                "\n",
                "◁ disperse, unbounded   ▷ aggregate, bounded   ┄ host and gpu crossing   ×N workers\n",
            )
        );
    }
}
