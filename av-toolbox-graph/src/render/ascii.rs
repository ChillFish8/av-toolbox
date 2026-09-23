use std::collections::HashMap;

use super::execution::render_execution;
use super::grid::Grid;
use super::{Metrics, Placement, Placements, RenderOptions, RenderView, chain_of, labels_for, place_slots};
use crate::graph::{CompiledGraph, NodeId, NodeRole};
use crate::layout::{Layout, Slot};

/// Two spaces of padding inside a box, and a border either side of that.
const BOX_PADDING: usize = 4;

const METRICS: Metrics = Metrics {
    column_gap: 3,
    rank_gap: 2,
    virtual_width: 1,
};

/// Draws the graph as boxes joined by lines.
pub fn render_ascii(graph: &CompiledGraph, options: &RenderOptions) -> String {
    if options.view == RenderView::Execution {
        return render_execution(graph, options);
    }

    if graph.is_empty() {
        return String::new();
    }

    let layout = Layout::build(graph);
    let labels: Vec<Vec<String>> = graph
        .nodes()
        .iter()
        .map(|node| {
            let plan = graph.execution_plan().map(|plan| plan.node(node.id()));
            labels_for(node, plan, options)
        })
        .collect();

    let sizes = sizes_of(graph, &labels);
    let placements = place_slots(&layout, &sizes, &METRICS);

    let mut grid = Grid::new(placements.width, placements.height);

    draw_pass_throughs(&mut grid, &layout, &placements);
    draw_boxes(&mut grid, graph, &layout, &placements, &labels);
    draw_edges(&mut grid, graph, &layout, &placements);

    grid.render()
}

fn sizes_of(graph: &CompiledGraph, labels: &[Vec<String>]) -> HashMap<NodeId, (usize, usize)> {
    graph
        .nodes()
        .iter()
        .map(|node| {
            let lines = &labels[node.id().index()];
            let widest = lines.iter().map(|line| line.chars().count()).max().unwrap_or(0);

            (node.id(), (widest + BOX_PADDING, lines.len() + 2))
        })
        .collect()
}

/// Runs a line down every virtual slot, so an edge crossing a rank has something to follow.
fn draw_pass_throughs(grid: &mut Grid, layout: &Layout, placements: &Placements) {
    for (rank, slots) in layout.ranks().iter().enumerate() {
        for slot in slots {
            let Slot::Virtual(_) = slot else {
                continue;
            };

            let placement = placements.get(rank, *slot);
            for y in placement.top..=placement.bottom() {
                grid.draw(placement.centre(), y, '│');
            }
        }
    }
}

fn draw_boxes(
    grid: &mut Grid,
    graph: &CompiledGraph,
    layout: &Layout,
    placements: &Placements,
    labels: &[Vec<String>],
) {
    for node in graph.nodes() {
        let rank = layout.rank_of_stage(node.id());
        let placement = placements.get(rank, Slot::Stage(node.id()));
        let dashed = node.role() != NodeRole::Stage;

        draw_box(grid, placement, &labels[node.id().index()], dashed);
    }
}

/// Draws one node's box, dashed when it is a crossing rather than a stage the pipeline declared.
fn draw_box(grid: &mut Grid, placement: Placement, labels: &[String], dashed: bool) {
    let (horizontal, vertical) = if dashed { ('┄', '┆') } else { ('─', '│') };
    let right = placement.x + placement.width - 1;
    let bottom = placement.bottom();

    for x in (placement.x + 1)..right {
        grid.draw(x, placement.top, horizontal);
        grid.draw(x, bottom, horizontal);
    }

    for y in (placement.top + 1)..bottom {
        grid.draw(placement.x, y, vertical);
        grid.draw(right, y, vertical);
    }

    grid.set(placement.x, placement.top, '┌');
    grid.set(right, placement.top, '┐');
    grid.set(placement.x, bottom, '└');
    grid.set(right, bottom, '┘');

    for (offset, label) in labels.iter().enumerate() {
        let inner = placement.width - 2;
        let padding = (inner - label.chars().count()) / 2;

        grid.text(placement.x + 1 + padding, placement.top + 1 + offset, label);
    }
}

fn draw_edges(grid: &mut Grid, graph: &CompiledGraph, layout: &Layout, placements: &Placements) {
    for node in graph.nodes() {
        for index in 0..node.inputs().len() {
            let chain = chain_of(layout, graph, node.id(), index);

            for pair in chain.windows(2) {
                let upper = placements.get(pair[0].0, pair[0].1);
                let lower = placements.get(pair[1].0, pair[1].1);
                let out_of_a_stage = matches!(pair[0].1, Slot::Stage(_));
                let into_a_stage = matches!(pair[1].1, Slot::Stage(_));

                draw_segment(grid, upper, lower, out_of_a_stage, into_a_stage);
            }
        }
    }
}

/// Runs a line out of the bottom of one slot and into the top of the next.
///
/// The line drops one row, travels sideways, then drops the rest of the way, which keeps every
/// run either vertical or horizontal.
///
/// Leaving a stage's box means merging onto its bottom border, which already has both
/// horizontal sides, so the exit point is drawn as a tee rather than a plain vertical bar.
/// Leaving a virtual slot means continuing an existing pass-through line, which is already
/// vertical, so a plain bar is what keeps that lane unbroken.
fn draw_segment(
    grid: &mut Grid,
    upper: Placement,
    lower: Placement,
    out_of_a_stage: bool,
    into_a_stage: bool,
) {
    let from_x = upper.centre();
    let to_x = lower.centre();
    let start = upper.bottom();
    let end = lower.top;
    let middle = start + 1;

    let exit = if out_of_a_stage { '┬' } else { '│' };
    grid.draw(from_x, start, exit);

    if from_x == to_x {
        for y in middle..end {
            grid.draw(from_x, y, '│');
        }
    } else {
        let turning_right = to_x > from_x;

        grid.draw(from_x, middle, if turning_right { '└' } else { '┘' });
        grid.draw(to_x, middle, if turning_right { '┐' } else { '┌' });

        for x in (from_x.min(to_x) + 1)..from_x.max(to_x) {
            grid.draw(x, middle, '─');
        }

        for y in (middle + 1)..end {
            grid.draw(to_x, y, '│');
        }
    }

    if into_a_stage {
        grid.set(to_x, end, '▼');
        return;
    }

    grid.draw(to_x, end, '│');
}

#[cfg(test)]
pub(crate) mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::{Spanned, StageSpec, build, testing};

    fn render(specs: &[StageSpec]) -> String {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();
        let options = RenderOptions {
            show_identifier: false,
            show_param_names: false,
            ..RenderOptions::default()
        };

        render_ascii(&graph, &options)
    }

    fn source() -> StageSpec {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));
        source
    }

    /// A host source, a device filter and a host sink, placed all the way through, so an upload
    /// and an unload both appear.
    pub(crate) fn placed_round_trip() -> CompiledGraph {
        let registry = testing::registry();
        let mut source = testing::spec(StageKind::Source, "input", "test/host-file");
        source.args = Spanned::unknown(json!({}));
        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));
        let mut sink = testing::spec(StageKind::Sink, "out", "test/host-stdout");
        sink.inputs = vec![testing::positional("blur")];

        let mut graph = build(&registry, &[source, blur, sink])
            .expect("should build")
            .into_graph();
        let capabilities = crate::residency::tests::capabilities();
        let mut formats = crate::solve(&graph, &registry, &capabilities).expect("should solve");
        crate::place_transfers(&mut graph, &registry, &mut formats).expect("should place");

        graph
    }

    #[test]
    fn a_placed_graph_names_each_nodes_side() {
        let rendered = render_ascii(&placed_round_trip(), &RenderOptions::default());

        assert!(rendered.contains("host"));
        assert!(rendered.contains("gpu"));
    }

    #[test]
    fn a_crossing_is_drawn_dashed() {
        let rendered = render_ascii(&placed_round_trip(), &RenderOptions::default());

        assert!(rendered.contains("upload(input)"));
        assert!(rendered.contains('┆'));
    }

    #[test]
    fn an_unplaced_graph_draws_no_side_and_no_dashes() {
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let registry = testing::registry();
        let graph = build(&registry, &[source(), sink])
            .expect("should build")
            .into_graph();

        let rendered = render_ascii(&graph, &RenderOptions::default());

        assert!(!rendered.contains('┆'));
        assert!(
            !rendered
                .lines()
                .any(|line| line.trim_matches(|c| c == '│' || c == ' ') == "gpu")
        );
    }

    #[test]
    fn a_source_feeding_a_sink_draws_two_boxes_joined_by_a_line() {
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let rendered = render(&[source(), sink]);

        assert_eq!(
            rendered,
            concat!(
                "┌───────┐\n",
                "│ input │\n",
                "└───┬───┘\n",
                "    │\n",
                "    │\n",
                " ┌──▼──┐\n",
                " │ out │\n",
                " └─────┘\n",
            )
        );
    }

    #[test]
    fn an_edge_skipping_a_rank_runs_alongside_the_boxes_in_it() {
        let mut blur = testing::spec(StageKind::Filter, "blur", "test/blur");
        blur.inputs = vec![testing::positional("input")];
        blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("blur"), testing::positional("input")];

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("merge")];

        let rendered = render(&[source(), blur, merge, sink]);

        let passes_alongside = rendered
            .lines()
            .any(|line| line.contains("blur") && line.matches('│').count() > 2);

        assert!(
            passes_alongside,
            "the skipping edge should run beside the blur box"
        );
    }

    #[test]
    fn a_rendered_graph_is_never_empty() {
        let registry = testing::registry();
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let graph = build(&registry, &[source(), sink])
            .expect("should build")
            .into_graph();

        assert!(!render_ascii(&graph, &RenderOptions::default()).is_empty());
    }

    #[test]
    fn the_identifier_is_printed_under_the_name_when_asked_for() {
        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let registry = testing::registry();
        let graph = build(&registry, &[source(), sink])
            .expect("should build")
            .into_graph();

        let rendered = render_ascii(&graph, &RenderOptions::default());

        assert!(rendered.contains("test/file"));
    }
}
