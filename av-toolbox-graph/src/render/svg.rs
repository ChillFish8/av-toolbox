use std::collections::HashMap;

use av_toolbox_core::StageKind;

use super::{Metrics, Placement, Placements, RenderOptions, chain_of, labels_for, place_slots};
use crate::graph::{CompiledGraph, NodeId};
use crate::layout::{Layout, Slot};

const CHARACTER_WIDTH: usize = 8;
const LINE_HEIGHT: usize = 20;
const BOX_PADDING: usize = 24;
const MARGIN: usize = 16;

const METRICS: Metrics = Metrics {
    column_gap: 32,
    rank_gap: 56,
    virtual_width: 8,
};

const STYLE: &str = "\
.node { stroke-width: 1.5; }
.node.source { fill: #dbeafe; stroke: #1e40af; }
.node.filter { fill: #dcfce7; stroke: #166534; }
.node.mask { fill: #fef3c7; stroke: #92400e; }
.node.transform { fill: #ede9fe; stroke: #5b21b6; }
.node.sink { fill: #fee2e2; stroke: #991b1b; }
.label { font: 13px monospace; fill: #111827; text-anchor: middle; }
.edge { fill: none; stroke: #4b5563; stroke-width: 1.5; }
.edge-label { font: 11px monospace; fill: #4b5563; text-anchor: middle; }";

/// Draws the graph as an SVG document that needs nothing else to display.
pub fn render_svg(graph: &CompiledGraph, options: &RenderOptions) -> String {
    let layout = Layout::build(graph);
    let labels: Vec<Vec<String>> = graph
        .nodes()
        .iter()
        .map(|node| labels_for(node, options))
        .collect();

    let sizes = sizes_of(graph, &labels);
    let placements = place_slots(&layout, &sizes, &METRICS);

    let width = placements.width + MARGIN * 2;
    let height = placements.height + MARGIN * 2;

    let mut out = String::new();
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"#
    ));
    out.push_str(&format!("\n<style>\n{STYLE}\n</style>\n"));
    out.push_str(ARROW_MARKER);

    for node in graph.nodes() {
        let rank = layout.rank_of_stage(node.id());
        let placement = placements.get(rank, Slot::Stage(node.id()));

        out.push_str(&draw_box(placement, node.kind(), &labels[node.id().index()]));
    }

    for node in graph.nodes() {
        for index in 0..node.inputs().len() {
            let chain = chain_of(&layout, graph, node.id(), index);
            out.push_str(&draw_edge(&placements, &chain));
        }
    }

    out.push_str("</svg>\n");
    out
}

const ARROW_MARKER: &str = "\
<defs>
<marker id=\"arrow\" viewBox=\"0 0 8 8\" refX=\"6\" refY=\"4\" markerWidth=\"6\" markerHeight=\"6\" orient=\"auto\">
<path d=\"M0,0 L8,4 L0,8 z\" fill=\"#4b5563\"/>
</marker>
</defs>
";

fn sizes_of(graph: &CompiledGraph, labels: &[Vec<String>]) -> HashMap<NodeId, (usize, usize)> {
    graph
        .nodes()
        .iter()
        .map(|node| {
            let lines = &labels[node.id().index()];
            let widest = lines.iter().map(|line| line.chars().count()).max().unwrap_or(0);
            let width = widest * CHARACTER_WIDTH + BOX_PADDING;

            (node.id(), (width, lines.len() * LINE_HEIGHT + BOX_PADDING))
        })
        .collect()
}

fn draw_box(placement: Placement, kind: StageKind, labels: &[String]) -> String {
    let class = class_of(kind);
    let x = placement.x + MARGIN;
    let y = placement.top + MARGIN;

    let mut out = format!(
        "<rect class=\"node {class}\" x=\"{x}\" y=\"{y}\" width=\"{}\" height=\"{}\" rx=\"4\"/>\n",
        placement.width, placement.height
    );

    let centre = x + placement.width / 2;
    for (offset, label) in labels.iter().enumerate() {
        let baseline = y + BOX_PADDING / 2 + LINE_HEIGHT * offset + LINE_HEIGHT / 2;
        out.push_str(&format!(
            "<text class=\"label\" x=\"{centre}\" y=\"{baseline}\">{}</text>\n",
            escape(label)
        ));
    }

    out
}

/// Draws one edge as a single polyline bending through each slot it passes.
fn draw_edge(placements: &Placements, chain: &[(usize, Slot)]) -> String {
    let mut points: Vec<(usize, usize)> = Vec::new();

    for pair in chain.windows(2) {
        let upper = placements.get(pair[0].0, pair[0].1);
        let lower = placements.get(pair[1].0, pair[1].1);

        let from_x = upper.centre() + MARGIN;
        let to_x = lower.centre() + MARGIN;
        let start = upper.bottom() + MARGIN;
        let end = lower.top + MARGIN;
        let middle = start + (end - start) / 2;

        points.push((from_x, start));
        points.push((from_x, middle));
        points.push((to_x, middle));
        points.push((to_x, end));
    }

    points.dedup();

    let rendered: Vec<String> = points.iter().map(|(x, y)| format!("{x},{y}")).collect();

    format!(
        "<polyline class=\"edge\" points=\"{}\" marker-end=\"url(#arrow)\"/>\n",
        rendered.join(" ")
    )
}

fn class_of(kind: StageKind) -> &'static str {
    match kind {
        StageKind::Source => "source",
        StageKind::Filter => "filter",
        StageKind::Mask => "mask",
        StageKind::Transform => "transform",
        StageKind::Sink => "sink",
    }
}

fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::{Spanned, StageSpec, build, testing};

    fn pipeline() -> Vec<StageSpec> {
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

    fn render() -> String {
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();

        render_svg(&graph, &RenderOptions::default())
    }

    #[test]
    fn the_document_is_self_contained() {
        let rendered = render();

        assert!(rendered.starts_with(r#"<svg xmlns="http://www.w3.org/2000/svg""#));
        assert!(rendered.contains("<style>"));
        assert!(rendered.trim_end().ends_with("</svg>"));
        assert!(!rendered.contains("<link"));
    }

    #[test]
    fn every_stage_gets_a_rect_carrying_its_kind() {
        let rendered = render();

        assert_eq!(rendered.matches("<rect").count(), 4);
        assert!(rendered.contains(r#"class="node source""#));
        assert!(rendered.contains(r#"class="node mask""#));
    }

    #[test]
    fn every_edge_gets_one_polyline() {
        let rendered = render();

        assert_eq!(rendered.matches("<polyline").count(), 4);
    }

    #[test]
    fn the_view_box_covers_the_whole_drawing() {
        let rendered = render();
        let start = rendered.find(r#"viewBox="0 0 "#).expect("should have a view box");
        let rest = &rendered[start + r#"viewBox="0 0 "#.len()..];
        let inside = &rest[..rest.find('"').expect("the view box closes")];

        let bounds: Vec<u32> = inside
            .split(' ')
            .map(|value| value.parse().expect("the bounds are numbers"))
            .collect();

        assert_eq!(bounds.len(), 2);
        assert!(bounds[0] > 0 && bounds[1] > 0);
    }

    #[test]
    fn a_label_is_escaped_rather_than_breaking_the_document() {
        let registry = testing::registry();
        let mut specs = pipeline();
        specs[1].name = Spanned::unknown(String::from("a<b&c"));
        specs[2].inputs = vec![testing::positional("a<b&c"), testing::positional("input")];

        let graph = build(&registry, &specs).expect("should build").into_graph();
        let rendered = render_svg(&graph, &RenderOptions::default());

        assert!(rendered.contains("a&lt;b&amp;c"));
        assert!(!rendered.contains("a<b&c"));
    }
}
