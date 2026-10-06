use av_toolbox_core::StageKind;
use serde_json::{Value, json};

use super::RenderOptions;
use super::diagram::{
    Block,
    Diagram,
    DiagramBox,
    Endpoint,
    Exchange,
    ExchangeKind,
    Link,
    ShownDemand,
    routing_text,
};
use crate::ProviderPlacement;
use crate::graph::CompiledGraph;

const PAGE: &str = include_str!("page.html");
const SCRIPT: &str = include_str!("draw.js");
const STYLE: &str = include_str!("style.css");

/// Draws the graph as a standalone HTML page, laid out in the browser by libraries from a CDN.
pub fn render_html(graph: &CompiledGraph, options: &RenderOptions) -> String {
    let diagram = Diagram::of(graph);
    let data = diagram_json(&diagram, options);
    let serialised = data.to_string();
    let escaped = escape_for_script(&serialised);

    // The diagram goes in last, so no stage name can be mistaken for a placeholder.
    PAGE.replace("{{style}}", STYLE)
        .replace("{{script}}", SCRIPT)
        .replace("{{diagram}}", &escaped)
}

/// Stops anything in the JSON from closing the script tag it sits in.
///
/// A `<script>` element's content is never scanned for character references, so an HTML entity
/// would reach `JSON.parse` unchanged and break it. A JSON unicode escape survives instead. The
/// browser's tag scanner never sees a literal `<`, and `JSON.parse` decodes the escape back.
fn escape_for_script(json: &str) -> String {
    json.replace('&', "\\u0026")
        .replace('<', "\\u003C")
        .replace('>', "\\u003E")
}

fn diagram_json(diagram: &Diagram, options: &RenderOptions) -> Value {
    let blocks: Vec<Value> = diagram.blocks.iter().map(block_json).collect();
    let boxes: Vec<Value> = diagram
        .boxes
        .iter()
        .map(|entry| box_json(entry, options))
        .collect();
    let exchanges: Vec<Value> = diagram.exchanges.iter().map(exchange_json).collect();
    let links: Vec<Value> = diagram.links.iter().map(link_json).collect();

    json!({
        "planned": diagram.planned,
        "options": {
            "show_identifier": options.show_identifier,
            "show_param_names": options.show_param_names,
        },
        "blocks": blocks,
        "boxes": boxes,
        "exchanges": exchanges,
        "links": links,
    })
}

fn block_json(block: &Block) -> Value {
    let granularity = block.granularity.map(|granularity| granularity.to_string());
    let routing = block.routing.map(routing_text);
    let boxes: Vec<String> = block
        .boxes
        .iter()
        .map(|node| endpoint_id(Endpoint::Box(*node)))
        .collect();

    json!({
        "id": block_id(block.id),
        "boxes": boxes,
        "heading": block.heading(),
        "width": block.width,
        "granularity": granularity,
        "routing": routing,
        "reason": block.reason,
    })
}

fn box_json(entry: &DiagramBox, options: &RenderOptions) -> Value {
    let residency = entry.residency.map(|residency| residency.to_string());
    let crossing = entry.crossing.map(|crossing| crossing.verb());
    let provider = entry.provider.as_ref().map(provider_json);
    let demands: Vec<Value> = entry.demands.iter().map(demand_json).collect();

    json!({
        "id": endpoint_id(Endpoint::Box(entry.node)),
        "block": block_id(entry.block),
        "name": entry.name,
        "identifier": entry.identifier,
        "kind": kind_name(entry.kind),
        "residency": residency,
        "crossing": crossing,
        "width": entry.width,
        "reason": entry.reason,
        "lines": entry.lines(options),
        "provider": provider,
        "demands": demands,
    })
}

fn provider_json(placement: &ProviderPlacement) -> Value {
    let layers: Vec<Value> = placement
        .layers
        .iter()
        .map(|served| {
            json!({
                "layer": served.layer.0,
                "description": served.description,
                "demandedBy": served.demanded_by,
            })
        })
        .collect();

    json!({
        "kind": placement.kind.name(),
        "layers": layers,
    })
}

fn demand_json(demand: &ShownDemand) -> Value {
    let layer = demand.layer.map(|layer| layer.0);

    json!({
        "input": demand.input,
        "description": demand.description,
        "layer": layer,
        "chunks": demand.chunks,
    })
}

fn exchange_json(exchange: &Exchange) -> Value {
    let (kind, slot) = match exchange.kind {
        ExchangeKind::Disperse { slot } => ("disperse", slot),
        ExchangeKind::Aggregate => ("aggregate", None),
    };

    json!({
        "id": endpoint_id(exchange.endpoint()),
        "block": block_id(exchange.block),
        "kind": kind,
        "width": exchange.width,
        "slot": slot,
        "lines": exchange.lines(),
    })
}

fn link_json(link: &Link) -> Value {
    json!({
        "from": endpoint_id(link.from),
        "to": endpoint_id(link.to),
        "param": link.param,
        "bounded": link.bounded,
    })
}

fn block_id(block: usize) -> String {
    format!("r{block}")
}

fn endpoint_id(endpoint: Endpoint) -> String {
    match endpoint {
        Endpoint::Box(node) => format!("n{}", node.index()),
        Endpoint::Disperse(block) => format!("d{block}"),
        Endpoint::Aggregate(block) => format!("a{block}"),
    }
}

fn kind_name(kind: StageKind) -> &'static str {
    match kind {
        StageKind::Source => "source",
        StageKind::Filter => "filter",
        StageKind::Mask => "mask",
        StageKind::Provider => "provider",
        StageKind::Crossing => "crossing",
        StageKind::Sink => "sink",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::fixtures;
    use crate::testing;

    const DIAGRAM_OPEN: &str = r#"<script type="application/json" id="diagram">"#;

    fn embedded(rendered: &str) -> Value {
        let start = rendered.find(DIAGRAM_OPEN).expect("the diagram is embedded") + DIAGRAM_OPEN.len();
        let rest = &rendered[start..];
        let end = rest.find("</script>").expect("the diagram tag closes");

        serde_json::from_str(&rest[..end]).expect("the diagram is valid JSON")
    }

    #[test]
    fn the_page_is_an_html_document() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());

        assert!(rendered.starts_with("<!DOCTYPE html>"));
        assert!(rendered.trim_end().ends_with("</html>"));
    }

    #[test]
    fn both_libraries_load_pinned_with_their_integrity() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());

        assert!(rendered.contains(concat!(
            r#"src="https://cdn.jsdelivr.net/npm/elkjs@0.9.3/lib/elk.bundled.js" "#,
            r#"integrity="sha384-KO35c6PFq7tFxAEx6FtATwL9vOBOmCzQYr7GM0iTr3oO22s7Jc9OkclOH4AV7jwK" "#,
            r#"crossorigin="anonymous""#,
        )));
        assert!(rendered.contains(concat!(
            r#"src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js" "#,
            r#"integrity="sha384-yc/c2Lk1s2V2ir1rxvjo8YyVD9PlOlYTqpNr3Wm1WIuAA30GlDYNx6U5104OiavY" "#,
            r#"crossorigin="anonymous""#,
        )));
    }

    #[test]
    fn the_embedded_diagram_matches_the_model() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        let blocks = data["blocks"].as_array().expect("blocks");
        let boxes = data["boxes"].as_array().expect("boxes");
        let links = data["links"].as_array().expect("links");

        assert_eq!(data["planned"], true);
        assert_eq!(data["exchanges"].as_array().expect("exchanges").len(), 2);
        assert_eq!(blocks.len(), diagram.blocks.len());
        assert_eq!(boxes.len(), diagram.boxes.len());
        assert_eq!(links.len(), diagram.links.len());

        for (shown, block) in blocks.iter().zip(&diagram.blocks) {
            let heading = block.heading();

            assert_eq!(shown["id"], block_id(block.id));
            assert_eq!(shown["heading"], heading);
        }

        for (shown, entry) in boxes.iter().zip(&diagram.boxes) {
            let id = endpoint_id(Endpoint::Box(entry.node));

            assert_eq!(shown["id"], id);
            assert_eq!(shown["block"], block_id(entry.block));
            assert_eq!(shown["name"], entry.name);
        }

        for (shown, link) in links.iter().zip(&diagram.links) {
            assert_eq!(shown["from"], endpoint_id(link.from));
            assert_eq!(shown["to"], endpoint_id(link.to));
            assert_eq!(shown["bounded"], json!(link.bounded));
        }
    }

    #[test]
    fn the_source_links_to_the_wide_regions_disperse_unbounded() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let denoise = fixtures::node_named(&graph, "denoise");
        let plan = graph.execution_plan().expect("graph is planned");
        let block = plan.node(denoise).region().index();
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        let expected_to = format!("d{block}");
        let links = data["links"].as_array().expect("links");
        let found = links
            .iter()
            .find(|link| link["from"] == "n0" && link["to"] == expected_to.as_str());

        let link = found.expect("the source feeds the disperse");
        assert_eq!(link["bounded"], false);
    }

    #[test]
    fn a_box_carries_its_lines_and_block() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        let boxes = data["boxes"].as_array().expect("boxes");
        let denoise = boxes
            .iter()
            .find(|entry| entry["name"] == "denoise")
            .expect("denoise is drawn");

        assert_eq!(
            denoise["lines"],
            serde_json::json!(["denoise", "test/temporal", "×4"])
        );
        assert_eq!(denoise["kind"], "filter");
        assert_eq!(denoise["width"], 4);
        assert!(denoise["block"].as_str().expect("a block id").starts_with('r'));
    }

    #[test]
    fn every_demand_links_to_its_provider_and_layer() {
        let specs = fixtures::demanding_pipeline();
        let graph = fixtures::satisfied_planned(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        let boxes = data["boxes"].as_array().expect("boxes");
        let providers: Vec<&Value> = boxes
            .iter()
            .filter(|entry| !entry["provider"].is_null())
            .collect();
        let demands: Vec<&Value> = boxes
            .iter()
            .flat_map(|entry| entry["demands"].as_array().expect("demands"))
            .collect();

        assert_eq!(providers.len(), 2);
        assert_eq!(demands.len(), 3);

        for demand in demands {
            let serving: Vec<&Value> = providers
                .iter()
                .flat_map(|entry| entry["provider"]["layers"].as_array().expect("layers"))
                .filter(|served| served["layer"] == demand["layer"])
                .collect();

            assert_eq!(serving.len(), 1, "{demand}");

            let description = demand["description"].as_str().expect("a description");
            let served = serving[0]["description"].as_str().expect("a description");
            assert!(
                description.ends_with(served),
                "{description} is served as {served}"
            );
        }
    }

    #[test]
    fn a_crossing_box_names_its_verb() {
        let graph = fixtures::planned_round_trip();
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        let boxes = data["boxes"].as_array().expect("boxes");
        let upload = boxes
            .iter()
            .find(|entry| entry["name"] == "upload(input)")
            .expect("the upload is drawn");

        assert_eq!(upload["crossing"], "upload");
        assert_eq!(upload["residency"], "gpu");
    }

    /// Covers the characters that close a script tag and the placeholders the page fills in.
    #[test]
    fn a_stage_name_can_neither_close_the_tag_nor_fill_a_placeholder() {
        let hostile = "a</script><b>&{{style}}";
        let mut specs = fixtures::skipping_pipeline();
        specs[1].name = crate::Spanned::unknown(String::from(hostile));
        specs[2].inputs = vec![testing::positional(hostile), testing::positional("input")];

        let graph = fixtures::built(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        assert!(!rendered.contains("a</script>"));
        assert!(rendered.contains("a\\u003C/script\\u003E\\u003Cb\\u003E\\u0026{{style}}"));

        let boxes = data["boxes"].as_array().expect("boxes");
        assert!(boxes.iter().any(|entry| entry["name"] == hostile));
    }

    #[test]
    fn the_page_explains_itself_when_the_layout_library_is_missing() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let rendered = render_html(&graph, &RenderOptions::default());

        assert!(rendered.contains("could not be loaded"));
    }

    #[test]
    fn an_empty_graph_embeds_an_empty_diagram() {
        let graph = CompiledGraph::new(Vec::new());
        let rendered = render_html(&graph, &RenderOptions::default());
        let data = embedded(&rendered);

        assert_eq!(data["boxes"], serde_json::json!([]));
        assert!(rendered.contains("no stages to draw"));
    }

    #[test]
    fn every_kind_has_its_own_name() {
        assert_eq!(kind_name(StageKind::Source), "source");
        assert_eq!(kind_name(StageKind::Filter), "filter");
        assert_eq!(kind_name(StageKind::Mask), "mask");
        assert_eq!(kind_name(StageKind::Provider), "provider");
        assert_eq!(kind_name(StageKind::Crossing), "crossing");
        assert_eq!(kind_name(StageKind::Sink), "sink");
    }
}
