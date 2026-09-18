use av_toolbox_core::StageKind;

use super::{RenderOptions, labels_for};
use crate::graph::{CompiledGraph, Node};

/// Renders the graph as a mermaid flowchart.
///
/// Mermaid lays the nodes out itself, so this says what connects to what and nothing about where
/// anything sits.
pub fn render_mermaid(graph: &CompiledGraph, options: &RenderOptions) -> String {
    let mut lines = vec![String::from("flowchart TD")];

    for node in graph.nodes() {
        lines.push(format!("    {}", shape_for(node, options)));
    }

    for node in graph.nodes() {
        for edge in node.inputs() {
            let producer = identity(edge.source().index());
            let consumer = identity(node.id().index());

            let label = edge.param().filter(|_| options.show_param_names);
            let line = match label {
                Some(param) => format!("    {producer} -- {param} --> {consumer}"),
                None => format!("    {producer} --> {consumer}"),
            };

            lines.push(line);
        }
    }

    lines.push(String::new());
    lines.extend(CLASS_DEFINITIONS.iter().map(|line| format!("    {line}")));

    for node in graph.nodes() {
        let identity = identity(node.id().index());
        lines.push(format!("    class {identity} {};", class_of(node.kind())));
    }

    lines.join("\n")
}

const CLASS_DEFINITIONS: &[&str] = &[
    "classDef source fill:#dbeafe,stroke:#1e40af;",
    "classDef filter fill:#dcfce7,stroke:#166534;",
    "classDef mask fill:#fef3c7,stroke:#92400e;",
    "classDef transform fill:#ede9fe,stroke:#5b21b6;",
    "classDef sink fill:#fee2e2,stroke:#991b1b;",
];

fn identity(index: usize) -> String {
    format!("n{index}")
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

/// Wraps a node's label in the bracket pair mermaid draws that kind with.
fn shape_for(node: &Node, options: &RenderOptions) -> String {
    let label = labels_for(node, options).join("<br/>").replace('"', "#quot;");
    let identity = identity(node.id().index());

    match node.kind() {
        StageKind::Source => format!(r#"{identity}(["{label}"])"#),
        StageKind::Filter => format!(r#"{identity}["{label}"]"#),
        StageKind::Mask => format!(r#"{identity}{{{{"{label}"}}}}"#),
        StageKind::Transform => format!(r#"{identity}[/"{label}"/]"#),
        StageKind::Sink => format!(r#"{identity}[("{label}")]"#),
    }
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

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("blur")];

        vec![source, blur, sink]
    }

    fn render(options: &RenderOptions) -> String {
        let registry = testing::registry();
        let graph = build(&registry, &pipeline()).expect("should build").into_graph();

        render_mermaid(&graph, options)
    }

    #[test]
    fn each_kind_gets_its_own_shape() {
        let options = RenderOptions {
            show_identifier: false,
            show_param_names: false,
        };

        let rendered = render(&options);

        assert!(rendered.starts_with("flowchart TD\n"));
        assert!(rendered.contains(r#"    n0(["input"])"#));
        assert!(rendered.contains(r#"    n1["blur"]"#));
        assert!(rendered.contains(r#"    n2[("out")]"#));
    }

    #[test]
    fn an_edge_is_labelled_with_the_param_it_fills() {
        let rendered = render(&RenderOptions::default());

        assert!(rendered.contains("    n0 -- reference --> n1"));
    }

    #[test]
    fn param_labels_can_be_turned_off() {
        let options = RenderOptions {
            show_identifier: true,
            show_param_names: false,
        };

        let rendered = render(&options);

        assert!(rendered.contains("    n0 --> n1"));
        assert!(!rendered.contains("reference"));
    }

    #[test]
    fn the_identifier_goes_under_the_name() {
        let rendered = render(&RenderOptions::default());

        assert!(rendered.contains("input<br/>test/file"));
    }

    #[test]
    fn every_node_is_given_its_kind_as_a_class() {
        let rendered = render(&RenderOptions::default());

        assert!(rendered.contains("classDef source"));
        assert!(rendered.contains("    class n0 source;"));
    }
}
