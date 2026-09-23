use av_toolbox_core::StageKind;

use super::execution::{NOT_PLANNED_MESSAGE, reason_text, shared_narrowed_reason};
use super::{RenderOptions, RenderView, labels_for};
use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::plan::{ExecutionPlan, Narrowed, NodePlan, Region, RegionId};

/// Renders the graph as a mermaid flowchart.
///
/// Mermaid lays the nodes out itself, so this says what connects to what and nothing about where
/// anything sits. A planned graph additionally groups its nodes into a `subgraph` per region.
///
/// Asked for the execution view instead, it draws the plan the engine will run. Regions become
/// subgraphs holding their own disperse and aggregate alongside their stages, and every edge
/// crossing between regions is labelled by whether that link can run bounded.
pub fn render_mermaid(graph: &CompiledGraph, options: &RenderOptions) -> String {
    if options.view == RenderView::Execution {
        return render_execution_view(graph, options);
    }

    let mut lines = vec![String::from("flowchart TD")];

    match graph.execution_plan() {
        Some(plan) => lines.extend(subgraphs_for(graph, plan, options)),
        None => {
            for node in graph.nodes() {
                lines.push(format!("    {}", shape_for(node, None, options)));
            }
        },
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
        let class = match node.role() {
            NodeRole::Crossing(_) => "transfer",
            NodeRole::Stage => class_of(node.kind()),
        };
        lines.push(format!("    class {identity} {class};"));
    }

    lines.join("\n")
}

const CLASS_DEFINITIONS: &[&str] = &[
    "classDef source fill:#dbeafe,stroke:#1e40af;",
    "classDef filter fill:#dcfce7,stroke:#166534;",
    "classDef mask fill:#fef3c7,stroke:#92400e;",
    "classDef transform fill:#ede9fe,stroke:#5b21b6;",
    "classDef sink fill:#fee2e2,stroke:#991b1b;",
    "classDef transfer fill:#f3f4f6,stroke:#4b5563,stroke-dasharray:5 5;",
];

fn render_execution_view(graph: &CompiledGraph, options: &RenderOptions) -> String {
    let Some(plan) = graph.execution_plan() else {
        return String::from(NOT_PLANNED_MESSAGE);
    };

    let mut lines = vec![String::from("flowchart TD")];

    for region in plan.regions() {
        lines.extend(execution_subgraph(graph, plan, region, options));
    }

    lines.extend(execution_edges(graph, plan));

    lines.join("\n")
}

/// One region's subgraph, holding its disperse and aggregate alongside its own stages.
fn execution_subgraph(
    graph: &CompiledGraph,
    plan: &ExecutionPlan,
    region: &Region,
    options: &RenderOptions,
) -> Vec<String> {
    let shared_reason = shared_narrowed_reason(plan, region);

    let mut lines = vec![format!(
        "    subgraph r{} [\"{}\"]",
        region.id().index(),
        execution_region_label(plan, region, shared_reason)
    )];

    if region.has_disperse() {
        let shape = exchange_shape(&disperse_identity(region.id()), &disperse_label(region));
        lines.push(format!("        {shape}"));
    }

    for &id in region.nodes() {
        let node = graph.node(id);
        let node_plan = plan.node(id);
        let reason = match shared_reason {
            Some(_) => None,
            None => node_plan.narrowed().and_then(reason_text),
        };

        lines.push(format!(
            "        {}",
            execution_shape_for(node, node_plan, reason, options)
        ));
    }

    if region.has_aggregate() {
        let shape = exchange_shape(&aggregate_identity(region.id()), "aggregate");
        lines.push(format!("        {shape}"));
    }

    lines.push(String::from("    end"));
    lines
}

fn execution_region_label(plan: &ExecutionPlan, region: &Region, shared_reason: Option<Narrowed>) -> String {
    let granularity = plan.node(region.head()).parallelism();
    let header = match region.width() {
        1 => format!("region {} {granularity}", region.id().index()),
        width => format!("region {} x{width} {granularity}", region.id().index()),
    };

    match shared_reason.and_then(reason_text) {
        Some(reason) => format!("{header} ({reason})"),
        None => header,
    }
}

fn disperse_label(region: &Region) -> String {
    match region.chunk_authority() {
        Some(slot) => format!("disperse by slot {slot}"),
        None => String::from("disperse"),
    }
}

/// Mermaid's flag shape, used for an exchange since no stage kind already draws one.
fn exchange_shape(id: &str, label: &str) -> String {
    format!(r#"{id}>"{label}"]"#)
}

fn disperse_identity(region: RegionId) -> String {
    format!("d{}", region.index())
}

fn aggregate_identity(region: RegionId) -> String {
    format!("a{}", region.index())
}

/// Every edge the execution view draws, both the ones the pipeline wrote and the ones a
/// region's own exchanges add.
fn execution_edges(graph: &CompiledGraph, plan: &ExecutionPlan) -> Vec<String> {
    let mut lines = Vec::new();

    for node in graph.nodes() {
        for edge in node.inputs() {
            lines.push(execution_edge(plan, edge.source(), node.id()));
        }
    }

    for region in plan.regions() {
        lines.extend(region_exchange_edges(region));
    }

    lines
}

/// One edge the pipeline wrote, redrawn against whichever exchange stands at each end.
///
/// An edge that stays inside one region is a direct handoff inside one worker, not a channel, so
/// it carries no bounding. A frame budget bounds anything crossing between regions instead, so
/// that edge is always unbounded.
fn execution_edge(plan: &ExecutionPlan, source: NodeId, destination: NodeId) -> String {
    let source_region = plan.node(source).region();
    let destination_region = plan.node(destination).region();

    if source_region == destination_region {
        return format!(
            "    {} --> {}",
            identity(source.index()),
            identity(destination.index())
        );
    }

    let from = region_exit(plan, source_region, source);
    let to = region_entry(plan, destination_region, destination);
    format!("    {from} -- unbounded --> {to}")
}

/// The identity an edge leaving a region actually connects to.
///
/// A region with an aggregate always serialises its workers back into one stream there, so
/// anything leaving the region leaves from the aggregate rather than the node that produced it.
fn region_exit(plan: &ExecutionPlan, region: RegionId, node: NodeId) -> String {
    match plan.region(region).has_aggregate() {
        true => aggregate_identity(region),
        false => identity(node.index()),
    }
}

/// The identity an edge entering a region actually connects to.
///
/// A region with a disperse always fans an incoming chunk out to its workers there, so anything
/// entering the region enters through the disperse rather than the node reading it.
fn region_entry(plan: &ExecutionPlan, region: RegionId, node: NodeId) -> String {
    match plan.region(region).has_disperse() {
        true => disperse_identity(region),
        false => identity(node.index()),
    }
}

/// The links a region's own exchanges add, beyond the edges the pipeline wrote.
///
/// A disperse feeding its workers is unbounded, since bounding it would deadlock. Every exit
/// node feeding the aggregate back is the one link an order guarantee lets run bounded, and a
/// region can have more than one when it feeds two different downstream regions.
fn region_exchange_edges(region: &Region) -> Vec<String> {
    let mut lines = Vec::new();

    if region.has_disperse() {
        let head = identity(region.head().index());
        lines.push(format!(
            "    {} -- unbounded --> {head}",
            disperse_identity(region.id())
        ));
    }

    if region.has_aggregate() {
        let aggregate = aggregate_identity(region.id());

        for exit in region.exits() {
            lines.push(format!(
                "    {} -- bounded --> {aggregate}",
                identity(exit.index())
            ));
        }
    }

    lines
}

/// Every region as a mermaid `subgraph`, each holding its nodes' shapes.
fn subgraphs_for(graph: &CompiledGraph, plan: &ExecutionPlan, options: &RenderOptions) -> Vec<String> {
    let mut lines = Vec::new();

    for region in plan.regions() {
        lines.push(format!(
            "    subgraph r{} [\"{}\"]",
            region.id().index(),
            region_label(region)
        ));

        for &id in region.nodes() {
            let node = graph.node(id);
            let node_plan = plan.node(id);

            lines.push(format!("        {}", shape_for(node, Some(node_plan), options)));
        }

        lines.push(String::from("    end"));
    }

    lines
}

fn region_label(region: &Region) -> String {
    match region.width() {
        1 => format!("region {}", region.id().index()),
        width => format!("region {} x{width}", region.id().index()),
    }
}

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

/// A node's label lines, wrapped in the bracket pair mermaid draws its kind with.
fn shape_for(node: &Node, plan: Option<&NodePlan>, options: &RenderOptions) -> String {
    let labels = labels_for(node, plan, options);
    wrap_shape(node, &labels)
}

/// A node's execution shape, with its own narrowing reason appended when it carries one.
fn execution_shape_for(
    node: &Node,
    node_plan: &NodePlan,
    reason: Option<&'static str>,
    options: &RenderOptions,
) -> String {
    let mut labels = labels_for(node, Some(node_plan), options);
    if let Some(reason) = reason {
        labels.push(format!("({reason})"));
    }

    wrap_shape(node, &labels)
}

/// Wraps a node's label lines in the bracket pair mermaid draws that kind with.
fn wrap_shape(node: &Node, labels: &[String]) -> String {
    let label = labels.join("<br/>").replace('"', "#quot;");
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

    /// A source and a sink stay one wide, and a `test/temporal` filter between them widens.
    fn planned_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut denoise = testing::spec(StageKind::Filter, "denoise", "test/temporal");
        denoise.inputs = vec![testing::positional("input")];
        denoise.args = Spanned::unknown(json!({ "radius": 2 }));
        denoise.workers = Some(4);

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("denoise")];

        vec![source, denoise, sink]
    }

    fn planned_graph() -> CompiledGraph {
        let registry = testing::registry();
        let mut graph = build(&registry, &planned_pipeline())
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        graph
    }

    fn render_planned(options: &RenderOptions) -> String {
        render_mermaid(&planned_graph(), options)
    }

    /// `test/widening-source` is frame mode, unlike every other fixture source, so widening it
    /// is the one way to get a frame mode node above width one.
    fn frame_planned_graph() -> CompiledGraph {
        let mut source = testing::spec(StageKind::Source, "input", "test/widening-source");
        source.workers = Some(4);

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let registry = testing::registry();
        let mut graph = build(&registry, &[source, sink])
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        graph
    }

    #[test]
    fn a_planned_node_shows_its_granularity_and_width() {
        let rendered = render_planned(&RenderOptions::default());

        assert!(rendered.contains("chunk x4 by chunk"));
    }

    #[test]
    fn a_frame_mode_node_shows_it_routes_by_frame() {
        let rendered = render_mermaid(&frame_planned_graph(), &RenderOptions::default());

        assert!(rendered.contains("frame x4 by frame"));
    }

    #[test]
    fn a_node_running_one_wide_shows_no_count() {
        let rendered = render_planned(&RenderOptions::default());

        assert!(rendered.contains("stream"));
        assert!(!rendered.contains("x1"));
    }

    #[test]
    fn an_unplanned_graph_renders_as_it_always_did() {
        let rendered = render(&RenderOptions::default());

        assert!(!rendered.contains("chunk"));
        assert!(!rendered.contains("stream"));
        assert!(!rendered.contains("subgraph"));
    }

    #[test]
    fn each_region_is_drawn_as_its_own_subgraph() {
        let graph = planned_graph();
        let rendered = render_mermaid(&graph, &RenderOptions::default());

        let regions = graph.execution_plan().expect("graph is planned").regions().len();

        assert_eq!(rendered.matches("subgraph").count(), regions);
        assert_eq!(rendered.matches("\n    end").count(), regions);
    }

    /// Every region's block holds its own nodes and none of another region's.
    ///
    /// Walking every region, first and last included, rules out an off by one in the slice this
    /// test takes between one region's header and its close.
    #[test]
    fn a_node_is_drawn_inside_its_own_regions_subgraph() {
        let graph = planned_graph();
        let rendered = render_mermaid(&graph, &RenderOptions::default());
        let plan = graph.execution_plan().expect("graph is planned");

        for region in plan.regions() {
            let header = format!("subgraph r{} [", region.id().index());
            let start = rendered
                .find(&header)
                .unwrap_or_else(|| panic!("{header} is drawn"));

            let rest = &rendered[start..];
            let close = rest
                .find("\n    end")
                .unwrap_or_else(|| panic!("{header} is closed"));
            let block = &rest[..close];

            for node in graph.nodes() {
                let belongs = region.nodes().contains(&node.id());
                let drawn = block.contains(&identity(node.id().index()));

                assert_eq!(
                    drawn,
                    belongs,
                    "{header} should hold {} exactly when it belongs to that region",
                    identity(node.id().index())
                );
            }
        }
    }

    fn execution_options() -> RenderOptions {
        RenderOptions {
            view: RenderView::Execution,
            ..RenderOptions::default()
        }
    }

    fn render_planned_execution() -> String {
        render_mermaid(&planned_graph(), &execution_options())
    }

    #[test]
    fn the_execution_view_draws_each_region_as_its_own_subgraph() {
        let graph = planned_graph();
        let rendered = render_mermaid(&graph, &execution_options());

        let regions = graph.execution_plan().expect("graph is planned").regions().len();

        assert_eq!(rendered.matches("subgraph").count(), regions);
        assert_eq!(rendered.matches("\n    end").count(), regions);
    }

    /// Walking every region, not just the wide one, rules out a version that draws every
    /// region's exchanges the same way regardless of whether that region actually has one.
    #[test]
    fn a_regions_subgraph_holds_its_own_exchanges_and_none_it_does_not_have() {
        let graph = planned_graph();
        let rendered = render_mermaid(&graph, &execution_options());
        let plan = graph.execution_plan().expect("graph is planned");

        for region in plan.regions() {
            let header = format!("subgraph r{} [", region.id().index());
            let start = rendered
                .find(&header)
                .unwrap_or_else(|| panic!("{header} is drawn"));

            let rest = &rendered[start..];
            let close = rest
                .find("\n    end")
                .unwrap_or_else(|| panic!("{header} is closed"));
            let block = &rest[..close];

            assert_eq!(
                block.contains(&disperse_identity(region.id())),
                region.has_disperse(),
                "{header} should hold a disperse exactly when it has one"
            );
            assert_eq!(
                block.contains(&aggregate_identity(region.id())),
                region.has_aggregate(),
                "{header} should hold an aggregate exactly when it has one"
            );
        }
    }

    /// Pins the disperse's and the aggregate's own links to their exact bounding, rather than
    /// checking the words appear somewhere, since `"unbounded"` itself contains `"bounded"`.
    #[test]
    fn a_regions_disperse_feeds_its_head_unbounded_and_its_head_feeds_the_aggregate_bounded() {
        let graph = planned_graph();
        let rendered = render_planned_execution();
        let plan = graph.execution_plan().expect("graph is planned");

        let wide_region = plan
            .regions()
            .iter()
            .find(|region| region.width() > 1)
            .expect("a wide region");

        let disperse = disperse_identity(wide_region.id());
        let aggregate = aggregate_identity(wide_region.id());
        let head = identity(wide_region.head().index());

        assert!(rendered.contains(&format!("{disperse} -- unbounded --> {head}")));
        assert!(rendered.contains(&format!("{head} -- bounded --> {aggregate}")));
    }

    /// The source feeds the wide region's disperse, and its own head has no producer inside its
    /// own region, so this link can only be the one crossing between the two regions.
    #[test]
    fn a_link_crossing_between_two_regions_is_unbounded() {
        let graph = planned_graph();
        let rendered = render_planned_execution();
        let plan = graph.execution_plan().expect("graph is planned");

        let wide_region = plan
            .regions()
            .iter()
            .find(|region| region.width() > 1)
            .expect("a wide region");

        let source = identity(0);
        let disperse = disperse_identity(wide_region.id());

        assert!(rendered.contains(&format!("{source} -- unbounded --> {disperse}")));
    }

    /// One branch widens and one does not, and both feed a stage that reads both, so that
    /// stage's narrowing reason is `MixedInputs`.
    fn mixed_inputs_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut wide = testing::spec(StageKind::Filter, "wide", "test/temporal");
        wide.inputs = vec![testing::positional("input")];
        wide.args = Spanned::unknown(json!({ "radius": 4 }));
        wide.workers = Some(4);

        let mut narrow = testing::spec(StageKind::Filter, "narrow", "test/temporal");
        narrow.inputs = vec![testing::positional("input")];
        narrow.args = Spanned::unknown(json!({ "radius": 4 }));

        let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
        merge.inputs = vec![testing::positional("wide"), testing::positional("narrow")];
        merge.args = Spanned::unknown(json!({ "radius": 0 }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("merge")];

        vec![source, wide, narrow, merge, sink]
    }

    fn render_mixed_inputs_execution() -> String {
        let registry = testing::registry();
        let mut graph = build(&registry, &mixed_inputs_pipeline())
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        render_mermaid(&graph, &execution_options())
    }

    /// Pins the reason to the node that actually narrowed, since a version that stamped every
    /// node in the region with the same reason would also pass a bare substring check.
    #[test]
    fn a_node_joining_differently_widened_branches_says_mixed_inputs_in_the_execution_view() {
        let rendered = render_mixed_inputs_execution();

        let merge_line = rendered
            .lines()
            .find(|line| line.contains("merge"))
            .expect("a merge line");

        assert!(merge_line.contains("mixed inputs"));
        assert!(!merge_line.contains("declared"));
    }

    /// A widened stage feeding two branches that each end at their own sink, so its region has
    /// two exits rather than one.
    fn two_exit_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut wide = testing::spec(StageKind::Filter, "wide", "test/temporal");
        wide.inputs = vec![testing::positional("input")];
        wide.args = Spanned::unknown(json!({ "radius": 0 }));
        wide.workers = Some(4);

        let mut left = testing::spec(StageKind::Filter, "left", "test/temporal");
        left.inputs = vec![testing::positional("wide")];
        left.args = Spanned::unknown(json!({ "radius": 0 }));

        let mut right = testing::spec(StageKind::Filter, "right", "test/temporal");
        right.inputs = vec![testing::positional("wide")];
        right.args = Spanned::unknown(json!({ "radius": 0 }));

        let mut sink_left = testing::spec(StageKind::Sink, "out_left", "test/stdout");
        sink_left.inputs = vec![testing::positional("left")];

        let mut sink_right = testing::spec(StageKind::Sink, "out_right", "test/stdout");
        sink_right.inputs = vec![testing::positional("right")];

        vec![source, wide, left, right, sink_left, sink_right]
    }

    fn two_exit_graph() -> CompiledGraph {
        let registry = testing::registry();
        let mut graph = build(&registry, &two_exit_pipeline())
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        graph
    }

    #[test]
    fn a_region_with_two_exits_links_both_to_the_aggregate() {
        let graph = two_exit_graph();
        let rendered = render_mermaid(&graph, &execution_options());
        let plan = graph.execution_plan().expect("graph is planned");

        let wide_region = plan
            .regions()
            .iter()
            .find(|region| region.width() > 1)
            .expect("a widened region");

        let exits = wide_region.exits();
        assert_eq!(exits.len(), 2);

        let aggregate = aggregate_identity(wide_region.id());
        for exit in exits {
            let expected = format!("{} -- bounded --> {aggregate}", identity(exit.index()));
            assert!(rendered.contains(&expected), "missing link for {expected}");
        }
    }

    #[test]
    fn an_unplanned_graph_says_so_rather_than_drawing_a_flowchart() {
        let rendered = render(&execution_options());

        assert!(rendered.contains("not been planned"));
        assert!(!rendered.contains("flowchart"));
    }

    #[test]
    fn each_kind_gets_its_own_shape() {
        let options = RenderOptions {
            show_identifier: false,
            show_param_names: false,
            ..RenderOptions::default()
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
            ..RenderOptions::default()
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
    fn a_crossing_is_classed_as_a_transfer() {
        let rendered = render_mermaid(
            &crate::render::ascii::tests::placed_round_trip(),
            &RenderOptions::default(),
        );

        assert!(rendered.contains("classDef transfer"));
        assert!(
            rendered
                .lines()
                .any(|line| line.trim_start().starts_with("class ") && line.ends_with("transfer;"))
        );
    }

    #[test]
    fn every_node_is_given_its_kind_as_a_class() {
        let rendered = render(&RenderOptions::default());

        assert!(rendered.contains("classDef source"));
        assert!(rendered.contains("    class n0 source;"));
    }
}
