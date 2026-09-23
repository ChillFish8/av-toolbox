use super::{RenderOptions, routing_text};
use crate::graph::{CompiledGraph, NodeId};
use crate::plan::{ExecutionPlan, Narrowed, Region};

/// Shown in place of any execution view a graph has no plan to draw.
pub(crate) const NOT_PLANNED_MESSAGE: &str =
    "this graph has not been planned yet, so there is no execution view to draw";

/// How wide a node's name column is, so the identifier beside it lines up down a region.
const NAME_COLUMN: usize = 22;

/// How wide an exchange's description is, so the bounding beside it lines up down a region.
const EXCHANGE_COLUMN: usize = 40;

/// Reports what the engine will actually run, rather than the graph the pipeline declared.
///
/// Regions are walked in the order they run. Each one prints its width, the granularity that
/// set it, how it routes above width one, its nodes, and its exchanges marked bounded or
/// unbounded. A node held at width one names why, from [Narrowed](crate::plan::Narrowed),
/// whenever the reason is more than nothing having asked for more.
pub(crate) fn render_execution(graph: &CompiledGraph, options: &RenderOptions) -> String {
    let Some(plan) = graph.execution_plan() else {
        return String::from(NOT_PLANNED_MESSAGE);
    };

    let blocks: Vec<String> = plan
        .regions()
        .iter()
        .map(|region| region_report(graph, plan, region, options))
        .collect();

    blocks.join("\n\n")
}

fn region_report(
    graph: &CompiledGraph,
    plan: &ExecutionPlan,
    region: &Region,
    options: &RenderOptions,
) -> String {
    let shared_reason = shared_narrowed_reason(plan, region);

    let mut lines = vec![region_header(plan, region, shared_reason)];

    for &id in region.nodes() {
        lines.push(node_line(graph, plan, id, shared_reason, options));
    }

    lines.extend(exchange_lines(region));

    lines.join("\n")
}

fn region_header(plan: &ExecutionPlan, region: &Region, shared_reason: Option<Narrowed>) -> String {
    let granularity = plan.node(region.head()).parallelism();
    let mut header = format!(
        "region {}  x{}  {granularity}",
        region.id().index(),
        region.width()
    );

    if region.width() > 1 {
        let routing = routing_text(region.routing());
        header.push_str(&format!("  {routing}"));
    }

    match shared_reason.and_then(reason_text) {
        Some(reason) => format!("{header}    ({reason})"),
        None => header,
    }
}

/// The one reason every node in the region narrowed for, when they all agree.
///
/// A region's nodes all run at the same width, so a node at width one always carries a reason.
pub(super) fn shared_narrowed_reason(plan: &ExecutionPlan, region: &Region) -> Option<Narrowed> {
    let mut reasons = region.nodes().iter().map(|&id| plan.node(id).narrowed());
    let first = reasons.next().flatten()?;

    reasons.all(|reason| reason == Some(first)).then_some(first)
}

fn node_line(
    graph: &CompiledGraph,
    plan: &ExecutionPlan,
    id: NodeId,
    shared_reason: Option<Narrowed>,
    options: &RenderOptions,
) -> String {
    let node = graph.node(id);
    let mut line = format!("  {:<width$}", node.name(), width = NAME_COLUMN);

    if options.show_identifier {
        line.push_str(node.identifier());
    }

    if shared_reason.is_none() {
        let own_reason = plan.node(id).narrowed().and_then(reason_text);

        if let Some(reason) = own_reason {
            line.push_str(&format!("    ({reason})"));
        }
    }

    line
}

fn exchange_lines(region: &Region) -> Vec<String> {
    if !region.has_disperse() && !region.has_aggregate() {
        return vec![String::from("  no exchanges")];
    }

    let mut lines = Vec::new();

    if region.has_disperse() {
        lines.push(exchange_line(disperse_description(region), "unbounded"));
    }

    if region.has_aggregate() {
        lines.push(exchange_line(String::from("aggregate"), "bounded"));
    }

    lines
}

fn disperse_description(region: &Region) -> String {
    match region.chunk_authority() {
        Some(slot) => format!("disperse by slot {slot}"),
        None => String::from("disperse"),
    }
}

fn exchange_line(description: String, bounding: &str) -> String {
    format!("  {:<width$}{bounding}", description, width = EXCHANGE_COLUMN)
}

/// Names a narrowing reason worth reporting.
///
/// Nothing having asked for more is the default state a node starts in, not an explanation of
/// anything unusual, so it is left out rather than printed.
pub(super) fn reason_text(reason: Narrowed) -> Option<&'static str> {
    match reason {
        Narrowed::Granularity => Some("granularity"),
        Narrowed::Declared => Some("declared"),
        Narrowed::MixedInputs => Some("mixed inputs"),
        Narrowed::NotAsked => None,
    }
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::render::{RenderView, render_ascii, render_svg};
    use crate::{Spanned, StageSpec, build, testing};

    fn planned_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut denoise = testing::spec(StageKind::Filter, "denoise", "test/temporal");
        denoise.inputs = vec![testing::positional("input")];
        denoise.args = Spanned::unknown(json!({ "radius": 4 }));
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

    fn execution_options() -> RenderOptions {
        RenderOptions {
            view: RenderView::Execution,
            ..RenderOptions::default()
        }
    }

    fn render_planned_execution() -> String {
        render_ascii(&planned_graph(), &execution_options())
    }

    fn render_unplanned_execution() -> String {
        let registry = testing::registry();

        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let graph = build(&registry, &[source, sink])
            .expect("should build")
            .into_graph();

        render_ascii(&graph, &execution_options())
    }

    fn render_svg_execution() -> String {
        render_svg(&planned_graph(), &execution_options())
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

    fn render_frame_planned_execution() -> String {
        render_ascii(&frame_planned_graph(), &execution_options())
    }

    /// A stage explicitly pinned to one worker, rather than one that simply never asked for
    /// more, so its narrowing reason is `Declared` rather than the deliberately unprinted
    /// `NotAsked`.
    fn declared_pipeline() -> Vec<StageSpec> {
        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut pinned = testing::spec(StageKind::Filter, "pinned", "test/temporal");
        pinned.inputs = vec![testing::positional("input")];
        pinned.args = Spanned::unknown(json!({ "radius": 4 }));
        pinned.workers = Some(1);

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("pinned")];

        vec![source, pinned, sink]
    }

    fn render_declared_pipeline() -> String {
        let registry = testing::registry();
        let mut graph = build(&registry, &declared_pipeline())
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        render_ascii(&graph, &execution_options())
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

    fn render_mixed_inputs_pipeline() -> String {
        let registry = testing::registry();
        let mut graph = build(&registry, &mixed_inputs_pipeline())
            .expect("should build")
            .into_graph();

        let made = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        render_ascii(&graph, &execution_options())
    }

    #[test]
    fn a_node_pinned_to_one_worker_says_declared() {
        let report = render_declared_pipeline();

        let pinned_line = report
            .lines()
            .find(|line| line.contains("pinned"))
            .expect("a pinned line");

        assert!(pinned_line.contains("(declared)"));
        assert!(!pinned_line.contains("mixed inputs"));
    }

    #[test]
    fn a_node_joining_differently_widened_branches_says_mixed_inputs() {
        let report = render_mixed_inputs_pipeline();

        let merge_line = report
            .lines()
            .find(|line| line.contains("merge"))
            .expect("a merge line");

        assert!(merge_line.contains("(mixed inputs)"));
        assert!(!merge_line.contains("declared"));
    }

    #[test]
    fn every_region_is_reported_with_its_width_and_granularity() {
        let report = render_planned_execution();

        assert!(report.contains("x4"));
        assert!(report.contains("by chunk"));
    }

    /// A chunk routed plan alone cannot tell this view apart from one that always reports
    /// `"by chunk"` regardless of how a region actually routes.
    #[test]
    fn a_frame_routed_region_is_reported_as_routing_by_frame() {
        let report = render_frame_planned_execution();

        assert!(report.contains("by frame"));
    }

    #[test]
    fn a_wide_region_reports_both_its_exchanges_and_how_they_are_bounded() {
        let report = render_planned_execution();

        assert!(report.contains("disperse"));
        assert!(report.contains("aggregate"));
        assert!(report.contains("bounded"));
        assert!(report.contains("unbounded"));
    }

    /// The weak substring checks above cannot tell a disperse and an aggregate apart, since
    /// `"unbounded"` itself contains `"bounded"`. This pins each exchange to its own bounding.
    #[test]
    fn the_disperse_line_and_the_aggregate_line_are_bounded_oppositely() {
        let report = render_planned_execution();

        let disperse_line = report
            .lines()
            .find(|line| line.contains("disperse"))
            .expect("a disperse line");
        let aggregate_line = report
            .lines()
            .find(|line| line.contains("aggregate"))
            .expect("an aggregate line");

        assert_eq!(disperse_line.split_whitespace().last(), Some("unbounded"));
        assert_eq!(aggregate_line.split_whitespace().last(), Some("bounded"));
    }

    #[test]
    fn a_disperse_names_the_slot_it_routes_by() {
        let report = render_planned_execution();

        assert!(report.contains("slot 0"));
    }

    /// The one question this view exists to answer.
    #[test]
    fn a_node_that_stayed_narrow_says_why() {
        let report = render_planned_execution();

        assert!(report.contains("granularity"));
    }

    /// A node nobody ever asked to widen has nothing unusual to explain, so its region carries
    /// no reason at all, unlike the sink held narrow by its granularity.
    #[test]
    fn a_node_that_was_simply_never_asked_to_widen_gives_no_reason() {
        let report = render_planned_execution();

        let source_region = report.lines().next().expect("a first region line");

        assert!(!source_region.contains('('));
    }

    /// Every node's name and its identifier belong on the same line as each other.
    #[test]
    fn a_nodes_name_and_identifier_share_a_line() {
        let report = render_planned_execution();

        let denoise_line = report
            .lines()
            .find(|line| line.contains("denoise"))
            .expect("a denoise line");

        assert!(denoise_line.contains("test/temporal"));
    }

    /// Both narrow regions say plainly that they have nothing to report, rather than the report
    /// only ever mentioning exchanges when a region happens to have one.
    #[test]
    fn a_region_without_exchanges_says_so() {
        let report = render_planned_execution();

        assert_eq!(report.matches("no exchanges").count(), 2);
    }

    #[test]
    fn an_unplanned_graph_says_so_rather_than_drawing_nothing() {
        let report = render_unplanned_execution();

        assert!(report.contains("not been planned"));
    }

    /// The brief asks for a one line message, not an error page, so a stray extra line would
    /// mean the render path fell through into drawing the declared graph after all.
    #[test]
    fn an_unplanned_graph_reports_exactly_one_line() {
        let report = render_unplanned_execution();

        assert_eq!(report.lines().count(), 1);
    }

    #[test]
    fn a_crossing_is_listed_inside_its_region() {
        let mut graph = crate::render::ascii::tests::placed_round_trip();
        let made = crate::plan::plan(&graph, &testing::registry()).expect("should plan");
        graph.set_execution_plan(made);

        let report = render_ascii(&graph, &execution_options());

        assert!(
            report
                .lines()
                .any(|line| line.trim_start().starts_with("unload(blur)"))
        );
    }

    #[test]
    fn svg_has_no_execution_view_and_says_so() {
        let rendered = render_svg_execution();

        assert!(rendered.contains("no execution view"));
    }

    /// SVG refuses the execution view outright, before it ever asks whether the graph has a
    /// plan, so an unplanned graph gets the same one line refusal rather than the "not been
    /// planned" message every other renderer would give it.
    #[test]
    fn svg_gives_its_own_refusal_even_when_the_graph_is_unplanned() {
        let registry = testing::registry();

        let mut source = testing::spec(StageKind::Source, "input", "test/file");
        source.args = Spanned::unknown(json!({ "path": "a.mkv" }));

        let mut sink = testing::spec(StageKind::Sink, "out", "test/stdout");
        sink.inputs = vec![testing::positional("input")];

        let graph = build(&registry, &[source, sink])
            .expect("should build")
            .into_graph();
        let rendered = render_svg(&graph, &execution_options());

        assert!(rendered.contains("no execution view"));
        assert!(!rendered.contains("not been planned"));
    }
}
