use av_toolbox_core::{Crossing, Parallelism, Residency, StageKind};

use super::RenderOptions;
use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::plan::{ExecutionPlan, Narrowed, Region, RegionId, Routing};

/// Everything a renderer draws, worked out once from a graph and its plan.
///
/// A block's id is its position in `blocks`, and a box sits at its node's index in `boxes`.
#[derive(Debug)]
pub(crate) struct Diagram {
    pub(crate) blocks: Vec<Block>,
    pub(crate) boxes: Vec<DiagramBox>,
    pub(crate) exchanges: Vec<Exchange>,
    pub(crate) links: Vec<Link>,
    pub(crate) planned: bool,
}

/// Boxes drawn as a group, one per region when the graph is planned and one per node otherwise.
#[derive(Debug)]
pub(crate) struct Block {
    pub(crate) id: usize,
    pub(crate) boxes: Vec<NodeId>,
    pub(crate) width: usize,
    pub(crate) granularity: Option<Parallelism>,
    pub(crate) routing: Option<Routing>,
    pub(crate) reason: Option<&'static str>,
}

#[derive(Debug)]
pub(crate) struct DiagramBox {
    pub(crate) node: NodeId,
    pub(crate) block: usize,
    pub(crate) name: String,
    pub(crate) identifier: String,
    pub(crate) kind: StageKind,
    pub(crate) residency: Option<Residency>,
    pub(crate) crossing: Option<Crossing>,
    pub(crate) width: usize,
    pub(crate) reason: Option<&'static str>,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ExchangeKind {
    Disperse { slot: Option<usize> },
    Aggregate,
}

/// Where a region's work splits across its workers, or comes back together.
#[derive(Debug)]
pub(crate) struct Exchange {
    pub(crate) block: usize,
    pub(crate) kind: ExchangeKind,
    pub(crate) width: usize,
}

/// Something a link can start or end at.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub(crate) enum Endpoint {
    Box(NodeId),
    Disperse(usize),
    Aggregate(usize),
}

/// One line drawn between two endpoints.
///
/// `bounded` is `None` for a handoff inside one worker, which is not a channel at all.
#[derive(Debug)]
pub(crate) struct Link {
    pub(crate) from: Endpoint,
    pub(crate) to: Endpoint,
    pub(crate) param: Option<&'static str>,
    pub(crate) bounded: Option<bool>,
}

impl Diagram {
    pub(crate) fn of(graph: &CompiledGraph) -> Self {
        match graph.execution_plan() {
            Some(plan) => planned(graph, plan),
            None => unplanned(graph),
        }
    }

    pub(crate) fn exchange(&self, endpoint: Endpoint) -> Option<&Exchange> {
        self.exchanges
            .iter()
            .find(|exchange| exchange.endpoint() == endpoint)
    }
}

impl Block {
    /// The line a frame prints along its top border.
    pub(crate) fn heading(&self) -> String {
        let mut parts = vec![format!("region {}", self.id)];

        if self.width > 1 {
            let width = format!("×{}", self.width);
            parts.push(width);
        }

        let mut heading = parts.join(" ");

        if let Some(granularity) = self.granularity {
            let granularity = format!(" · {granularity}");
            heading.push_str(&granularity);
        }

        if let Some(routing) = self.routing {
            let routing = format!(" · {}", routing_text(routing));
            heading.push_str(&routing);
        }

        if let Some(reason) = self.reason {
            let reason = format!(" ({reason})");
            heading.push_str(&reason);
        }

        heading
    }

    /// The line a text frame prints along its top border.
    pub(crate) fn short_heading(&self) -> String {
        let granularity = self.granularity.map(|granularity| granularity.to_string());

        match (self.width, granularity) {
            (1, Some(granularity)) => format!("region {} · {granularity}", self.id),
            (width, Some(granularity)) => format!("region {} · ×{width} {granularity}", self.id),
            (1, None) => format!("region {}", self.id),
            (width, None) => format!("region {} · ×{width}", self.id),
        }
    }

    /// The routing and the reason, for the line just inside a text frame, when there is either.
    pub(crate) fn detail(&self) -> Option<String> {
        let routing = self.routing.map(routing_text);

        match (routing, self.reason) {
            (Some(routing), Some(reason)) => Some(format!("{routing} ({reason})")),
            (Some(routing), None) => Some(routing.to_string()),
            (None, Some(reason)) => Some(format!("({reason})")),
            (None, None) => None,
        }
    }
}

impl DiagramBox {
    /// The lines of text the box holds, in the order they are printed.
    pub(crate) fn lines(&self, options: &RenderOptions) -> Vec<String> {
        let mut lines = vec![self.name.clone()];

        if options.show_identifier {
            lines.push(self.identifier.clone());
        }

        if let Some(side) = self.side_line() {
            lines.push(side);
        }

        if let Some(crossing) = self.crossing {
            let from = crossing.from_side();
            let to = crossing.to_side();
            lines.push(format!("{from}→{to}"));
        }

        if let Some(reason) = self.reason {
            lines.push(format!("({reason})"));
        }

        lines
    }

    /// The side the box runs on, followed by how many workers run it when that is more than one.
    fn side_line(&self) -> Option<String> {
        match (self.residency, self.width) {
            (Some(residency), 1) => Some(residency.to_string()),
            (Some(residency), width) => Some(format!("{residency} ×{width}")),
            (None, 1) => None,
            (None, width) => Some(format!("×{width}")),
        }
    }
}

impl Exchange {
    pub(crate) fn endpoint(&self) -> Endpoint {
        match self.kind {
            ExchangeKind::Disperse { .. } => Endpoint::Disperse(self.block),
            ExchangeKind::Aggregate => Endpoint::Aggregate(self.block),
        }
    }

    /// The symbol and ratio a text marker prints.
    pub(crate) fn ratio(&self) -> String {
        match self.kind {
            ExchangeKind::Disperse { .. } => format!("◁ 1:{}", self.width),
            ExchangeKind::Aggregate => format!("▷ {}:1", self.width),
        }
    }

    /// The ratio, the role and the bounding, one per line.
    pub(crate) fn lines(&self) -> [String; 3] {
        let ratio = self.ratio();

        match self.kind {
            ExchangeKind::Disperse { slot } => {
                let role = match slot {
                    Some(slot) => format!("disperse slot {slot}"),
                    None => String::from("disperse"),
                };

                [ratio, role, String::from("unbounded")]
            },
            ExchangeKind::Aggregate => [ratio, String::from("aggregate"), String::from("bounded")],
        }
    }
}

pub(crate) fn routing_text(routing: Routing) -> &'static str {
    match routing {
        Routing::ByFrame => "by frame",
        Routing::ByChunk => "by chunk",
    }
}

/// Names a narrowing reason worth printing.
///
/// Nothing having asked for more is the state every node starts in, so it is left out.
fn reason_text(reason: Narrowed) -> Option<&'static str> {
    match reason {
        Narrowed::Granularity => Some("granularity"),
        Narrowed::Declared => Some("declared"),
        Narrowed::MixedInputs => Some("mixed inputs"),
        Narrowed::NotAsked => None,
    }
}

fn unplanned(graph: &CompiledGraph) -> Diagram {
    let mut blocks = Vec::with_capacity(graph.len());
    let mut boxes = Vec::with_capacity(graph.len());
    let mut links = Vec::new();

    for node in graph.nodes() {
        let index = node.id().index();

        blocks.push(Block {
            id: index,
            boxes: vec![node.id()],
            width: 1,
            granularity: None,
            routing: None,
            reason: None,
        });
        boxes.push(box_of(node, index, 1, None));

        for edge in node.inputs() {
            links.push(Link {
                from: Endpoint::Box(edge.source()),
                to: Endpoint::Box(node.id()),
                param: edge.param(),
                bounded: None,
            });
        }
    }

    Diagram {
        blocks,
        boxes,
        exchanges: Vec::new(),
        links,
        planned: false,
    }
}

fn planned(graph: &CompiledGraph, plan: &ExecutionPlan) -> Diagram {
    let blocks = plan
        .regions()
        .iter()
        .map(|region| block_of(plan, region))
        .collect();
    let boxes = graph.nodes().iter().map(|node| planned_box(plan, node)).collect();
    let exchanges = plan.regions().iter().flat_map(exchanges_of).collect();

    let mut links = edge_links(graph, plan);
    for region in plan.regions() {
        let added = exchange_links(region);
        links.extend(added);
    }

    Diagram {
        blocks,
        boxes,
        exchanges,
        links,
        planned: true,
    }
}

fn block_of(plan: &ExecutionPlan, region: &Region) -> Block {
    let head_node = region.head();
    let head = plan.node(head_node);
    let shared = shared_reason(plan, region);
    let routing = match region.width() {
        1 => None,
        _ => Some(region.routing()),
    };

    Block {
        id: region.id().index(),
        boxes: region.nodes().to_vec(),
        width: region.width(),
        granularity: Some(head.parallelism()),
        routing,
        reason: shared.and_then(reason_text),
    }
}

/// A planned node's box, carrying its own reason only when its region has no shared one.
fn planned_box(plan: &ExecutionPlan, node: &Node) -> DiagramBox {
    let node_plan = plan.node(node.id());
    let region_id = node_plan.region();
    let region = plan.region(region_id);
    let shared = shared_reason(plan, region);

    let reason = match shared {
        Some(_) => None,
        None => node_plan.narrowed().and_then(reason_text),
    };

    let block = region_id.index();
    let width = node_plan.width();
    box_of(node, block, width, reason)
}

fn box_of(node: &Node, block: usize, width: usize, reason: Option<&'static str>) -> DiagramBox {
    let crossing = match node.role() {
        NodeRole::Crossing(crossing) => Some(crossing),
        NodeRole::Stage => None,
    };

    DiagramBox {
        node: node.id(),
        block,
        name: node.name().to_string(),
        identifier: node.identifier().to_string(),
        kind: node.kind(),
        residency: node.residency(),
        crossing,
        width,
        reason,
    }
}

/// The one reason every node in the region narrowed for, when they all agree.
fn shared_reason(plan: &ExecutionPlan, region: &Region) -> Option<Narrowed> {
    let mut reasons = region.nodes().iter().map(|&id| plan.node(id).narrowed());
    let first = reasons.next().flatten()?;

    reasons.all(|reason| reason == Some(first)).then_some(first)
}

fn exchanges_of(region: &Region) -> Vec<Exchange> {
    let block = region.id().index();
    let mut exchanges = Vec::new();

    if region.has_disperse() {
        exchanges.push(Exchange {
            block,
            kind: ExchangeKind::Disperse {
                slot: region.chunk_authority(),
            },
            width: region.width(),
        });
    }

    if region.has_aggregate() {
        exchanges.push(Exchange {
            block,
            kind: ExchangeKind::Aggregate,
            width: region.width(),
        });
    }

    exchanges
}

/// Every edge the pipeline wrote, redrawn against whichever exchange stands at each end.
///
/// An edge inside one region is a handoff inside one worker, so it carries no bounding. A frame
/// budget bounds anything crossing between regions instead, so that edge is always unbounded.
fn edge_links(graph: &CompiledGraph, plan: &ExecutionPlan) -> Vec<Link> {
    let mut links = Vec::new();

    for node in graph.nodes() {
        for edge in node.inputs() {
            let source_node = plan.node(edge.source());
            let source_region = source_node.region();
            let destination_node = plan.node(node.id());
            let destination_region = destination_node.region();

            let (from, to, bounded) = match source_region == destination_region {
                true => (Endpoint::Box(edge.source()), Endpoint::Box(node.id()), None),
                false => {
                    let from = region_exit(plan, source_region, edge.source());
                    let to = region_entry(plan, destination_region, node.id());
                    (from, to, Some(false))
                },
            };

            links.push(Link {
                from,
                to,
                param: edge.param(),
                bounded,
            });
        }
    }

    links
}

/// A region with an aggregate serialises its workers there, so anything leaving leaves from it.
fn region_exit(plan: &ExecutionPlan, region: RegionId, node: NodeId) -> Endpoint {
    match plan.region(region).has_aggregate() {
        true => Endpoint::Aggregate(region.index()),
        false => Endpoint::Box(node),
    }
}

/// A region with a disperse fans work out there, so anything entering enters through it.
fn region_entry(plan: &ExecutionPlan, region: RegionId, node: NodeId) -> Endpoint {
    match plan.region(region).has_disperse() {
        true => Endpoint::Disperse(region.index()),
        false => Endpoint::Box(node),
    }
}

/// The links a region's own exchanges add, beyond the edges the pipeline wrote.
///
/// A disperse feeding its workers is unbounded, since bounding it would deadlock. Every exit
/// feeding the aggregate is the one link an order guarantee lets run bounded.
fn exchange_links(region: &Region) -> Vec<Link> {
    let block = region.id().index();
    let mut links = Vec::new();

    if region.has_disperse() {
        links.push(Link {
            from: Endpoint::Disperse(block),
            to: Endpoint::Box(region.head()),
            param: None,
            bounded: Some(false),
        });
    }

    if region.has_aggregate() {
        for exit in region.exits() {
            links.push(Link {
                from: Endpoint::Box(*exit),
                to: Endpoint::Aggregate(block),
                param: None,
                bounded: Some(true),
            });
        }
    }

    links
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::fixtures;

    fn box_named<'diagram>(diagram: &'diagram Diagram, name: &str) -> &'diagram DiagramBox {
        let found = diagram.boxes.iter().find(|entry| entry.name == name);
        found.expect("the box is drawn")
    }

    fn wide_region(graph: &CompiledGraph) -> &Region {
        let plan = graph.execution_plan().expect("graph is planned");
        let found = plan.regions().iter().find(|region| region.width() > 1);

        found.expect("a wide region")
    }

    #[test]
    fn every_region_becomes_one_block_holding_its_own_nodes() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let plan = graph.execution_plan().expect("graph is planned");
        let diagram = Diagram::of(&graph);

        assert_eq!(diagram.blocks.len(), plan.regions().len());

        for (block, region) in diagram.blocks.iter().zip(plan.regions()) {
            assert_eq!(block.id, region.id().index());
            assert_eq!(block.boxes, region.nodes());
        }

        for entry in &diagram.boxes {
            let region = plan.node(entry.node).region();
            assert_eq!(entry.block, region.index());
        }
    }

    #[test]
    fn a_block_has_an_exchange_exactly_when_its_region_does() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let plan = graph.execution_plan().expect("graph is planned");
        let diagram = Diagram::of(&graph);

        for region in plan.regions() {
            let block = region.id().index();
            let disperse = Endpoint::Disperse(block);
            let aggregate = Endpoint::Aggregate(block);

            let has_disperse = diagram
                .exchanges
                .iter()
                .any(|exchange| exchange.endpoint() == disperse);
            let has_aggregate = diagram
                .exchanges
                .iter()
                .any(|exchange| exchange.endpoint() == aggregate);

            assert_eq!(has_disperse, region.has_disperse());
            assert_eq!(has_aggregate, region.has_aggregate());
        }
    }

    #[test]
    fn a_disperse_feeds_its_head_unbounded() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let region = wide_region(&graph);
        let diagram = Diagram::of(&graph);

        let from = Endpoint::Disperse(region.id().index());
        let to = Endpoint::Box(region.head());
        let found = diagram
            .links
            .iter()
            .find(|link| link.from == from && link.to == to);

        let link = found.expect("the disperse feeds the head");
        assert_eq!(link.bounded, Some(false));
        assert_eq!(link.param, None);
    }

    #[test]
    fn every_exit_feeds_the_aggregate_bounded() {
        let specs = fixtures::two_exit_pipeline();
        let graph = fixtures::planned(&specs);
        let region = wide_region(&graph);
        let diagram = Diagram::of(&graph);
        let aggregate = Endpoint::Aggregate(region.id().index());

        assert_eq!(region.exits().len(), 2);

        for exit in region.exits() {
            let from = Endpoint::Box(*exit);
            let found = diagram
                .links
                .iter()
                .find(|link| link.from == from && link.to == aggregate);

            let link = found.expect("the exit feeds the aggregate");
            assert_eq!(link.bounded, Some(true));
        }
    }

    #[test]
    fn a_link_between_regions_is_unbounded_and_attaches_to_the_exchanges() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let region = wide_region(&graph);
        let diagram = Diagram::of(&graph);
        let input = fixtures::node_named(&graph, "input");
        let out = fixtures::node_named(&graph, "out");
        let block = region.id().index();

        let entering = diagram
            .links
            .iter()
            .find(|link| link.from == Endpoint::Box(input))
            .expect("the source links onward");
        let leaving = diagram
            .links
            .iter()
            .find(|link| link.to == Endpoint::Box(out))
            .expect("the sink is linked to");

        assert_eq!(entering.to, Endpoint::Disperse(block));
        assert_eq!(entering.bounded, Some(false));
        assert_eq!(leaving.from, Endpoint::Aggregate(block));
        assert_eq!(leaving.bounded, Some(false));
    }

    #[test]
    fn a_link_inside_one_region_has_no_bounding() {
        let specs = fixtures::two_exit_pipeline();
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let wide = fixtures::node_named(&graph, "wide");
        let left = fixtures::node_named(&graph, "left");

        let from = Endpoint::Box(wide);
        let to = Endpoint::Box(left);
        let found = diagram
            .links
            .iter()
            .find(|link| link.from == from && link.to == to);

        let link = found.expect("wide feeds left directly");
        assert_eq!(link.bounded, None);
    }

    #[test]
    fn a_reason_every_node_shares_sits_on_the_block_rather_than_the_box() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let out = box_named(&diagram, "out");

        assert_eq!(diagram.blocks[out.block].reason, Some("granularity"));
        assert_eq!(out.reason, None);
    }

    #[test]
    fn a_node_pinned_to_one_worker_says_declared() {
        let specs = fixtures::declared_pipeline();
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);

        assert_eq!(box_named(&diagram, "pinned").reason, Some("declared"));
    }

    #[test]
    fn a_node_joining_differently_widened_branches_says_mixed_inputs() {
        let specs = fixtures::mixed_inputs_pipeline();
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);

        assert_eq!(box_named(&diagram, "merge").reason, Some("mixed inputs"));
    }

    #[test]
    fn a_node_nobody_asked_to_widen_gives_no_reason() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let input = box_named(&diagram, "input");

        assert_eq!(input.reason, None);
        assert_eq!(diagram.blocks[input.block].reason, None);
    }

    #[test]
    fn a_wide_chunk_region_routes_by_chunk() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let denoise = box_named(&diagram, "denoise");

        assert_eq!(diagram.blocks[denoise.block].routing, Some(Routing::ByChunk));
    }

    #[test]
    fn a_wide_frame_region_routes_by_frame() {
        let specs = fixtures::frame_pipeline();
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let input = box_named(&diagram, "input");

        assert_eq!(diagram.blocks[input.block].routing, Some(Routing::ByFrame));
    }

    #[test]
    fn a_one_wide_region_has_no_routing() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let out = box_named(&diagram, "out");

        assert_eq!(diagram.blocks[out.block].routing, None);
    }

    #[test]
    fn the_heading_names_the_width_granularity_routing_and_reason() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let denoise = box_named(&diagram, "denoise");
        let out = box_named(&diagram, "out");

        let wide_heading = diagram.blocks[denoise.block].heading();
        let narrow_heading = diagram.blocks[out.block].heading();

        let expected_wide = format!("region {} ×4 · chunk · by chunk", denoise.block);
        let expected_narrow = format!("region {} · stream (granularity)", out.block);
        assert_eq!(wide_heading, expected_wide);
        assert_eq!(narrow_heading, expected_narrow);
    }

    #[test]
    fn a_wide_box_prints_its_width_after_its_name_and_identifier() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let denoise = box_named(&diagram, "denoise");

        let shown = denoise.lines(&RenderOptions::default());
        let hidden_options = RenderOptions {
            show_identifier: false,
            ..RenderOptions::default()
        };
        let hidden = denoise.lines(&hidden_options);

        assert_eq!(shown, vec!["denoise", "test/temporal", "×4"]);
        assert_eq!(hidden, vec!["denoise", "×4"]);
    }

    #[test]
    fn a_crossing_box_carries_its_direction() {
        let graph = fixtures::planned_round_trip();
        let diagram = Diagram::of(&graph);
        let upload = box_named(&diagram, "upload(input)");

        let lines = upload.lines(&RenderOptions::default());

        assert_eq!(upload.crossing, Some(Crossing::Upload));
        assert!(lines.contains(&String::from("host→gpu")));
    }

    #[test]
    fn each_exchange_prints_its_ratio_role_and_bounding() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let disperse = diagram
            .exchanges
            .iter()
            .find(|exchange| matches!(exchange.kind, ExchangeKind::Disperse { .. }))
            .expect("a disperse");
        let aggregate = diagram
            .exchanges
            .iter()
            .find(|exchange| exchange.kind == ExchangeKind::Aggregate)
            .expect("an aggregate");

        assert_eq!(disperse.lines(), ["◁ 1:4", "disperse slot 0", "unbounded"]);
        assert_eq!(aggregate.lines(), ["▷ 4:1", "aggregate", "bounded"]);
    }

    #[test]
    fn the_short_heading_names_the_width_and_granularity() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let denoise = box_named(&diagram, "denoise");
        let out = box_named(&diagram, "out");

        let wide = diagram.blocks[denoise.block].short_heading();
        let narrow = diagram.blocks[out.block].short_heading();

        assert_eq!(wide, format!("region {} · ×4 chunk", denoise.block));
        assert_eq!(narrow, format!("region {} · stream", out.block));
    }

    #[test]
    fn the_detail_carries_the_routing_and_the_reason() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let denoise = box_named(&diagram, "denoise");
        let out = box_named(&diagram, "out");
        let input = box_named(&diagram, "input");

        assert_eq!(
            diagram.blocks[denoise.block].detail(),
            Some(String::from("by chunk"))
        );
        assert_eq!(
            diagram.blocks[out.block].detail(),
            Some(String::from("(granularity)"))
        );
        assert_eq!(diagram.blocks[input.block].detail(), None);
    }

    #[test]
    fn each_exchange_prints_its_ratio() {
        let specs = fixtures::wide_pipeline(4);
        let graph = fixtures::planned(&specs);
        let diagram = Diagram::of(&graph);
        let ratios: Vec<String> = diagram
            .exchanges
            .iter()
            .map(|exchange| exchange.ratio())
            .collect();

        assert_eq!(ratios, vec!["◁ 1:4", "▷ 4:1"]);
    }

    #[test]
    fn an_unplanned_graph_gets_one_block_per_node_and_no_exchanges() {
        let specs = fixtures::skipping_pipeline();
        let graph = fixtures::built(&specs);
        let diagram = Diagram::of(&graph);

        assert!(!diagram.planned);
        assert_eq!(diagram.blocks.len(), graph.len());
        assert!(diagram.exchanges.is_empty());
        assert!(diagram.links.iter().all(|link| link.bounded.is_none()));
    }
}
