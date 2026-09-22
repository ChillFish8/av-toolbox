use av_toolbox_core::{Parallelism, PipelineRegistry, PreparedStage};

use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, NodeId};
use crate::{Diagnostic, DiagnosticCode, Diagnostics};

/// A region's position in the plan.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub struct RegionId(usize);

impl RegionId {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(&self) -> usize {
        self.0
    }
}

/// Why a node ended up running one worker wide.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Narrowed {
    /// Its granularity is the whole stream.
    Granularity,
    /// The pipeline asked for one.
    Declared,
    /// Its inputs come from regions of differing widths.
    MixedInputs,
    /// Nothing asked for more.
    NotAsked,
}

/// How one node will be run.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct NodePlan {
    parallelism: Parallelism,
    width: usize,
    region: RegionId,
    narrowed: Option<Narrowed>,
}

impl NodePlan {
    /// A node whose stage never prepared, since `prepare_stages` already reported why.
    fn fallback() -> Self {
        Self {
            parallelism: Parallelism::Stream,
            width: 1,
            region: RegionId::new(0),
            narrowed: Some(Narrowed::NotAsked),
        }
    }

    pub fn parallelism(&self) -> Parallelism {
        self.parallelism
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn region(&self) -> RegionId {
        self.region
    }

    pub fn narrowed(&self) -> Option<Narrowed> {
        self.narrowed
    }
}

/// A set of nodes one worker runs a private copy of.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Region {
    id: RegionId,
    width: usize,
    head: NodeId,
    nodes: Vec<NodeId>,
    chunk_authority: Option<usize>,
    disperse: bool,
    aggregate: bool,
    exits: Vec<NodeId>,
}

impl Region {
    pub fn id(&self) -> RegionId {
        self.id
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn head(&self) -> NodeId {
        self.head
    }

    /// The region's nodes in the order they joined, which is topological, so the head is first.
    pub fn nodes(&self) -> &[NodeId] {
        &self.nodes
    }

    /// The `Inputs` slot the disperse routes the head node's chunks by.
    ///
    /// [Edge::slot](crate::graph::Edge::slot) is the matching space, so the edge feeding this
    /// authority is the one whose slot equals this index.
    pub fn chunk_authority(&self) -> Option<usize> {
        self.chunk_authority
    }

    pub fn has_disperse(&self) -> bool {
        self.disperse
    }

    pub fn has_aggregate(&self) -> bool {
        self.aggregate
    }

    /// The nodes in this region an edge outside it reads from, in the order they were placed.
    pub fn exits(&self) -> &[NodeId] {
        &self.exits
    }
}

/// How wide every node of a graph runs.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    nodes: Vec<NodePlan>,
    regions: Vec<Region>,
}

impl ExecutionPlan {
    pub fn node(&self, id: NodeId) -> &NodePlan {
        &self.nodes[id.index()]
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub fn region(&self, id: RegionId) -> &Region {
        &self.regions[id.index()]
    }
}

/// Works out how many instances of each node run, and groups them into regions.
///
/// A node whose granularity is the whole stream always runs one wide, and asking it to widen is
/// an error rather than something to ignore. Otherwise a node runs as wide as the pipeline
/// declared, or as wide as its inputs agree on when it declared nothing, or one wide otherwise.
///
/// Once every width is settled, nodes are grouped into regions in the same topological order. A
/// node joins its producer's region when every input comes from that one region and the width
/// matches, and heads a region of its own otherwise.
pub fn plan(graph: &CompiledGraph, registry: &PipelineRegistry) -> Result<ExecutionPlan, Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);

    let mut planner = Planner {
        graph,
        prepared,
        nodes: (0..graph.len()).map(|_| None).collect(),
        regions: Vec::new(),
        diagnostics,
    };

    for id in graph.order() {
        planner.decide(*id);
    }

    for id in graph.order() {
        planner.place_in_region(*id);
    }

    planner.compute_exits();

    planner.finish()
}

/// The state one planning pass carries from node to node.
struct Planner<'a> {
    graph: &'a CompiledGraph,
    prepared: Vec<Option<PreparedStage>>,
    nodes: Vec<Option<NodePlan>>,
    regions: Vec<Region>,
    diagnostics: Diagnostics,
}

impl Planner<'_> {
    /// Settles the width of one node.
    ///
    /// A node whose stage never prepared has no granularity to read, so it is left at width one
    /// without a second diagnostic, since `prepare_stages` already reported why it failed.
    fn decide(&mut self, id: NodeId) {
        let Some(stage) = self.prepared[id.index()].as_ref() else {
            self.nodes[id.index()] = Some(NodePlan::fallback());
            return;
        };

        let node = self.graph.node(id);

        if node.workers() == Some(0) {
            self.report_zero_workers(id);
            self.nodes[id.index()] = Some(NodePlan::fallback());
            return;
        }

        let parallelism = stage.parallelism();

        if parallelism == Parallelism::Stream && node.workers().is_some_and(|workers| workers > 1) {
            self.report_whole_stream(id);
            self.nodes[id.index()] = Some(NodePlan::fallback());
            return;
        }

        let (candidate, reason) = self.candidate_width(id, node.workers());
        let (width, narrowed) = clamp_to_granularity(parallelism, candidate, reason);

        self.nodes[id.index()] = Some(NodePlan {
            parallelism,
            width,
            region: RegionId::new(0),
            narrowed,
        });
    }

    /// The width a node would run at before its own granularity can narrow it back down.
    ///
    /// A declared width wins outright. Otherwise every input's already-decided width is compared,
    /// and the node takes that width when they all agree, or one when they do not.
    fn candidate_width(&self, id: NodeId, requested: Option<usize>) -> (usize, Option<Narrowed>) {
        if let Some(workers) = requested {
            let reason = if workers == 1 {
                Some(Narrowed::Declared)
            } else {
                None
            };
            return (workers, reason);
        }

        let sources: Vec<NodeId> = self
            .graph
            .node(id)
            .inputs()
            .iter()
            .map(|edge| edge.source())
            .collect();

        let Some(first) = sources.first() else {
            return (1, Some(Narrowed::NotAsked));
        };

        let leading = self.width_of(*first);
        let agrees = sources.iter().all(|source| self.width_of(*source) == leading);

        if !agrees {
            return (1, Some(Narrowed::MixedInputs));
        }

        let reason = if leading == 1 {
            Some(Narrowed::NotAsked)
        } else {
            None
        };
        (leading, reason)
    }

    fn width_of(&self, id: NodeId) -> usize {
        self.nodes[id.index()].as_ref().map(NodePlan::width).unwrap_or(1)
    }

    /// Places one node into a region, following a producer's region or opening a new one.
    fn place_in_region(&mut self, id: NodeId) {
        let width = self.width_of(id);
        let sources: Vec<NodeId> = self
            .graph
            .node(id)
            .inputs()
            .iter()
            .map(|edge| edge.source())
            .collect();

        match self.shared_producer_region(&sources, width) {
            Some(region) => self.join_region(id, region),
            None => {
                self.open_region(id, width);
            },
        }
    }

    /// The one region every input already shares, when its width matches this node's own.
    fn shared_producer_region(&self, sources: &[NodeId], width: usize) -> Option<RegionId> {
        let first_source = *sources.first()?;
        let candidate = self.region_of(first_source);

        let all_agree = sources.iter().all(|source| self.region_of(*source) == candidate);
        if !all_agree {
            return None;
        }

        let region = &self.regions[candidate.index()];
        (region.width == width).then_some(candidate)
    }

    fn region_of(&self, id: NodeId) -> RegionId {
        self.nodes[id.index()]
            .as_ref()
            .map(NodePlan::region)
            .expect("a producer is placed into a region before its consumer, since order is topological")
    }

    /// Adds a node to an existing region.
    fn join_region(&mut self, id: NodeId, region: RegionId) {
        self.regions[region.index()].nodes.push(id);
        self.set_region(id, region);
    }

    /// Opens a region headed by the given node, with its exchanges set from the rules.
    ///
    /// An aggregate exists above one worker. A disperse exists when the head node has an input
    /// to route, and either the region runs above one worker or the head has more than one
    /// input to pair.
    fn open_region(&mut self, head: NodeId, width: usize) -> RegionId {
        let id = RegionId::new(self.regions.len());
        let head_inputs = self.graph.node(head).inputs().len();
        let chunk_authority = self.prepared[head.index()]
            .as_ref()
            .and_then(PreparedStage::chunk_authority);

        self.regions.push(Region {
            id,
            width,
            head,
            nodes: vec![head],
            chunk_authority,
            disperse: head_inputs > 0 && (width > 1 || head_inputs > 1),
            aggregate: width > 1,
            exits: Vec::new(),
        });

        self.set_region(head, id);
        id
    }

    fn set_region(&mut self, id: NodeId, region: RegionId) {
        if let Some(node) = self.nodes[id.index()].as_mut() {
            node.region = region;
        }
    }

    /// Records which node in each region an edge outside it reads from.
    ///
    /// Runs once every node has a region, since a node's own consumers are only known to have
    /// settled once the whole graph has been walked.
    fn compute_exits(&mut self) {
        let regions_of: Vec<Option<RegionId>> = self
            .nodes
            .iter()
            .map(|plan| plan.as_ref().map(NodePlan::region))
            .collect();
        let graph = self.graph;

        for region in &mut self.regions {
            let region_id = region.id;

            region.exits = region
                .nodes
                .iter()
                .copied()
                .filter(|&id| {
                    graph
                        .node(id)
                        .consumers()
                        .iter()
                        .any(|consumer| regions_of[consumer.index()] != Some(region_id))
                })
                .collect();
        }
    }

    fn report_zero_workers(&mut self, id: NodeId) {
        let node = self.graph.node(id);
        let message = format!(
            "{} asked for zero workers, but a stage runs at least one worker",
            node.name()
        );

        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        ));
    }

    fn report_whole_stream(&mut self, id: NodeId) {
        let node = self.graph.node(id);
        let message = format!(
            "{} takes the whole stream at once, so it runs one worker wide",
            node.name()
        );

        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        ));
    }

    fn finish(self) -> Result<ExecutionPlan, Diagnostics> {
        if self.diagnostics.has_errors() {
            return Err(self.diagnostics);
        }

        let nodes = self
            .nodes
            .into_iter()
            .map(|plan| plan.expect("every node is decided while walking the topological order"))
            .collect();

        Ok(ExecutionPlan {
            nodes,
            regions: self.regions,
        })
    }
}

/// Applies a node's granularity to the width it would otherwise run at.
///
/// A node whose granularity is the whole stream always ends up one wide, and the reason recorded
/// then names the granularity rather than whatever reason the pre-clamp width carried, since a
/// stage that only ever takes the whole stream could never have widened at any candidate width.
/// The one exception is a candidate that was never asked to widen at all, since a source really
/// is one wide because nothing asked.
fn clamp_to_granularity(
    parallelism: Parallelism,
    candidate: usize,
    reason: Option<Narrowed>,
) -> (usize, Option<Narrowed>) {
    if parallelism != Parallelism::Stream {
        let narrowed = if candidate == 1 { reason } else { None };
        return (candidate, narrowed);
    }

    if reason == Some(Narrowed::NotAsked) {
        return (1, reason);
    }

    (1, Some(Narrowed::Granularity))
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::StageKind;
    use serde_json::json;

    use super::*;
    use crate::{Diagnostics, Spanned, StageSpec, build, testing};

    fn source(name: &str) -> StageSpec {
        testing::spec(StageKind::Source, name, "test/wide")
    }

    fn sink(name: &str, from: &str) -> StageSpec {
        let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
        spec.inputs = vec![testing::positional(from)];

        spec
    }

    fn temporal(name: &str, input: &str, radius: u32) -> StageSpec {
        let mut spec = testing::spec(StageKind::Filter, name, "test/temporal");
        spec.inputs = vec![testing::positional(input)];
        spec.args = Spanned::unknown(json!({ "radius": radius }));

        spec
    }

    fn plan_of(specs: &[StageSpec]) -> ExecutionPlan {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();

        plan(&graph, &registry).expect("should plan")
    }

    /// Plans a graph while keeping the graph itself around, for a test that needs to look at an
    /// edge alongside the plan.
    fn graph_and_plan_of(specs: &[StageSpec]) -> (CompiledGraph, ExecutionPlan) {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();
        let made = plan(&graph, &registry).expect("should plan");

        (graph, made)
    }

    fn plan_error(specs: &[StageSpec]) -> Diagnostics {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();

        plan(&graph, &registry).expect_err("should fail to plan")
    }

    #[test]
    fn a_stage_that_asked_for_nothing_runs_one_wide() {
        let plan = plan_of(&[source("input"), sink("out", "input")]);

        assert_eq!(plan.node(NodeId::new(0)).width(), 1);
        assert_eq!(plan.node(NodeId::new(0)).narrowed(), Some(Narrowed::NotAsked));
    }

    #[test]
    fn a_frame_stage_widens_to_what_it_asked_for() {
        let mut spatial = temporal("spatial", "input", 0);
        spatial.workers = Some(4);

        let plan = plan_of(&[source("input"), spatial, sink("out", "spatial")]);

        assert_eq!(plan.node(NodeId::new(1)).parallelism(), Parallelism::Frame);
        assert_eq!(plan.node(NodeId::new(1)).width(), 4);
        assert_eq!(plan.node(NodeId::new(1)).narrowed(), None);
    }

    #[test]
    fn a_width_propagates_to_a_consumer_that_asked_for_nothing() {
        let mut spatial = temporal("spatial", "input", 0);
        spatial.workers = Some(4);
        let next = temporal("next", "spatial", 0);

        let plan = plan_of(&[source("input"), spatial, next, sink("out", "next")]);

        assert_eq!(plan.node(NodeId::new(2)).width(), 4);
    }

    /// A sink is `Stream`, so a width reaching it stops rather than carrying on.
    #[test]
    fn a_stream_stage_stays_one_wide_however_wide_its_input_is() {
        let mut spatial = temporal("spatial", "input", 0);
        spatial.workers = Some(4);

        let plan = plan_of(&[source("input"), spatial, sink("out", "spatial")]);

        assert_eq!(plan.node(NodeId::new(2)).width(), 1);
        assert_eq!(plan.node(NodeId::new(2)).narrowed(), Some(Narrowed::Granularity));
    }

    #[test]
    fn a_stream_stage_asked_to_widen_is_an_error() {
        let mut sink = sink("out", "input");
        sink.workers = Some(4);

        let diagnostics = plan_error(&[source("input"), sink]);

        assert_eq!(diagnostics.iter().count(), 1);
        let reported = diagnostics.iter().next().expect("a diagnostic");
        assert_eq!(reported.code(), DiagnosticCode::InvalidWorkers);
        assert!(reported.message().contains("whole stream"));
    }

    #[test]
    fn asking_for_no_workers_at_all_is_an_error() {
        let mut spatial = temporal("spatial", "input", 0);
        spatial.workers = Some(0);

        let diagnostics = plan_error(&[source("input"), spatial, sink("out", "spatial")]);

        let reported = diagnostics.iter().next().expect("a diagnostic");
        assert_eq!(reported.code(), DiagnosticCode::InvalidWorkers);
    }

    /// `test/temporal-merge` at `radius: 0` is `Frame`, so nothing about its own granularity
    /// stops it widening. The mismatch between its two branches is the only reason it narrows.
    #[test]
    fn a_stage_reading_two_differently_widened_branches_narrows_to_one() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);
        let narrow = temporal("narrow", "input", 0);
        let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
        merge.inputs = vec![testing::positional("wide"), testing::positional("narrow")];
        merge.args = Spanned::unknown(json!({ "radius": 0 }));

        let plan = plan_of(&[source("input"), wide, narrow, merge, sink("out", "merge")]);

        assert_eq!(plan.node(NodeId::new(3)).width(), 1);
        assert_eq!(plan.node(NodeId::new(3)).narrowed(), Some(Narrowed::MixedInputs));
    }

    #[test]
    fn a_stream_node_with_disagreeing_input_widths_reports_granularity() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);
        let mut narrow = temporal("narrow", "input", 0);
        narrow.workers = Some(2);
        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("wide"), testing::positional("narrow")];

        let plan = plan_of(&[source("input"), wide, narrow, merge, sink("out", "merge")]);

        assert_eq!(plan.node(NodeId::new(3)).width(), 1);
        assert_eq!(plan.node(NodeId::new(3)).narrowed(), Some(Narrowed::Granularity));
    }

    #[test]
    fn a_chain_of_equal_width_nodes_is_one_region() {
        let mut first = temporal("first", "input", 0);
        first.workers = Some(4);
        let second = temporal("second", "first", 0);

        let plan = plan_of(&[source("input"), first, second, sink("out", "second")]);

        let first_region = plan.node(NodeId::new(1)).region();
        assert_eq!(plan.node(NodeId::new(2)).region(), first_region);
        assert_eq!(
            plan.region(first_region).nodes(),
            &[NodeId::new(1), NodeId::new(2)]
        );
    }

    #[test]
    fn a_change_of_width_starts_a_new_region() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);

        let plan = plan_of(&[source("input"), wide, sink("out", "wide")]);

        assert_ne!(
            plan.node(NodeId::new(0)).region(),
            plan.node(NodeId::new(1)).region()
        );
        assert_ne!(
            plan.node(NodeId::new(1)).region(),
            plan.node(NodeId::new(2)).region()
        );
    }

    /// A fan out and fan in below one widened stage runs entirely in one worker.
    #[test]
    fn a_diamond_below_one_widened_stage_stays_in_one_region() {
        let mut spread = temporal("spread", "input", 0);
        spread.workers = Some(4);
        let left = temporal("left", "spread", 0);
        let right = temporal("right", "spread", 0);
        let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
        merge.inputs = vec![testing::positional("left"), testing::positional("right")];
        merge.args = Spanned::unknown(json!({ "radius": 0 }));

        let plan = plan_of(&[source("input"), spread, left, right, merge, sink("out", "merge")]);

        let region = plan.node(NodeId::new(1)).region();
        assert_eq!(plan.node(NodeId::new(2)).region(), region);
        assert_eq!(plan.node(NodeId::new(3)).region(), region);
        assert_eq!(plan.node(NodeId::new(4)).region(), region);
        assert_eq!(plan.region(region).head(), NodeId::new(1));
    }

    #[test]
    fn a_wide_region_has_both_exchanges() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);

        let plan = plan_of(&[source("input"), wide, sink("out", "wide")]);

        let region = plan.region(plan.node(NodeId::new(1)).region());
        assert!(region.has_disperse());
        assert!(region.has_aggregate());
    }

    /// One input and one worker means there is nothing to route and nothing to pair.
    #[test]
    fn a_narrow_single_input_region_has_neither_exchange() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);

        let plan = plan_of(&[source("input"), wide, sink("out", "wide")]);

        let region = plan.region(plan.node(NodeId::new(2)).region());
        assert!(!region.has_disperse());
        assert!(!region.has_aggregate());
    }

    /// Pairing two inputs is reason enough for a disperse even at one worker.
    ///
    /// Two sources sit in two regions, so the stage joining them cannot be in either.
    #[test]
    fn a_narrow_region_joining_two_inputs_still_has_a_disperse() {
        let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
        merge.inputs = vec![testing::positional("left"), testing::positional("right")];

        let plan = plan_of(&[source("left"), source("right"), merge, sink("out", "merge")]);

        let region = plan.region(plan.node(NodeId::new(2)).region());
        assert_eq!(region.width(), 1);
        assert!(region.has_disperse());
        assert!(!region.has_aggregate());
    }

    /// A source has nothing to route, so widening it alone is no reason for a disperse.
    #[test]
    fn a_widened_source_has_an_aggregate_but_no_disperse() {
        let mut wide_source = testing::spec(StageKind::Source, "input", "test/widening-source");
        wide_source.workers = Some(4);

        let plan = plan_of(&[wide_source, sink("out", "input")]);

        let region = plan.region(plan.node(NodeId::new(0)).region());
        assert!(region.has_aggregate());
        assert!(!region.has_disperse());
    }

    #[test]
    fn a_region_carries_its_head_nodes_chunk_authority() {
        let mut spatial = temporal("spatial", "input", 0);
        spatial.workers = Some(4);

        let plan = plan_of(&[source("input"), spatial, sink("out", "spatial")]);

        let region = plan.region(plan.node(NodeId::new(1)).region());
        assert_eq!(region.head(), NodeId::new(1));
        assert_eq!(region.chunk_authority(), Some(0));
    }

    /// `test/temporal-merge` declares `[base, overlay]`. Writing them the other way round by
    /// name puts edge order and signature order at odds, so a chunk authority read by edge
    /// position rather than by slot would route by `overlay`'s chunks instead of `base`'s.
    #[test]
    fn a_chunk_authority_survives_inputs_written_out_of_signature_order() {
        let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
        merge.inputs = vec![testing::named("overlay", "right"), testing::named("base", "left")];
        merge.args = Spanned::unknown(json!({ "radius": 0 }));

        let (graph, plan) =
            graph_and_plan_of(&[source("left"), source("right"), merge, sink("out", "merge")]);

        let region = plan.region(plan.node(NodeId::new(2)).region());
        let authority = region.chunk_authority().expect("an authority");
        let routed = graph
            .node(NodeId::new(2))
            .inputs()
            .iter()
            .find(|edge| edge.slot() == authority)
            .expect("an edge fills the authority slot");

        assert_eq!(routed.param(), Some("base"));
    }

    /// `test/gather` has an optional param ahead of its variadic tail. Binding only the tail
    /// leaves the param empty, so the authority (the tail's slot) lands past the end of the one
    /// edge the stage has, the exact shape that indexes `Inputs()` out of range rather than
    /// looking an edge up by the slot it actually fills.
    #[test]
    fn a_chunk_authority_past_the_last_edge_is_found_by_slot_not_index() {
        let mut wide = temporal("wide", "input", 0);
        wide.workers = Some(4);

        let mut gather = testing::spec(StageKind::Mask, "gather", "test/gather");
        gather.inputs = vec![testing::named("extra", "wide")];

        let (graph, plan) = graph_and_plan_of(&[source("input"), wide, gather, sink("out", "gather")]);

        let region = plan.region(plan.node(NodeId::new(2)).region());
        let authority = region.chunk_authority().expect("an authority");
        let node = graph.node(NodeId::new(2));

        assert!(
            authority >= node.inputs().len(),
            "the authority should sit past the node's one edge"
        );

        let routed = node
            .inputs()
            .iter()
            .find(|edge| edge.slot() == authority)
            .expect("an edge fills the authority slot");

        assert_eq!(routed.param(), None);
    }

    #[test]
    fn a_graph_carries_its_plan_once_it_has_one() {
        let registry = testing::registry();
        let specs = vec![source("input"), sink("out", "input")];
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        assert!(graph.execution_plan().is_none());

        let made = plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made);

        assert!(graph.execution_plan().is_some());
    }
}
