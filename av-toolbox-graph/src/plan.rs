use av_toolbox_core::{AnyDemand, Granularity, PipelineRegistry, PreparedStage, Residency};
use av_toolbox_gpu::LayerId;

use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, NodeId, NodeRole};
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
    /// Its stage suggested one.
    Suggested,
    /// Its inputs come from regions of differing widths.
    MixedInputs,
    /// Nothing asked for more.
    NotAsked,
}

/// How a region's disperse picks a worker for each bundle.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Routing {
    /// By the authority input's chunk id, so one worker holds a whole chunk.
    ByChunk,
    /// By whichever worker has room, one bundle at a time.
    ByFrame,
}

/// How one node will be run.
#[derive(Debug, Clone, PartialEq)]
pub struct NodePlan {
    granularity: Granularity,
    demand: Option<AnyDemand>,
    width: usize,
    region: RegionId,
    narrowed: Option<Narrowed>,
    routing: Routing,
    chunk_authority: Option<usize>,
    layer: Option<LayerId>,
}

impl NodePlan {
    /// A node whose stage never prepared, since `prepare_stages` already reported why.
    fn fallback() -> Self {
        Self {
            granularity: Granularity::Stream,
            demand: None,
            width: 1,
            region: RegionId::new(0),
            narrowed: Some(Narrowed::NotAsked),
            routing: Routing::ByChunk,
            chunk_authority: None,
            layer: None,
        }
    }

    pub fn granularity(&self) -> Granularity {
        self.granularity
    }

    /// The demand a chunk node is chunked by.
    pub fn demand(&self) -> Option<&AnyDemand> {
        self.demand.as_ref()
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

    pub fn routing(&self) -> Routing {
        self.routing
    }

    /// The input slot this node reads its chunk from, if its stage names one.
    pub fn chunk_authority(&self) -> Option<usize> {
        self.chunk_authority
    }

    /// The scene layer this node routes and restarts its chunks on.
    pub fn layer(&self) -> Option<LayerId> {
        self.layer
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
    layer: Option<LayerId>,
    routing: Routing,
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

    /// The scene layer the region routes its chunks on, taken from the first of its nodes that has one.
    pub fn layer(&self) -> Option<LayerId> {
        self.layer
    }

    pub fn routing(&self) -> Routing {
        self.routing
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
/// matches, and heads a region of its own otherwise. A node with a scene layer also needs the
/// region to route on that layer, or on none yet. A crossing only joins when it is its producer's only consumer.
///
/// A region may have at most one exit. When one has more, a node at its branch is sealed, so
/// nothing reading it may join its region, and the regions are placed again.
pub fn plan(graph: &CompiledGraph, registry: &PipelineRegistry) -> Result<ExecutionPlan, Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);

    let mut planner = Planner {
        graph,
        prepared,
        nodes: (0..graph.len()).map(|_| None).collect(),
        regions: Vec::new(),
        sealed: vec![false; graph.len()],
        diagnostics,
    };

    for id in graph.order() {
        planner.check_side_settled(*id);
        planner.decide(*id);
    }

    // A region with more than one exit always yields a node not sealed before, so the loop ends.
    loop {
        planner.place_regions();
        planner.compute_exits();

        let sealed_any = planner.seal_regions_with_many_exits();
        if !sealed_any {
            break;
        }
    }

    debug_assert!(
        planner.regions.iter().all(|region| region.exits.len() <= 1),
        "every region has at most one exit once sealing ends",
    );

    planner.compute_routing();

    planner.finish()
}

/// The state one planning pass carries from node to node.
struct Planner<'a> {
    graph: &'a CompiledGraph,
    prepared: Vec<Option<PreparedStage>>,
    nodes: Vec<Option<NodePlan>>,
    regions: Vec<Region>,
    sealed: Vec<bool>,
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
        let granularity = parallelism.granularity();

        if granularity == Granularity::Stream && node.workers().is_some_and(|workers| workers > 1) {
            self.report_whole_stream(id);
            self.nodes[id.index()] = Some(NodePlan::fallback());
            return;
        }

        let suggested = stage.suggested_workers().filter(|workers| *workers > 0);
        let (candidate, reason) = match (node.workers(), suggested) {
            (Some(workers), _) => asked_width(workers, Narrowed::Declared),
            (None, Some(workers)) => asked_width(workers, Narrowed::Suggested),
            (None, None) => self.candidate_width(id),
        };
        let (width, narrowed) = clamp_to_granularity(granularity, candidate, reason);

        self.nodes[id.index()] = Some(NodePlan {
            granularity,
            demand: parallelism.demand().cloned(),
            width,
            region: RegionId::new(0),
            narrowed,
            routing: Routing::ByChunk,
            chunk_authority: stage.chunk_authority(),
            layer: node.layer(),
        });
    }

    /// The width a node would run at before its own granularity can narrow it back down.
    ///
    /// Every input's already-decided width is compared, and the node takes that width when they
    /// all agree, or one when they do not.
    fn candidate_width(&self, id: NodeId) -> (usize, Option<Narrowed>) {
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

    /// Places every node into a region from scratch, in topological order.
    fn place_regions(&mut self) {
        self.regions.clear();

        let graph = self.graph;
        for id in graph.order() {
            self.place_in_region(*id);
        }
    }

    /// Places one node into a region, following a producer's region or opening a new one.
    fn place_in_region(&mut self, id: NodeId) {
        let width = self.width_of(id);
        let node = self.graph.node(id);
        let sources: Vec<NodeId> = node.inputs().iter().map(|edge| edge.source()).collect();

        if matches!(node.role(), NodeRole::Crossing(_)) && !self.is_only_consumer(id, &sources) {
            self.open_region(id, width);
            return;
        }

        match self.shared_producer_region(&sources, width, node.layer()) {
            Some(region) => self.join_region(id, region),
            None => {
                self.open_region(id, width);
            },
        }
    }

    fn is_only_consumer(&self, id: NodeId, sources: &[NodeId]) -> bool {
        sources
            .iter()
            .all(|source| self.graph.node(*source).consumers() == [id])
    }

    /// The one region every input already shares, when its width matches this node's own.
    ///
    /// A sealed input never counts, so a node reading one heads a region of its own. A node with a
    /// layer only joins a region routed on that same layer, or on none yet.
    fn shared_producer_region(
        &self,
        sources: &[NodeId],
        width: usize,
        layer: Option<LayerId>,
    ) -> Option<RegionId> {
        let first_source = *sources.first()?;

        let any_sealed = sources.iter().any(|source| self.sealed[source.index()]);
        if any_sealed {
            return None;
        }

        let candidate = self.region_of(first_source);

        let all_agree = sources.iter().all(|source| self.region_of(*source) == candidate);
        if !all_agree {
            return None;
        }

        let region = &self.regions[candidate.index()];
        let same_layer = layer.is_none() || region.layer.is_none() || layer == region.layer;

        (region.width == width && same_layer).then_some(candidate)
    }

    fn region_of(&self, id: NodeId) -> RegionId {
        self.nodes[id.index()]
            .as_ref()
            .map(NodePlan::region)
            .expect("a producer is placed into a region before its consumer, since order is topological")
    }

    /// Adds a node to an existing region, which routes on the node's layer when it had none.
    fn join_region(&mut self, id: NodeId, region: RegionId) {
        let joined = &mut self.regions[region.index()];
        joined.nodes.push(id);

        if joined.layer.is_none() {
            joined.layer = self.graph.node(id).layer();
        }

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
            layer: self.graph.node(head).layer(),
            routing: Routing::ByChunk,
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

    /// Seals one branch in every region with more than one exit, returning whether any node was
    /// newly sealed.
    fn seal_regions_with_many_exits(&mut self) -> bool {
        let mut to_seal = Vec::new();

        for region in &self.regions {
            if region.exits.len() > 1 {
                let branch = self.branch_of(region);
                to_seal.extend(branch);
            }
        }

        let mut sealed_any = false;
        for id in to_seal {
            let already_sealed = self.sealed[id.index()];
            self.sealed[id.index()] = true;
            sealed_any |= !already_sealed;
        }

        sealed_any
    }

    /// The nodes to seal so a region's earliest exit stops sharing the region with another.
    ///
    /// The earliest exit is sealed itself when another member reads it. Otherwise its producers
    /// inside the region are sealed, since the region branches there instead.
    fn branch_of(&self, region: &Region) -> Vec<NodeId> {
        let earliest_exit = region.exits[0];
        let node = self.graph.node(earliest_exit);

        let read_inside = node
            .consumers()
            .iter()
            .any(|consumer| self.region_of(*consumer) == region.id);
        if read_inside {
            return vec![earliest_exit];
        }

        node.inputs()
            .iter()
            .map(|edge| edge.source())
            .filter(|source| self.region_of(*source) == region.id)
            .collect()
    }

    /// Settles how each region routes, once every node has joined one.
    ///
    /// A region routes by frame only when every node in it is frame mode, since its nodes run
    /// inside one worker as a unit and the most demanding of them decides what that unit needs.
    fn compute_routing(&mut self) {
        let mut modes = Vec::with_capacity(self.regions.len());

        for region in &self.regions {
            let greatest = region
                .nodes
                .iter()
                .filter_map(|id| self.nodes[id.index()].as_ref())
                .map(NodePlan::granularity)
                .max();

            let mode = match greatest {
                Some(Granularity::Frame) => Routing::ByFrame,
                _ => Routing::ByChunk,
            };

            modes.push(mode);
        }

        for (region, mode) in self.regions.iter_mut().zip(modes) {
            region.routing = mode;

            for &id in &region.nodes {
                if let Some(node) = self.nodes[id.index()].as_mut() {
                    node.routing = mode;
                }
            }
        }
    }

    /// Reports a stage that runs off the device but has no side settled yet.
    fn check_side_settled(&mut self, id: NodeId) {
        let Some(stage) = self.prepared[id.index()].as_ref() else {
            return;
        };

        let node = self.graph.node(id);
        if stage.residency() == Residency::Device || node.residency().is_some() {
            return;
        }

        let message = format!(
            "{} does not run on the gpu, so transfers must be placed before planning",
            node.name()
        );

        self.diagnostics.push(Diagnostic::error(
            DiagnosticCode::TransfersNotPlaced,
            stage_path(node.kind(), node.name()),
            node.span(),
            message,
        ));
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

/// A width the pipeline or the stage asked for, labelled with `reason` when it is one.
fn asked_width(workers: usize, reason: Narrowed) -> (usize, Option<Narrowed>) {
    if workers == 1 {
        return (workers, Some(reason));
    }

    (workers, None)
}

/// Applies a node's granularity to the width it would otherwise run at.
///
/// A node whose granularity is the whole stream always ends up one wide, and the reason recorded
/// then names the granularity rather than whatever reason the pre-clamp width carried, since a
/// stage that only ever takes the whole stream could never have widened at any candidate width.
/// The one exception is a candidate that was never asked to widen at all, since a source really
/// is one wide because nothing asked.
fn clamp_to_granularity(
    granularity: Granularity,
    candidate: usize,
    reason: Option<Narrowed>,
) -> (usize, Option<Narrowed>) {
    if granularity != Granularity::Stream {
        let narrowed = if candidate == 1 { reason } else { None };
        return (candidate, narrowed);
    }

    if reason == Some(Narrowed::NotAsked) {
        return (1, reason);
    }

    (1, Some(Narrowed::Granularity))
}
