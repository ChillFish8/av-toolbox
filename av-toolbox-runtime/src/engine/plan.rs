use std::sync::Arc;
use std::time::Duration;

use av_toolbox_core::StageKind;
use av_toolbox_gpu::FrameBudget;
use av_toolbox_graph::{
    CompiledGraph,
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    NodeId,
    NodeRole,
    Region,
    Routing,
    SolvedFormats,
};

use crate::worker::{Feed, RegionEdges};

/// Builds one region's internal wiring from the graph it came from.
///
/// Every node after the head reads only from producers already inside the region, so each of its
/// edges names a producer in [Region::nodes] and the slot it fills.
pub(super) fn build_region_edges(graph: &CompiledGraph, region: &Region) -> RegionEdges {
    let nodes = region.nodes();
    let feeds = nodes[1..]
        .iter()
        .map(|node_id| {
            graph
                .node(*node_id)
                .inputs()
                .iter()
                .map(|edge| Feed {
                    producer: edge.source(),
                    slot: edge.slot(),
                })
                .collect()
        })
        .collect();

    let paths = nodes
        .iter()
        .map(|node_id| graph.node(*node_id))
        .filter(|node| node.role() == NodeRole::Stage)
        .map(|node| {
            let path = node.path();
            (node.id(), Arc::from(path))
        })
        .collect();

    RegionEdges::new(feeds, region.exits().to_vec()).with_paths(paths)
}

/// Which of a head's input slots are bound, sized to the widest slot any edge names.
pub(super) fn bound_slots(graph: &CompiledGraph, head: NodeId) -> Vec<bool> {
    let edges = graph.node(head).inputs();
    let width = edges.iter().map(|edge| edge.slot() + 1).max().unwrap_or(0);
    let mut bound = vec![false; width];

    for edge in edges {
        bound[edge.slot()] = true;
    }

    bound
}

pub(super) fn is_source_region(graph: &CompiledGraph, region: &Region) -> bool {
    graph.node(region.head()).kind() == StageKind::Source
}

/// Rejects the region shapes this engine cannot run, reporting each against the node that named
/// it rather than silently doing something other than what was asked.
///
/// A widened source has nothing to route its own output through, so nothing here can shard it
/// across workers without inventing an unspecified policy, and quietly running it at width one
/// would look exactly like the parallelism doing nothing. A region with more than one exit
/// needs one aggregate per exit to keep each of its output streams in order on its own, which
/// this engine does not build, so broadcasting every exit's output to every consumer would
/// silently duplicate frames instead. The last two shapes are ones the wiring below cannot make
/// sense of at all: a region that reads nothing and produces nothing, and a chunk routed disperse
/// whose authority names a slot no input was bound to. A frame routed disperse reads no chunk id,
/// so an authority it never looks at is left alone.
pub(super) fn region_is_supported(
    graph: &CompiledGraph,
    region: &Region,
    diagnostics: &mut Diagnostics,
) -> bool {
    let mut supported = true;

    if !region.has_disperse()
        && !is_source_region(graph, region)
        && graph.node(region.head()).inputs().is_empty()
    {
        let head = graph.node(region.head());
        let message = format!(
            "the region headed by {} is not a source and reads no input, so nothing would ever run it",
            head.name(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    let routes_by_chunk = region.routing() == Routing::ByChunk;
    if routes_by_chunk && region.has_disperse() && !authority_is_bound(graph, region) {
        let head = graph.node(region.head());
        let message = format!(
            "{} routes its chunks by an input slot nothing was bound to",
            head.name(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidInputs,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    if is_source_region(graph, region) && region.width() > 1 {
        let node = graph.node(region.head());
        let message = format!(
            "{} asked for {} workers, but a source cannot yet be split across more than one",
            node.name(),
            region.width(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            format!("{}.{}", node.kind(), node.name()),
            node.span(),
            message,
        ));
        supported = false;
    }

    if region.exits().len() > 1 {
        let head = graph.node(region.head());
        let message = format!(
            "the region headed by {} has {} exits, and only one is supported per region",
            head.name(),
            region.exits().len(),
        );

        diagnostics.push(Diagnostic::error(
            DiagnosticCode::InvalidWorkers,
            format!("{}.{}", head.kind(), head.name()),
            head.span(),
            message,
        ));
        supported = false;
    }

    supported
}

/// Whether a region's chunk authority names one of the head's bound input slots.
///
/// [Disperse](crate::Disperse) routes every bundle by the item in that slot, so a slot outside
/// the head's own inputs, or one no edge fills, leaves it with nothing to read a chunk id from.
fn authority_is_bound(graph: &CompiledGraph, region: &Region) -> bool {
    let Some(authority) = region.chunk_authority() else {
        return false;
    };

    bound_slots(graph, region.head())
        .get(authority)
        .copied()
        .unwrap_or(false)
}

/// The largest stream item the solver resolved anywhere in the plan, in bytes.
///
/// Masks count as well as frames, because a stage allocating one charges it to the same budget a
/// frame is charged to, and a mask is not always the smaller of the two.
fn largest_item_bytes(formats: &SolvedFormats) -> usize {
    let mut largest = 0;

    for index in 0..formats.len() {
        let resolved = formats.get(NodeId::new(index));

        for input in resolved.inputs().iter().flatten() {
            largest = largest.max(input.total_bytes());
        }

        if let Some(output) = resolved.output() {
            largest = largest.max(output.total_bytes());
        }
    }

    largest
}

/// How many items a source may hold outstanding on top of the ones its own region holds.
pub(crate) const HEADROOM_ITEMS_PER_WORKER: usize = 4;

/// How many units a frame routed worker may hold that its aggregate has not drained.
///
/// Matched to the budget's per worker headroom, so a frame routed region at full stretch holds
/// what the budget was sized for rather than something it never counted.
pub(crate) const FRAME_UNITS_IN_FLIGHT: usize = HEADROOM_ITEMS_PER_WORKER;

/// How many nodes the longest source region runs.
///
/// A source region's nodes all share the source's one budgeted context, and
/// [RegionRunner::run](crate::RegionRunner::run) holds every one of their outputs until it
/// returns, so this is how many items that one share has to fit at once.
fn longest_source_region(graph: &CompiledGraph, regions: &[Region]) -> usize {
    regions
        .iter()
        .filter(|region| is_source_region(graph, region))
        .map(|region| region.nodes().len())
        .max()
        .unwrap_or(0)
}

/// How many frames every node in the plan may hold back at once, counting one window per worker.
fn held_by_stages(regions: &[Region], held: &[usize]) -> usize {
    let mut total = 0;

    for region in regions {
        for node in region.nodes() {
            total += held[node.index()] * region.width();
        }
    }

    total
}

/// The frame memory one source may hold outstanding at once, derived from the plan.
///
/// It covers three things. One item per node in the longest source region, so a region holding
/// every one of its outputs still fits. Every frame a stage may hold back, once per worker, so
/// a stage filling its window and a join waiting on it both fit. And four items per worker on
/// top, so a source can run ahead of the workers reading from it rather than waiting on each
/// item to come all the way back.
pub(super) fn default_source_share(
    graph: &CompiledGraph,
    formats: &SolvedFormats,
    regions: &[Region],
    held: &[usize],
) -> usize {
    let total_workers: usize = regions.iter().map(Region::width).sum();
    let region_holds = longest_source_region(graph, regions);
    let stages_hold = held_by_stages(regions, held);
    let headroom = HEADROOM_ITEMS_PER_WORKER * total_workers;
    let held_at_once = region_holds + stages_hold;

    (held_at_once + headroom) * largest_item_bytes(formats)
}

/// One source's budget, holding `share` bytes and giving up on a wait that lasts `deadline`.
pub(super) fn source_budget(share: usize, deadline: Option<Duration>) -> Arc<FrameBudget> {
    match deadline {
        Some(deadline) => FrameBudget::with_stall_deadline(share, deadline),
        None => FrameBudget::new(share),
    }
}
