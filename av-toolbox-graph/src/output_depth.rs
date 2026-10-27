use av_toolbox_core::{PipelineRegistry, PreparedStage, StageKind, StreamType};
use av_toolbox_gpu::{Capabilities, SampleFormat};
use serde_json::json;

use crate::Diagnostics;
use crate::graph::{CompiledGraph, InjectedStage, NodeId, NodeRole};
use crate::widen::{DEPTH_INPUT, DEPTH_STAGE};

/// The least depth the graph writes out.
const OUTPUT_DEPTH: u8 = 10;

/// Puts a `resample/depth` in front of every sink that wants the output depth, unless the user wrote one there.
///
/// The conversion targets the deepest integer sample a source would rather emit, or 10 bits when that is
/// deeper. It allows any deeper integer sample too, and never narrows a stream already deeper than the target,
/// so the sink settles on the first one it can write.
pub fn inject_output_depth(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    capabilities: &Capabilities,
) -> Result<(), Diagnostics> {
    let wanting = unconverted_edges(graph, registry);
    if wanting.is_empty() {
        return Ok(());
    }

    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let target = target_depth(graph, &prepared, capabilities);

    for (sink, edge) in wanting {
        let sink_node = graph.node(sink);
        let producer = sink_node.inputs()[edge].source();
        let injected = InjectedStage {
            kind: StageKind::Filter,
            identifier: DEPTH_STAGE,
            input: DEPTH_INPUT,
            name: format!("depth({})", sink_node.name()),
            args: json!({ "depth": target, "at_least": true }),
            span: sink_node.span(),
        };

        let node = graph.push_stage(producer, injected);
        graph.redirect(sink, edge, node);
    }

    graph.finish_edit();

    Ok(())
}

/// Every frames edge into a sink that wants the output depth and is not already fed by a `resample/depth`.
///
/// A mask edge is left alone so the sink reports it.
fn unconverted_edges(graph: &CompiledGraph, registry: &PipelineRegistry) -> Vec<(NodeId, usize)> {
    let mut edges = Vec::new();

    let sinks = graph
        .nodes()
        .iter()
        .filter(|node| node.kind() == StageKind::Sink)
        .filter(|node| wants_output_depth(registry, node.kind(), node.identifier()));
    for sink in sinks {
        for (edge, input) in sink.inputs().iter().enumerate() {
            let source = input.source();
            let carries_mask = graph.node(source).output_stream() == Some(StreamType::Mask);
            if carries_mask || is_converted(graph, source) {
                continue;
            }

            edges.push((sink.id(), edge));
        }
    }

    edges
}

fn wants_output_depth(registry: &PipelineRegistry, kind: StageKind, identifier: &str) -> bool {
    let mut stages = registry.stages();
    let stage = stages.find(|info| info.kind() == kind && info.full_identifier() == identifier);

    stage.is_some_and(|info| info.wants_output_depth())
}

/// Whether `feeding` is a `resample/depth`, looking through any providers placed in front of it.
fn is_converted(graph: &CompiledGraph, feeding: NodeId) -> bool {
    let mut node = graph.node(feeding);
    while let NodeRole::Provider(_) = node.role() {
        let source = node.inputs()[0].source();
        node = graph.node(source);
    }

    node.kind() == StageKind::Filter && node.identifier() == DEPTH_STAGE
}

/// The deepest integer sample any source would rather emit, or [OUTPUT_DEPTH] when that is deeper.
fn target_depth(
    graph: &CompiledGraph,
    prepared: &[Option<PreparedStage>],
    capabilities: &Capabilities,
) -> u8 {
    let mut target = OUTPUT_DEPTH;

    let sources = graph
        .nodes()
        .iter()
        .filter(|node| node.kind() == StageKind::Source);
    for source in sources {
        let Some(stage) = &prepared[source.id().index()] else {
            continue;
        };

        let Ok(output) = stage.produces(capabilities, &[]) else {
            continue;
        };

        let preferred = output.samples.preferences().first().copied();
        let bits = preferred.and_then(integer_bits);
        if let Some(bits) = bits {
            target = target.max(bits);
        }
    }

    target
}

fn integer_bits(sample: SampleFormat) -> Option<u8> {
    match sample {
        SampleFormat::U8 => Some(8),
        SampleFormat::U16 { depth } => Some(depth.bits()),
        SampleFormat::F32 => None,
    }
}
