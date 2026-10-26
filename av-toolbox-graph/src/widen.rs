use av_toolbox_core::StageKind;
use av_toolbox_gpu::{ResolvedFormats, SampleFormat, StreamFormat};
use serde_json::{Value, json};

use crate::SolvedFormats;
use crate::graph::{CompiledGraph, InjectedStage, NodeId};

/// The stage the graph injects to change a stream's depth.
pub(crate) const DEPTH_STAGE: &str = "resample/depth";

/// The name of [DEPTH_STAGE]'s one input.
const DEPTH_INPUT: &str = "input";

/// One depth stage to inject, with every edge that reads it.
struct Planned {
    producer: NodeId,
    from: StreamFormat,
    to: StreamFormat,
    readers: Vec<(NodeId, usize)>,
}

/// Inserts a `resample/depth` on every edge the solver widened.
///
/// Readers widening one producer to the same sample share a single stage. An edge that already
/// carries its wider format is left alone, so inserting a second time adds nothing.
pub fn insert_widens(graph: &mut CompiledGraph, formats: &mut SolvedFormats) {
    debug_assert_eq!(formats.len(), graph.len(), "every node has solved formats");

    let planned = plan_stages(graph, formats);
    if planned.is_empty() {
        return;
    }

    for stage in &planned {
        let name = stage_name(graph, &planned, stage);
        let depth = depth_arg(stage.to.sample());
        let injected = InjectedStage {
            kind: StageKind::Filter,
            identifier: DEPTH_STAGE,
            input: DEPTH_INPUT,
            name,
            args: json!({ "depth": depth }),
            span: graph.node(stage.producer).span(),
        };

        let node = graph.push_stage(stage.producer, injected);
        formats.push(ResolvedFormats::new(vec![Some(stage.from)], Some(stage.to)));

        for &(consumer, edge) in &stage.readers {
            graph.redirect(consumer, edge, node);
        }
    }

    graph.finish_edit();
}

/// Groups every widened edge still reading its producer by that producer and the wider sample.
fn plan_stages(graph: &CompiledGraph, formats: &SolvedFormats) -> Vec<Planned> {
    let mut planned: Vec<Planned> = Vec::new();

    for widen in formats.widens() {
        let producer = graph.node(widen.consumer).inputs()[widen.edge].source();
        if formats.output(producer) == Some(widen.to) {
            continue;
        }

        let reader = (widen.consumer, widen.edge);
        let existing = planned
            .iter_mut()
            .find(|stage| stage.producer == producer && stage.to.sample() == widen.to.sample());

        match existing {
            Some(stage) => stage.readers.push(reader),
            None => planned.push(Planned {
                producer,
                from: widen.from,
                to: widen.to,
                readers: vec![reader],
            }),
        }
    }

    planned
}

/// The node's name, such as `depth(denoise)`.
///
/// A producer widened to more than one sample names each stage's target too, as `depth(denoise, f32)`.
fn stage_name(graph: &CompiledGraph, planned: &[Planned], stage: &Planned) -> String {
    let producer = graph.node(stage.producer).name();
    let targets = planned
        .iter()
        .filter(|other| other.producer == stage.producer)
        .count();

    if targets == 1 {
        return format!("depth({producer})");
    }

    let depth = depth_arg(stage.to.sample());
    match depth {
        Value::String(target) => format!("depth({producer}, {target})"),
        bits => format!("depth({producer}, {bits})"),
    }
}

/// The `depth` argument naming `sample`, its bits or `"f32"`.
fn depth_arg(sample: SampleFormat) -> Value {
    match sample {
        SampleFormat::U8 => json!(8),
        SampleFormat::U16 { depth } => json!(depth.bits()),
        SampleFormat::F32 => json!("f32"),
    }
}
