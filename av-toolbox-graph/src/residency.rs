use std::collections::HashMap;

use av_toolbox_core::{Crossing, PipelineRegistry, PreparedStage, Residency};
use av_toolbox_gpu::ResolvedFormats;

use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, SolvedFormats};

/// Settles which side every node runs on and inserts a crossing wherever a stream has to move.
///
/// Runs after the format solve and before planning. A crossing belongs to a producer's output, so
/// a producer read by several consumers on the other side gets one crossing they all read, and
/// same-side consumers keep reading it directly. Placing transfers a second time inserts nothing,
/// because every edge outside a crossing already joins two nodes on the same side, and a
/// crossing's own input is exempt: it crosses sides by design.
///
/// The `Ok` side carries warnings only. A stage that fails to prepare was already reported by the
/// solve, so it is reported again here as an error rather than guessed at.
pub fn place_transfers(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    formats: &mut SolvedFormats,
) -> Result<Diagnostics, Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let sides = settle_sides(graph, &prepared);
    for (index, side) in sides.iter().enumerate() {
        graph.set_residency(NodeId::new(index), *side);
    }

    insert_crossings(graph, &sides, formats);
    graph.finish_edit();

    warn_round_trips(graph, &mut diagnostics);

    Ok(diagnostics)
}

/// Every node's side, in topological order so an `Either` node can read its producers'.
fn settle_sides(graph: &CompiledGraph, prepared: &[Option<PreparedStage>]) -> Vec<Residency> {
    let mut sides = vec![Residency::Device; graph.len()];

    for &id in graph.order() {
        let node = graph.node(id);
        let side = match (node.role(), prepared[id.index()].as_ref()) {
            (NodeRole::Crossing(crossing), _) => crossing.to_side(),
            (NodeRole::Stage, Some(stage)) => match stage.residency() {
                Residency::Either => follow_inputs(node, stage.chunk_authority(), &sides),
                fixed => fixed,
            },
            (NodeRole::Stage, None) => Residency::Device,
        };

        sides[id.index()] = side;
    }

    sides
}

/// The side an `Either` node follows: its authority input's, or its first input's without one.
fn follow_inputs(node: &Node, authority: Option<usize>, sides: &[Residency]) -> Residency {
    let by_authority = authority.and_then(|slot| node.inputs().iter().find(|edge| edge.slot() == slot));
    let edge = by_authority.or_else(|| node.inputs().first());

    edge.map(|edge| sides[edge.source().index()])
        .unwrap_or(Residency::Device)
}

/// Inserts one crossing per producer with any consumer on the other side, and points those
/// consumers at it.
fn insert_crossings(graph: &mut CompiledGraph, sides: &[Residency], formats: &mut SolvedFormats) {
    let mut redirects: HashMap<NodeId, Vec<(NodeId, usize)>> = HashMap::new();

    for consumer in graph.nodes() {
        // A crossing's own input is definitionally on the other side from where it lands, since
        // that is the transfer it performs. Reading it as a mismatch would wrap every crossing in
        // another one, so a graph that already has its crossings would never stop growing them.
        if matches!(consumer.role(), NodeRole::Crossing(_)) {
            continue;
        }

        for (index, edge) in consumer.inputs().iter().enumerate() {
            let producer = edge.source();
            if sides[producer.index()] != sides[consumer.id().index()] {
                redirects
                    .entry(producer)
                    .or_default()
                    .push((consumer.id(), index));
            }
        }
    }

    let mut producers: Vec<NodeId> = redirects.keys().copied().collect();
    producers.sort();

    for producer in producers {
        let to = sides[redirects[&producer][0].0.index()];
        let crossing = Crossing::between(sides[producer.index()], to)
            .expect("a producer and consumer on different settled sides are host and device");

        let output = formats.output(producer);
        let node = graph.push_crossing(producer, crossing);
        formats.push(ResolvedFormats::new(vec![output], output));

        for &(consumer, edge) in &redirects[&producer] {
            graph.redirect(consumer, edge, node);
        }
    }
}

/// Warns about every stream that leaves the device only to be uploaded again.
///
/// Each upload is walked back through the host stages feeding it. Reaching an unload means the
/// stream was on the device, came off it for those stages, and is going back, which is two
/// transfers the pipeline could avoid by moving those stages or choosing device versions of them.
/// A stream that started on the host never reaches an unload, so a host source is never warned
/// about.
pub(crate) fn warn_round_trips(graph: &CompiledGraph, diagnostics: &mut Diagnostics) {
    for node in graph.nodes() {
        if node.role() != NodeRole::Crossing(Crossing::Upload) {
            continue;
        }

        let mut host_stages = Vec::new();
        let mut came_off_the_device = false;
        let mut pending = vec![node.inputs()[0].source()];
        let mut seen = std::collections::HashSet::new();

        while let Some(id) = pending.pop() {
            if !seen.insert(id) {
                continue;
            }

            let producer = graph.node(id);
            match producer.role() {
                NodeRole::Crossing(Crossing::Unload) => came_off_the_device = true,
                NodeRole::Crossing(Crossing::Upload) => {},
                NodeRole::Stage => {
                    host_stages.push(id);
                    pending.extend(producer.inputs().iter().map(|edge| edge.source()));
                },
            }
        }

        if !came_off_the_device {
            continue;
        }

        host_stages.sort_by_key(|id| graph.order().iter().position(|each| each == id));
        let names: Vec<&str> = host_stages.iter().map(|id| graph.node(*id).name()).collect();
        let last = graph.node(
            *host_stages
                .last()
                .expect("an upload after an unload has a host stage between"),
        );

        diagnostics.push(Diagnostic::warning(
            DiagnosticCode::ResidencyRoundTrip,
            stage_path(last.kind(), last.name()),
            last.span(),
            format!(
                "frames leave the gpu for {} and are uploaded again afterwards, which costs a transfer each way",
                names.join(", "),
            ),
        ));
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use av_toolbox_core::{Crossing, StageKind};
    use av_toolbox_gpu::{Accelerator, Capabilities, ElementType, default_accelerators};
    use serde_json::json;

    use super::*;
    use crate::graph::NodeRole;
    use crate::{DiagnosticCode, Severity, Spanned, StageSpec, build, solve, testing};

    pub(crate) fn capabilities() -> Capabilities {
        let every = [
            ElementType::F32,
            ElementType::F16,
            ElementType::U8,
            ElementType::U16,
        ];
        let accelerator: Accelerator = default_accelerators()[0];

        Capabilities::builder(accelerator)
            .storable(every)
            .computable(every)
            .plane_size(32, 32)
            .max_units_per_cube(1024)
            .max_cube_count(65535, 65535, 65535)
            .build()
    }

    fn stage(kind: StageKind, name: &str, stage_type: &str, inputs: &[&str]) -> StageSpec {
        let mut spec = testing::spec(kind, name, stage_type);
        spec.inputs = inputs.iter().map(|input| testing::positional(input)).collect();
        if stage_type == "test/file" {
            spec.args = Spanned::unknown(json!({ "path": "a.mkv" }));
        }
        if stage_type == "test/blur" {
            spec.args = Spanned::unknown(json!({ "sigma": 1.0 }));
        }

        spec
    }

    fn placed(specs: &[StageSpec]) -> (CompiledGraph, Diagnostics) {
        let registry = testing::registry();
        let mut graph = build(&registry, specs).expect("should build").into_graph();
        let mut formats = solve(&graph, &registry, &capabilities()).expect("should solve");

        let warnings = place_transfers(&mut graph, &registry, &mut formats).expect("should place");
        assert_eq!(formats.len(), graph.len(), "every inserted node has formats");

        (graph, warnings)
    }

    fn crossings(graph: &CompiledGraph, crossing: Crossing) -> Vec<&Node> {
        graph
            .nodes()
            .iter()
            .filter(|node| node.role() == NodeRole::Crossing(crossing))
            .collect()
    }

    fn named<'a>(graph: &'a CompiledGraph, name: &str) -> &'a Node {
        graph
            .nodes()
            .iter()
            .find(|node| node.name() == name)
            .expect("a node with that name")
    }

    #[test]
    fn a_graph_of_device_stages_gains_no_crossings() {
        let (graph, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Filter, "blur", "test/blur", &["input"]),
            stage(StageKind::Sink, "out", "test/stdout", &["blur"]),
        ]);

        assert_eq!(graph.len(), 3);
        assert!(warnings.is_empty());
        assert!(
            graph
                .nodes()
                .iter()
                .all(|node| node.residency() == Some(Residency::Device))
        );
    }

    #[test]
    fn two_host_sinks_reading_one_device_stream_share_one_unload() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Filter, "blur", "test/blur", &["input"]),
            stage(StageKind::Sink, "first", "test/host-stdout", &["blur"]),
            stage(StageKind::Sink, "second", "test/host-stdout", &["blur"]),
        ]);

        let unloads = crossings(&graph, Crossing::Unload);
        assert_eq!(unloads.len(), 1);
        assert_eq!(unloads[0].name(), "unload(blur)");
        assert_eq!(named(&graph, "first").inputs()[0].source(), unloads[0].id());
        assert_eq!(named(&graph, "second").inputs()[0].source(), unloads[0].id());
    }

    #[test]
    fn two_device_filters_reading_one_host_stream_share_one_upload() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Filter, "left", "test/blur", &["input"]),
            stage(StageKind::Filter, "right", "test/blur", &["input"]),
            stage(StageKind::Sink, "a", "test/stdout", &["left"]),
            stage(StageKind::Sink, "b", "test/stdout", &["right"]),
        ]);

        let uploads = crossings(&graph, Crossing::Upload);
        assert_eq!(uploads.len(), 1);
        assert_eq!(named(&graph, "left").inputs()[0].source(), uploads[0].id());
        assert_eq!(named(&graph, "right").inputs()[0].source(), uploads[0].id());
    }

    #[test]
    fn a_consumer_on_the_producers_own_side_keeps_reading_it_directly() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Sink, "gpu", "test/stdout", &["input"]),
            stage(StageKind::Sink, "host", "test/host-stdout", &["input"]),
        ]);

        let input = named(&graph, "input").id();
        assert_eq!(named(&graph, "gpu").inputs()[0].source(), input);
        assert_eq!(
            named(&graph, "host").inputs()[0].source(),
            named(&graph, "unload(input)").id()
        );
    }

    #[test]
    fn host_source_device_filter_host_sink_crosses_once_each_way_without_warning() {
        let (graph, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Filter, "blur", "test/blur", &["input"]),
            stage(StageKind::Sink, "out", "test/host-stdout", &["blur"]),
        ]);

        assert_eq!(crossings(&graph, Crossing::Upload).len(), 1);
        assert_eq!(crossings(&graph, Crossing::Unload).len(), 1);
        assert!(warnings.is_empty());
    }

    #[test]
    fn an_either_transform_follows_its_only_inputs_side() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Transform, "restamp", "test/restamp", &["input"]),
            stage(StageKind::Sink, "out", "test/host-stdout", &["restamp"]),
        ]);

        assert_eq!(named(&graph, "restamp").residency(), Some(Residency::Host));
        assert_eq!(graph.len(), 3);
    }

    #[test]
    fn an_either_transform_transfers_only_the_input_off_its_authoritys_side() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "host", "test/host-file", &[]),
            stage(StageKind::Source, "gpu", "test/file", &[]),
            stage(StageKind::Transform, "restamp", "test/restamp", &["host", "gpu"]),
            stage(StageKind::Sink, "out", "test/stdout", &["restamp"]),
        ]);

        let restamp = named(&graph, "restamp");
        assert_eq!(
            restamp.residency(),
            Some(Residency::Device),
            "the authority is the device input"
        );
        let uploads = crossings(&graph, Crossing::Upload);
        assert_eq!(uploads.len(), 1);
        assert_eq!(restamp.inputs()[0].source(), uploads[0].id());
        assert_eq!(restamp.inputs()[1].source(), named(&graph, "gpu").id());
    }

    #[test]
    fn a_mask_stream_crosses_as_a_mask() {
        let mut deband = stage(StageKind::Filter, "deband", "test/deband", &["input"]);
        deband.inputs.push(testing::named("overlay", "merge"));

        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Mask, "merge", "test/merge", &["input", "input"]),
            deband,
            stage(StageKind::Sink, "out", "test/host-stdout", &["deband"]),
            stage(StageKind::Sink, "mask", "test/host-stdout", &["merge"]),
        ]);

        let unload = crossings(&graph, Crossing::Unload)
            .into_iter()
            .find(|node| node.name() == "unload(merge)")
            .expect("the mask is unloaded for its host sink");
        assert_eq!(unload.output_stream(), Some(av_toolbox_core::StreamType::Mask));
    }

    #[test]
    fn placing_transfers_twice_inserts_nothing_the_second_time() {
        let registry = testing::registry();
        let specs = [
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Filter, "blur", "test/blur", &["input"]),
            stage(StageKind::Sink, "out", "test/host-stdout", &["blur"]),
        ];
        let mut graph = build(&registry, &specs).expect("should build").into_graph();
        let mut formats = solve(&graph, &registry, &capabilities()).expect("should solve");

        place_transfers(&mut graph, &registry, &mut formats).expect("should place");
        let after_first = graph.len();
        place_transfers(&mut graph, &registry, &mut formats).expect("should place again");

        assert_eq!(graph.len(), after_first);
    }

    #[test]
    fn leaving_the_device_and_coming_back_warns_once_naming_the_host_stage() {
        let (_, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Filter, "detect", "test/host-blur", &["input"]),
            stage(StageKind::Filter, "blur", "test/blur", &["detect"]),
            stage(StageKind::Sink, "out", "test/stdout", &["blur"]),
        ]);

        let round_trips: Vec<_> = warnings
            .iter()
            .filter(|diagnostic| diagnostic.code() == DiagnosticCode::ResidencyRoundTrip)
            .collect();

        assert_eq!(round_trips.len(), 1);
        assert_eq!(round_trips[0].severity(), Severity::Warning);
        assert!(round_trips[0].message().contains("detect"));
    }

    #[test]
    fn every_host_stage_in_a_round_trip_is_named() {
        let (_, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Filter, "first", "test/host-blur", &["input"]),
            stage(StageKind::Filter, "second", "test/host-blur", &["first"]),
            stage(StageKind::Filter, "blur", "test/blur", &["second"]),
            stage(StageKind::Sink, "out", "test/stdout", &["blur"]),
        ]);

        let message = warnings
            .iter()
            .find(|diagnostic| diagnostic.code() == DiagnosticCode::ResidencyRoundTrip)
            .expect("a round trip")
            .message()
            .to_string();

        assert!(message.contains("first"));
        assert!(message.contains("second"));
    }

    #[test]
    fn a_host_source_whose_stream_never_comes_back_is_not_a_round_trip() {
        let (_, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Filter, "detect", "test/host-blur", &["input"]),
            stage(StageKind::Filter, "blur", "test/blur", &["detect"]),
            stage(StageKind::Sink, "out", "test/host-stdout", &["blur"]),
        ]);

        assert!(warnings.is_empty());
    }
}
