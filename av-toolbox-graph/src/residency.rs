use std::collections::{HashMap, HashSet};

use av_toolbox_core::{Crossing, PipelineRegistry, PreparedStage, Residency};
use av_toolbox_gpu::ResolvedFormats;

use crate::diagnostic::stage_path;
use crate::graph::{CompiledGraph, Node, NodeId, NodeRole};
use crate::{Diagnostic, DiagnosticCode, Diagnostics, SolvedFormats};

/// Settles which side every node runs on and inserts a crossing wherever a stream has to move.
///
/// Placing transfers a second time inserts nothing, because every edge outside a crossing already
/// joins two nodes on the same side.
pub fn place_transfers(
    graph: &mut CompiledGraph,
    registry: &PipelineRegistry,
    formats: &mut SolvedFormats,
) -> Result<Diagnostics, Diagnostics> {
    debug_assert_eq!(formats.len(), graph.len(), "every node has solved formats");

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

/// The side an `Either` node follows, its authority input's side when there is one, or its first
/// input's otherwise.
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
pub(crate) fn warn_round_trips(graph: &CompiledGraph, diagnostics: &mut Diagnostics) {
    let off_the_device = streams_off_the_device(graph);

    for node in graph.nodes() {
        if node.role() != NodeRole::Crossing(Crossing::Upload) {
            continue;
        }

        let mut host_stages = Vec::new();
        let mut pending = vec![node.inputs()[0].source()];
        let mut seen = HashSet::new();

        while let Some(id) = pending.pop() {
            if !seen.insert(id) || !off_the_device[id.index()] {
                continue;
            }

            let producer = graph.node(id);
            if producer.role() == NodeRole::Stage {
                host_stages.push(id);
                pending.extend(producer.inputs().iter().map(|edge| edge.source()));
            }
        }

        host_stages.sort_by_key(|id| graph.order().iter().position(|each| each == id));
        let Some(last_id) = host_stages.last() else {
            continue;
        };

        let last = graph.node(*last_id);
        let names: Vec<&str> = host_stages.iter().map(|id| graph.node(*id).name()).collect();

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

/// Whether each node's output came off the device through an unload, with no upload since.
fn streams_off_the_device(graph: &CompiledGraph) -> Vec<bool> {
    let mut off_the_device = vec![false; graph.len()];

    for &id in graph.order() {
        let node = graph.node(id);
        off_the_device[id.index()] = match node.role() {
            NodeRole::Crossing(Crossing::Unload) => true,
            NodeRole::Crossing(Crossing::Upload) => false,
            NodeRole::Stage => node
                .inputs()
                .iter()
                .any(|edge| off_the_device[edge.source().index()]),
        };
    }

    off_the_device
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
    fn a_nodes_path_is_its_kind_and_name() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Transform, "scenes.detect", "test/restamp", &["input"]),
            stage(StageKind::Sink, "out", "test/host-stdout", &["scenes.detect"]),
        ]);

        assert_eq!(named(&graph, "scenes.detect").path(), "transform.scenes.detect");
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
    fn a_meta_sink_fed_by_a_device_stage_stays_on_the_device() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Sink, "log", "test/log", &["input"]),
        ]);

        assert_eq!(named(&graph, "log").residency(), Some(Residency::Device));
        assert!(crossings(&graph, Crossing::Unload).is_empty());
    }

    #[test]
    fn a_meta_sink_fed_by_a_host_stage_stays_on_the_host() {
        let (graph, _) = placed(&[
            stage(StageKind::Source, "input", "test/host-file", &[]),
            stage(StageKind::Sink, "log", "test/log", &["input"]),
        ]);

        assert_eq!(named(&graph, "log").residency(), Some(Residency::Host));
        assert!(crossings(&graph, Crossing::Upload).is_empty());
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

    #[test]
    fn a_source_opens_once_across_every_pass() {
        crate::testing::COUNTED_OPENS.store(0, std::sync::atomic::Ordering::SeqCst);
        let registry = testing::registry();
        let specs = [
            stage(StageKind::Source, "input", "test/counted", &[]),
            stage(StageKind::Sink, "out", "test/stdout", &["input"]),
        ];
        let mut graph = build(&registry, &specs).expect("should build").into_graph();

        let capabilities = capabilities();
        let mut formats = solve(&graph, &registry, &capabilities).expect("should solve");
        place_transfers(&mut graph, &registry, &mut formats).expect("should place");
        let made_plan = crate::plan::plan(&graph, &registry).expect("should plan");
        graph.set_execution_plan(made_plan);

        // Preparing again after solve, place and plan is exactly what building does. If that pass
        // opened through a fresh slot instead of the node's own, asking the prepared source what it
        // produces here would open the input a second time.
        let mut diagnostics = Diagnostics::new();
        let prepared = graph.prepare_stages(&registry, &mut diagnostics);
        let input = named(&graph, "input").id();
        let prepared_source = prepared
            .into_iter()
            .nth(input.index())
            .flatten()
            .expect("the source should have prepared");
        prepared_source
            .produces(&capabilities, &[])
            .expect("should produce again");

        assert_eq!(
            crate::testing::COUNTED_OPENS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn a_source_that_cannot_open_is_reported_on_its_node() {
        let registry = testing::registry();
        let specs = [
            stage(StageKind::Source, "input", "test/unopenable", &[]),
            stage(StageKind::Sink, "out", "test/stdout", &["input"]),
        ];
        let graph = build(&registry, &specs).expect("should build").into_graph();

        let diagnostics = solve(&graph, &registry, &capabilities()).expect_err("the open fails");

        let reported = diagnostics
            .iter()
            .any(|diagnostic| diagnostic.path().contains("input"));
        assert!(reported);
    }

    #[test]
    fn two_identical_sources_merged_by_deduplication_open_once() {
        crate::testing::COUNTED_OPENS.store(0, std::sync::atomic::Ordering::SeqCst);
        let registry = testing::registry();
        let specs = [
            stage(StageKind::Source, "first", "test/counted", &[]),
            stage(StageKind::Source, "second", "test/counted", &[]),
            stage(StageKind::Sink, "a", "test/stdout", &["first"]),
            stage(StageKind::Sink, "b", "test/stdout", &["second"]),
        ];
        let mut graph = build(&registry, &specs).expect("should build").into_graph();
        crate::optimise(&mut graph, &crate::default_passes());

        let capabilities = capabilities();
        solve(&graph, &registry, &capabilities).expect("should solve");

        // The same double check as the single-source test above, now against whichever of the two
        // duplicate nodes deduplication kept.
        let mut diagnostics = Diagnostics::new();
        let prepared = graph.prepare_stages(&registry, &mut diagnostics);
        let merged = graph
            .nodes()
            .iter()
            .find(|node| node.kind() == StageKind::Source)
            .expect("one source survives deduplication")
            .id();
        let prepared_source = prepared
            .into_iter()
            .nth(merged.index())
            .flatten()
            .expect("the surviving source should have prepared");
        prepared_source
            .produces(&capabilities, &[])
            .expect("should produce again");

        assert_eq!(
            crate::testing::COUNTED_OPENS.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[test]
    fn a_host_stage_that_never_left_the_device_is_not_named_in_a_round_trip() {
        let (_, warnings) = placed(&[
            stage(StageKind::Source, "input", "test/file", &[]),
            stage(StageKind::Source, "extra", "test/host-file", &[]),
            stage(StageKind::Filter, "detect", "test/host-blur", &["input"]),
            stage(
                StageKind::Transform,
                "restamp",
                "test/restamp",
                &["detect", "extra"],
            ),
            stage(StageKind::Filter, "blur", "test/blur", &["restamp"]),
            stage(StageKind::Sink, "out", "test/stdout", &["blur"]),
        ]);

        let message = warnings
            .iter()
            .find(|diagnostic| diagnostic.code() == DiagnosticCode::ResidencyRoundTrip)
            .expect("a round trip")
            .message()
            .to_string();

        assert!(message.contains("detect"));
        assert!(message.contains("restamp"));
        assert!(!message.contains("extra"));
    }
}
