use av_toolbox_core::{InputParam, PipelineRegistry, Residency};
use av_toolbox_gpu::{FrameLayout, MaskChroma, StreamFormat, Subsampling};

use crate::diagnostic::{input_path, stage_path};
use crate::graph::{CompiledGraph, Edge, Node, NodeId};
use crate::solve::signature_of;
use crate::{Diagnostic, DiagnosticCode, Diagnostics, SolvedFormats, Span};

/// Gives each mask the chroma plane its readers ask for with `planes_of`.
///
/// A reader whose named frame input is YUV 4:2:0 or 4:2:2 needs a chroma plane of that shape, sited
/// where that frame sites its chroma. The need lands on the stage producing the mask, and its
/// output and every input reading it then carry the plane. Readers needing different planes, and a
/// need reaching a host mask, are errors.
pub fn settle_mask_planes(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    formats: &mut SolvedFormats,
) -> Result<(), Diagnostics> {
    let mut diagnostics = Diagnostics::new();
    let prepared = graph.prepare_stages(registry, &mut diagnostics);
    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    let needs = collect_needs(graph, registry, formats);

    for &producer in graph.order() {
        let wanted = &needs[producer.index()];
        let Some(first) = wanted.first() else {
            continue;
        };

        let conflicting = wanted.iter().find(|need| need.chroma != first.chroma);
        if let Some(other) = conflicting {
            report_conflict(graph, producer, wanted, other, &mut diagnostics);
            continue;
        }

        let residency = prepared[producer.index()].as_ref().map(|stage| stage.residency());
        if residency == Some(Residency::Host) {
            report_host_mask(graph, producer, first, &mut diagnostics);
            continue;
        }

        give_chroma(graph, formats, producer, first.chroma);
    }

    if diagnostics.has_errors() {
        return Err(diagnostics);
    }

    Ok(())
}

/// One reader asking the stage producing its mask for a chroma plane.
#[derive(Debug, Copy, Clone)]
struct Need {
    reader: NodeId,
    /// Where the mask input sits in the reader's definition, which is what a diagnostic names.
    edge: usize,
    chroma: MaskChroma,
}

/// Every need in the graph, grouped by the node producing the mask it is for.
fn collect_needs(
    graph: &CompiledGraph,
    registry: &PipelineRegistry,
    formats: &SolvedFormats,
) -> Vec<Vec<Need>> {
    let mut needs: Vec<Vec<Need>> = vec![Vec::new(); graph.len()];

    for &reader in graph.order() {
        let node = graph.node(reader);
        let Some(signature) = signature_of(registry, node) else {
            continue;
        };

        for param in signature.params() {
            let Some((producer, need)) = need_of(node, param, formats) else {
                continue;
            };

            needs[producer.index()].push(need);
        }
    }

    needs
}

/// What one mask input asks of its producer, if it asks for anything at all.
fn need_of(node: &Node, param: &InputParam, formats: &SolvedFormats) -> Option<(NodeId, Need)> {
    let frames_name = param.planes_of_input()?;
    let mask_name = param.name();
    let (edge, mask_input) = input_named(node, mask_name)?;
    let (_, frames_input) = input_named(node, frames_name)?;

    let reader = node.id();
    let frames_slot = frames_input.slot();
    let resolved = formats.get(reader);
    let frames = resolved.input(frames_slot)?;
    let chroma = chroma_for(frames)?;

    let need = Need { reader, edge, chroma };
    let producer = mask_input.source();

    Some((producer, need))
}

/// The input bound to the signature param `name`, with its position in the definition.
fn input_named<'a>(node: &'a Node, name: &str) -> Option<(usize, &'a Edge)> {
    node.inputs()
        .iter()
        .enumerate()
        .find(|(_, edge)| edge.param() == Some(name))
}

/// The chroma plane a mask needs to be blended with `format`, which is none unless its chroma is
/// subsampled.
fn chroma_for(format: StreamFormat) -> Option<MaskChroma> {
    let StreamFormat::Frames(frames) = format else {
        return None;
    };

    match frames.layout {
        FrameLayout::Yuv {
            subsampling: subsampling @ (Subsampling::Yuv420 | Subsampling::Yuv422),
        } => Some(MaskChroma {
            subsampling,
            location: frames.colour.chroma_location,
        }),
        _ => None,
    }
}

/// Sets the chroma plane on what `producer` emits and on every input reading it.
fn give_chroma(graph: &CompiledGraph, formats: &mut SolvedFormats, producer: NodeId, chroma: MaskChroma) {
    let Some(StreamFormat::Mask(mut mask)) = formats.output(producer) else {
        return;
    };

    mask.chroma = Some(chroma);
    let carried = StreamFormat::Mask(mask);
    formats.set_output(producer, carried);

    for &consumer in graph.node(producer).consumers() {
        let reading = graph
            .node(consumer)
            .inputs()
            .iter()
            .filter(|edge| edge.source() == producer);
        for edge in reading {
            let slot = edge.slot();
            formats.set_input(consumer, slot, carried);
        }
    }
}

/// Names the first reader and one that disagrees with it, with a note for every reader.
fn report_conflict(
    graph: &CompiledGraph,
    producer: NodeId,
    wanted: &[Need],
    other: &Need,
    diagnostics: &mut Diagnostics,
) {
    let producing = graph.node(producer);
    let name = producing.name();
    let first = &wanted[0];
    let first_path = need_path(graph, first);
    let other_path = need_path(graph, other);
    let first_shape = describe(first.chroma);
    let other_shape = describe(other.chroma);

    let message = format!(
        "{name} is read by {first_path}, which needs {first_shape}, and by {other_path}, which needs \
         {other_shape}. A mask carries one chroma plane, so every reader asking for one has to read \
         frames with the same subsampling and chroma location.",
    );

    let kind = producing.kind();
    let path = stage_path(kind, name);
    let span = producing.span();
    let mut diagnostic = Diagnostic::error(DiagnosticCode::ConflictingMaskPlanes, path, span, message);

    for need in wanted {
        let reader_path = need_path(graph, need);
        let span = need_span(graph, need);
        let shape = describe(need.chroma);
        let note = format!("{reader_path} asks for {shape}");

        diagnostic = diagnostic.with_note(note, span);
    }

    diagnostics.push(diagnostic);
}

fn report_host_mask(graph: &CompiledGraph, producer: NodeId, need: &Need, diagnostics: &mut Diagnostics) {
    let producing = graph.node(producer);
    let name = producing.name();
    let path = need_path(graph, need);
    let shape = describe(need.chroma);

    let message = format!(
        "{path} needs {shape} from {name}, which runs on the host, and host masks cannot carry \
         chroma planes. Use a mask stage that runs on the gpu.",
    );

    let need_span = need_span(graph, need);
    let note = format!("{path} asks for it here");

    let kind = producing.kind();
    let producer_path = stage_path(kind, name);
    let producer_span = producing.span();
    let diagnostic = Diagnostic::error(
        DiagnosticCode::HostMaskPlanes,
        producer_path,
        producer_span,
        message,
    );
    let noted = diagnostic.with_note(note, need_span);
    diagnostics.push(noted);
}

fn need_path(graph: &CompiledGraph, need: &Need) -> String {
    let reader = graph.node(need.reader);
    let kind = reader.kind();
    let name = reader.name();

    input_path(kind, name, need.edge)
}

fn need_span(graph: &CompiledGraph, need: &Need) -> Span {
    let reader = graph.node(need.reader);

    reader.inputs()[need.edge].span()
}

/// Spells out a chroma plane for a diagnostic, such as `a 4:2:0 chroma plane sited Left`.
fn describe(chroma: MaskChroma) -> String {
    format!(
        "a {} chroma plane sited {:?}",
        chroma.subsampling, chroma.location
    )
}

#[cfg(test)]
mod tests {
    use av_toolbox_core::{Crossing, StageKind};
    use av_toolbox_gpu::ChromaLocation;
    use serde_json::json;

    use super::*;
    use crate::graph::NodeRole;
    use crate::residency::tests::capabilities;
    use crate::{Spanned, StageSpec, build, place_transfers, solve, testing};

    fn source(name: &str, layout: &str, location: &str) -> StageSpec {
        let mut spec = testing::spec(StageKind::Source, name, "test/layout-file");
        spec.args = Spanned::unknown(json!({ "layout": layout, "location": location }));

        spec
    }

    fn stage(kind: StageKind, name: &str, stage_type: &str, inputs: &[&str]) -> StageSpec {
        let mut spec = testing::spec(kind, name, stage_type);
        spec.inputs = inputs.iter().map(|input| testing::positional(input)).collect();

        spec
    }

    fn blend(name: &str, stage_type: &str, frames: &str, mask: &str) -> StageSpec {
        stage(StageKind::Filter, name, stage_type, &[frames, frames, mask])
    }

    fn sink(name: &str, from: &str) -> StageSpec {
        stage(StageKind::Sink, name, "test/stdout", &[from])
    }

    fn settled(specs: &[StageSpec]) -> Result<(CompiledGraph, SolvedFormats), Diagnostics> {
        let registry = testing::registry();
        let graph = build(&registry, specs).expect("should build").into_graph();
        let capabilities = capabilities();
        let mut formats = solve(&graph, &registry, &capabilities).expect("should solve");

        settle_mask_planes(&graph, &registry, &mut formats)?;

        Ok((graph, formats))
    }

    fn id_of(graph: &CompiledGraph, name: &str) -> NodeId {
        let node = graph
            .nodes()
            .iter()
            .find(|node| node.name() == name)
            .expect("a node with that name");

        node.id()
    }

    fn chroma_of(format: Option<StreamFormat>) -> Option<MaskChroma> {
        match format {
            Some(StreamFormat::Mask(mask)) => mask.chroma,
            other => panic!("expected a mask, got {other:?}"),
        }
    }

    fn output_chroma(graph: &CompiledGraph, formats: &SolvedFormats, name: &str) -> Option<MaskChroma> {
        let id = id_of(graph, name);
        let output = formats.output(id);

        chroma_of(output)
    }

    fn input_chroma(
        graph: &CompiledGraph,
        formats: &SolvedFormats,
        name: &str,
        slot: usize,
    ) -> Option<MaskChroma> {
        let id = id_of(graph, name);
        let input = formats.get(id).input(slot);

        chroma_of(input)
    }

    fn two_readers_over(layout: &str, location: &str) -> Vec<StageSpec> {
        vec![
            source("input", layout, location),
            stage(StageKind::Mask, "edges", "test/merge", &["input"]),
            blend("first", "test/blend", "input", "edges"),
            blend("second", "test/blend", "input", "edges"),
            sink("a", "first"),
            sink("b", "second"),
        ]
    }

    #[test]
    fn two_420_readers_of_one_mask_give_it_one_chroma_plane() {
        let specs = two_readers_over("yuv420", "left");

        let (graph, formats) = settled(&specs).expect("should settle");

        let expected = Some(MaskChroma {
            subsampling: Subsampling::Yuv420,
            location: ChromaLocation::Left,
        });
        assert_eq!(output_chroma(&graph, &formats, "edges"), expected);
        assert_eq!(input_chroma(&graph, &formats, "first", 2), expected);
        assert_eq!(input_chroma(&graph, &formats, "second", 2), expected);
    }

    #[test]
    fn a_444_reader_adds_no_chroma() {
        let specs = two_readers_over("yuv444", "left");

        let (graph, formats) = settled(&specs).expect("should settle");

        assert_eq!(output_chroma(&graph, &formats, "edges"), None);
        assert_eq!(input_chroma(&graph, &formats, "first", 2), None);
    }

    #[test]
    fn rgb_and_grey_readers_add_no_chroma() {
        for layout in ["rgb", "grey"] {
            let specs = two_readers_over(layout, "left");

            let (graph, formats) = settled(&specs).expect("should settle");

            assert_eq!(output_chroma(&graph, &formats, "edges"), None, "{layout}");
        }
    }

    #[test]
    fn a_422_reader_gives_a_422_chroma_plane() {
        let specs = two_readers_over("yuv422", "left");

        let (graph, formats) = settled(&specs).expect("should settle");

        let chroma = output_chroma(&graph, &formats, "edges").expect("a chroma plane");
        assert_eq!(chroma.subsampling, Subsampling::Yuv422);
    }

    #[test]
    fn the_chroma_location_follows_the_frame() {
        let specs = two_readers_over("yuv420", "center");

        let (graph, formats) = settled(&specs).expect("should settle");

        let chroma = output_chroma(&graph, &formats, "edges").expect("a chroma plane");
        assert_eq!(chroma.location, ChromaLocation::Center);
    }

    #[test]
    fn demand_stops_at_the_producer() {
        let specs = [
            source("input", "yuv420", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["input"]),
            stage(StageKind::Mask, "combined", "test/combine", &["edges"]),
            blend("first", "test/blend", "input", "combined"),
            sink("a", "first"),
        ];

        let (graph, formats) = settled(&specs).expect("should settle");

        let combined = output_chroma(&graph, &formats, "combined");
        let read = input_chroma(&graph, &formats, "first", 2);
        assert!(combined.is_some());
        assert_eq!(read, combined);
        assert_eq!(output_chroma(&graph, &formats, "edges"), None);
        assert_eq!(input_chroma(&graph, &formats, "combined", 0), None);
    }

    #[test]
    fn every_reader_of_a_chroma_mask_receives_the_plane() {
        let specs = [
            source("input", "yuv420", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["input"]),
            blend("first", "test/blend", "input", "edges"),
            blend("plain", "test/plain-blend", "input", "edges"),
            sink("a", "first"),
            sink("b", "plain"),
        ];

        let (graph, formats) = settled(&specs).expect("should settle");

        let expected = output_chroma(&graph, &formats, "edges");
        assert!(expected.is_some());
        assert_eq!(input_chroma(&graph, &formats, "plain", 2), expected);
    }

    #[test]
    fn readers_needing_different_shapes_are_refused() {
        let specs = [
            source("left", "yuv420", "left"),
            source("right", "yuv422", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["left"]),
            blend("first", "test/blend", "left", "edges"),
            blend("second", "test/blend", "right", "edges"),
            sink("a", "first"),
            sink("b", "second"),
        ];

        let diagnostics = settled(&specs).expect_err("the readers disagree");

        let refusal = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code() == DiagnosticCode::ConflictingMaskPlanes)
            .expect("a conflicting mask planes error");
        let message = refusal.message();
        assert_eq!(refusal.path(), "mask.edges");
        assert!(message.contains("filter.first.inputs[2]"), "{message}");
        assert!(message.contains("filter.second.inputs[2]"), "{message}");
        assert!(message.contains("4:2:0"), "{message}");
        assert!(message.contains("4:2:2"), "{message}");
        assert_eq!(refusal.notes().len(), 2, "one note per reader");
    }

    /// Binds the mask first and the base last, so each input's definition position differs from
    /// its signature slot.
    fn out_of_order_blend(name: &str, base: &str, overlay: &str, mask: &str) -> StageSpec {
        let mut spec = testing::spec(StageKind::Filter, name, "test/blend");
        spec.inputs = vec![
            testing::named("mask", mask),
            testing::named("overlay", overlay),
            testing::named("base", base),
        ];

        spec
    }

    #[test]
    fn a_mask_bound_out_of_order_is_read_and_written_at_its_slot() {
        let specs = [
            source("base", "yuv420", "left"),
            source("overlay", "yuv444", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["base"]),
            out_of_order_blend("first", "base", "overlay", "edges"),
            sink("a", "first"),
        ];

        let (graph, formats) = settled(&specs).expect("should settle");

        let expected = Some(MaskChroma {
            subsampling: Subsampling::Yuv420,
            location: ChromaLocation::Left,
        });
        let first = id_of(&graph, "first");
        let base_slot = formats.get(first).input(0);
        assert_eq!(output_chroma(&graph, &formats, "edges"), expected);
        assert_eq!(input_chroma(&graph, &formats, "first", 2), expected);
        assert!(matches!(base_slot, Some(StreamFormat::Frames(_))));
    }

    #[test]
    fn a_conflict_names_a_mask_by_its_definition_position() {
        let specs = [
            source("left", "yuv420", "left"),
            source("right", "yuv422", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["left"]),
            out_of_order_blend("first", "left", "left", "edges"),
            blend("second", "test/blend", "right", "edges"),
            sink("a", "first"),
            sink("b", "second"),
        ];

        let diagnostics = settled(&specs).expect_err("the readers disagree");

        let refusal = diagnostics
            .iter()
            .find(|diagnostic| diagnostic.code() == DiagnosticCode::ConflictingMaskPlanes)
            .expect("a conflicting mask planes error");
        let message = refusal.message();
        assert!(message.contains("filter.first.inputs[0]"), "{message}");
        assert!(message.contains("filter.second.inputs[2]"), "{message}");
    }

    #[test]
    fn a_host_mask_producer_is_refused() {
        let specs = [
            source("input", "yuv420", "left"),
            stage(StageKind::Mask, "edges", "test/host-masker", &["input"]),
            blend("first", "test/blend", "input", "edges"),
            sink("a", "first"),
        ];

        let diagnostics = settled(&specs).expect_err("a host mask cannot carry chroma");

        let codes: Vec<DiagnosticCode> = diagnostics.iter().map(|diagnostic| diagnostic.code()).collect();
        assert_eq!(codes, [DiagnosticCode::HostMaskPlanes]);
    }

    #[test]
    fn a_reader_without_planes_of_needs_nothing() {
        let specs = [
            source("input", "yuv420", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["input"]),
            blend("plain", "test/plain-blend", "input", "edges"),
            sink("a", "plain"),
        ];

        let (graph, formats) = settled(&specs).expect("should settle");

        assert_eq!(output_chroma(&graph, &formats, "edges"), None);
    }

    #[test]
    fn a_crossing_after_a_chroma_mask_keeps_it() {
        let specs = [
            source("input", "yuv420", "left"),
            stage(StageKind::Mask, "edges", "test/merge", &["input"]),
            blend("first", "test/blend", "input", "edges"),
            sink("a", "first"),
            stage(StageKind::Sink, "mask", "test/host-stdout", &["edges"]),
        ];
        let registry = testing::registry();
        let (mut graph, mut formats) = settled(&specs).expect("should settle");

        place_transfers(&mut graph, &registry, &mut formats).expect("should place");

        let unload = graph
            .nodes()
            .iter()
            .find(|node| *node.role() == NodeRole::Crossing(Crossing::Unload))
            .expect("the mask is unloaded for its host sink");
        let carried = formats.output(unload.id());
        assert!(chroma_of(carried).is_some());
    }
}
