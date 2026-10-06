use av_toolbox_core::{AnyDemand, Crossing, DemandDefinition, DemandKind, Granularity, Residency, StageKind};
use av_toolbox_gpu::LayerId;
use serde_json::json;

use crate::graph::{CompiledGraph, NodeId};
use crate::plan::Routing;
use crate::render::diagram::{
    Block,
    Diagram,
    DiagramBox,
    Endpoint,
    Exchange,
    ExchangeKind,
    Link,
    ShownDemand,
};
use crate::{ProviderPlacement, ServedLayer, Spanned, StageSpec, build, testing};

pub(crate) fn built(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let built = build(&registry, specs).expect("should build");

    built.into_graph()
}

pub(crate) fn planned(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = built(specs);
    let made = crate::plan::plan(&graph, &registry).expect("should plan");

    graph.set_execution_plan(made);
    graph
}

/// Builds `specs` and places a provider for every demand its stages make.
pub(crate) fn satisfied(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = built(specs);
    crate::satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");

    graph
}

/// The same as [satisfied], planned afterwards.
pub(crate) fn satisfied_planned(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = satisfied(specs);
    let made = crate::plan::plan(&graph, &registry).expect("should plan");

    graph.set_execution_plan(made);
    graph
}

pub(crate) fn node_named(graph: &CompiledGraph, name: &str) -> NodeId {
    let found = graph.nodes().iter().find(|node| node.name() == name);

    found.expect("the node is in the graph").id()
}

fn source(name: &str, path: &str) -> StageSpec {
    let mut source = testing::spec(StageKind::Source, name, "test/file");
    source.args = Spanned::unknown(json!({ "path": path }));
    source
}

fn temporal(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = testing::spec(StageKind::Filter, name, "test/temporal");
    filter.inputs = vec![testing::positional(input)];
    filter.args = Spanned::unknown(json!({ "radius": 4 }));
    filter.workers = workers;
    filter
}

fn blur(name: &str, input: &str) -> StageSpec {
    let mut blur = testing::spec(StageKind::Filter, name, "test/blur");
    blur.inputs = vec![testing::positional(input)];
    blur.args = Spanned::unknown(json!({ "sigma": 1.0 }));
    blur
}

fn stdout(name: &str, input: &str) -> StageSpec {
    let mut sink = testing::spec(StageKind::Sink, name, "test/stdout");
    sink.inputs = vec![testing::positional(input)];
    sink
}

fn demanding(name: &str, input: &str, every: u32) -> StageSpec {
    let mut filter = testing::spec(StageKind::Filter, name, "test/demanding");
    filter.inputs = vec![testing::positional(input)];
    filter.args = Spanned::unknown(json!({ "every": every }));
    filter
}

pub(crate) fn source_to_sink() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let out = stdout("out", "input");

    vec![input, out]
}

/// A source and a sink stay one wide, and a `test/temporal` filter between them widens.
pub(crate) fn wide_pipeline(workers: usize) -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let denoise = temporal("denoise", "input", Some(workers));
    let out = stdout("out", "denoise");

    vec![input, denoise, out]
}

/// `test/widening-source` is frame mode, so widening it is the one way to route by frame.
pub(crate) fn frame_pipeline() -> Vec<StageSpec> {
    let mut input = testing::spec(StageKind::Source, "input", "test/widening-source");
    input.workers = Some(4);
    let out = stdout("out", "input");

    vec![input, out]
}

/// A stage explicitly pinned to one worker, so its reason is `Declared`.
pub(crate) fn declared_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let pinned = temporal("pinned", "input", Some(1));
    let out = stdout("out", "pinned");

    vec![input, pinned, out]
}

/// One branch widens and one does not, and both feed a stage that reads both.
pub(crate) fn mixed_inputs_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let wide = temporal("wide", "input", Some(4));
    let narrow = temporal("narrow", "input", None);

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::positional("wide"), testing::positional("narrow")];
    merge.args = Spanned::unknown(json!({ "radius": 0 }));

    let out = stdout("out", "merge");

    vec![input, wide, narrow, merge, out]
}

/// Two sources read by one mask through its named `base` and `overlay` inputs.
pub(crate) fn named_inputs_pipeline() -> Vec<StageSpec> {
    let first = source("first", "a.mkv");
    let second = source("second", "b.mkv");

    let mut matched = testing::spec(StageKind::Mask, "matched", "test/matched");
    matched.inputs = vec![
        testing::named("base", "first"),
        testing::named("overlay", "second"),
    ];

    let out = stdout("out", "matched");

    vec![first, second, matched, out]
}

/// A chain of stages each widened to a different width, so every one gets its own region.
pub(crate) fn many_region_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let first = temporal("first", "input", Some(4));
    let second = temporal("second", "first", Some(2));
    let third = temporal("third", "second", Some(8));
    let out = stdout("out", "third");

    vec![input, first, second, third, out]
}

/// A widened stage feeding two branches that each end at their own sink.
pub(crate) fn two_exit_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let wide = temporal("wide", "input", Some(4));
    let left = temporal("left", "wide", None);
    let right = temporal("right", "wide", None);
    let out_left = stdout("out_left", "left");
    let out_right = stdout("out_right", "right");

    vec![input, wide, left, right, out_left, out_right]
}

/// A source read directly by a mask, and also through a filter, so one edge skips a column.
pub(crate) fn skipping_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let blurred = blur("blur", "input");

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
    merge.inputs = vec![testing::positional("blur"), testing::positional("input")];

    let out = stdout("out", "merge");

    vec![input, blurred, merge, out]
}

/// Two sources whose consumers are declared in the opposite order, which crosses the edges.
pub(crate) fn crossed_pipeline() -> Vec<StageSpec> {
    let first_source = source("first_source", "a.mkv");
    let second_source = source("second_source", "b.mkv");
    let first_blur = blur("first_blur", "second_source");
    let second_blur = blur("second_blur", "first_source");
    let first_out = stdout("first_out", "first_blur");
    let second_out = stdout("second_out", "second_blur");

    vec![
        first_source,
        second_source,
        first_blur,
        second_blur,
        first_out,
        second_out,
    ]
}

/// `first` is read by a later region as well as by the stages after it in its own region, so
/// that region's exit sits in its first column.
pub(crate) fn middle_exit_pipeline() -> Vec<StageSpec> {
    let first = source("first", "a.mkv");
    let second = source("second", "b.mkv");
    let blurred = blur("blur", "first");
    let out = stdout("out", "blur");

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/merge");
    merge.inputs = vec![testing::positional("first"), testing::positional("second")];

    let merged_out = stdout("merged_out", "merge");

    vec![first, second, blurred, out, merge, merged_out]
}

/// One source read by a stage chunking on its cuts, and by two stages each demanding breaks.
pub(crate) fn demanding_pipeline() -> Vec<StageSpec> {
    let input = source("input", "a.mkv");
    let denoise = temporal("denoise", "input", None);
    let first = demanding("first", "input", 4);
    let second = demanding("second", "input", 8);
    let denoise_out = stdout("denoise_out", "denoise");
    let first_out = stdout("first_out", "first");
    let second_out = stdout("second_out", "second");

    vec![input, denoise, first, second, denoise_out, first_out, second_out]
}

/// Two sources, each read by its own stage demanding breaks.
pub(crate) fn two_source_demanding_pipeline() -> Vec<StageSpec> {
    let first = source("first", "a.mkv");
    let second = source("second", "b.mkv");
    let first_breaks = demanding("first_breaks", "first", 4);
    let second_breaks = demanding("second_breaks", "second", 8);
    let first_out = stdout("first_out", "first_breaks");
    let second_out = stdout("second_out", "second_breaks");

    vec![first, second, first_breaks, second_breaks, first_out, second_out]
}

/// A host source feeding a device blur feeding a host sink, with both crossings placed.
pub(crate) fn placed_round_trip() -> CompiledGraph {
    let mut input = testing::spec(StageKind::Source, "input", "test/host-file");
    input.args = Spanned::unknown(json!({}));
    let blurred = blur("blur", "input");

    let mut out = testing::spec(StageKind::Sink, "out", "test/host-stdout");
    out.inputs = vec![testing::positional("blur")];

    let registry = testing::registry();
    let mut graph = built(&[input, blurred, out]);
    let capabilities = crate::residency::tests::capabilities();
    let mut formats = crate::solve(&graph, &registry, &capabilities).expect("should solve");
    crate::place_transfers(&mut graph, &registry, &mut formats).expect("should place");

    graph
}

pub(crate) fn planned_round_trip() -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = placed_round_trip();
    let made = crate::plan::plan(&graph, &registry).expect("should plan");

    graph.set_execution_plan(made);
    graph
}

/// The demand kind the hand built example's provider serves.
struct ExampleScenes;

impl DemandDefinition for ExampleScenes {
    type Request = ();
    type Output = ();

    fn kind() -> &'static str {
        "scenes"
    }
}

/// One stage of the hand built example, before it becomes a box.
struct ExampleStage {
    name: &'static str,
    identifier: &'static str,
    kind: StageKind,
    residency: Residency,
    crossing: Option<Crossing>,
    provider: Option<ProviderPlacement>,
    demands: Vec<ShownDemand>,
    block: usize,
}

fn example_stage(
    name: &'static str,
    identifier: &'static str,
    kind: StageKind,
    residency: Residency,
    block: usize,
) -> ExampleStage {
    ExampleStage {
        name,
        identifier,
        kind,
        residency,
        crossing: None,
        provider: None,
        demands: Vec::new(),
        block,
    }
}

/// A stage chunking on the example's scene layer `layer`, described as `description`.
fn example_chunking(
    name: &'static str,
    identifier: &'static str,
    kind: StageKind,
    residency: Residency,
    block: usize,
    layer: (u32, &str),
) -> ExampleStage {
    let (layer, description) = layer;
    let demand = ShownDemand {
        input: String::from("clip"),
        description: format!("scenes: {description}"),
        layer: Some(LayerId(layer)),
        chunks: true,
    };

    ExampleStage {
        demands: vec![demand],
        ..example_stage(name, identifier, kind, residency, block)
    }
}

/// The example's scene provider behind its source, serving each layer as `(description, demanded by)`.
fn example_provider(name: &'static str, block: usize, layers: &[(&str, &str)]) -> ExampleStage {
    let layers = layers
        .iter()
        .enumerate()
        .map(|(index, (description, demanded_by))| ServedLayer {
            layer: LayerId(index as u32),
            demand: AnyDemand::new::<ExampleScenes>(()),
            description: description.to_string(),
            demanded_by: vec![demanded_by.to_string()],
        })
        .collect();
    let placement = ProviderPlacement {
        kind: DemandKind::of::<ExampleScenes>(),
        node: NodeId::new(0),
        layers,
    };

    ExampleStage {
        provider: Some(placement),
        ..example_stage(
            name,
            "provider/scenes",
            StageKind::Provider,
            Residency::Host,
            block,
        )
    }
}

fn example_crossing(name: &'static str, crossing: Crossing, block: usize) -> ExampleStage {
    ExampleStage {
        name,
        identifier: crossing.identifier(),
        kind: StageKind::Crossing,
        residency: crossing.to_side(),
        crossing: Some(crossing),
        provider: None,
        demands: Vec::new(),
        block,
    }
}

fn example_block(
    id: usize,
    boxes: &[usize],
    width: usize,
    routing: Option<Routing>,
    reason: Option<&'static str>,
    layer: Option<&str>,
) -> Block {
    let granularity = match width {
        1 => Granularity::Stream,
        _ => Granularity::Chunk,
    };
    let boxes = boxes.iter().map(|index| NodeId::new(*index)).collect();

    Block {
        id,
        boxes,
        width,
        granularity: Some(granularity),
        routing,
        reason,
        layer: layer.map(|layer| format!("scenes: {layer}")),
    }
}

fn example_link(from: Endpoint, to: Endpoint, param: Option<&'static str>, bounded: Option<bool>) -> Link {
    Link {
        from,
        to,
        param,
        bounded,
    }
}

fn example_node(index: usize) -> Endpoint {
    let node = NodeId::new(index);
    Endpoint::Box(node)
}

/// A realistic planned pipeline, built by hand.
///
/// Region 0 decodes, finds scenes for the denoise and the encode, and uploads. Region 1 runs eight
/// workers on the gpu, where a denoise feeds a deband and an edge mask and a blend reads all three.
/// Region 2 dithers, region 3 encodes on six workers, region 4 muxes, and region 5 scores
/// the output against the source.
pub(crate) fn example_diagram() -> Diagram {
    let stages = vec![
        example_stage("source", "ffms2/video", StageKind::Source, Residency::Host, 0),
        example_provider(
            "scenes(source)",
            0,
            &[("natural", "denoise"), ("limited ≤10 s", "encode")],
        ),
        example_crossing("upload(scenes)", Crossing::Upload, 0),
        example_chunking(
            "denoise",
            "nlmeans/temporal",
            StageKind::Filter,
            Residency::Device,
            1,
            (0, "natural"),
        ),
        example_stage("deband", "f3kdb/deband", StageKind::Filter, Residency::Device, 1),
        example_stage("edges", "mask/sobel", StageKind::Mask, Residency::Device, 1),
        example_stage("merge", "mask/blend", StageKind::Mask, Residency::Device, 1),
        example_crossing("unload(merge)", Crossing::Unload, 1),
        example_stage("dither", "dither/ordered", StageKind::Filter, Residency::Host, 2),
        example_chunking(
            "encode",
            "svt-av1/encode",
            StageKind::Filter,
            Residency::Host,
            3,
            (1, "limited ≤10 s"),
        ),
        example_stage("mux", "matroska/write", StageKind::Sink, Residency::Host, 4),
        example_stage("quality", "vmaf/report", StageKind::Sink, Residency::Host, 5),
    ];

    let blocks = vec![
        example_block(0, &[0, 1, 2], 1, None, None, None),
        example_block(
            1,
            &[3, 4, 5, 6, 7],
            8,
            Some(Routing::ByChunk),
            None,
            Some("natural"),
        ),
        example_block(2, &[8], 1, None, Some("granularity"), None),
        example_block(3, &[9], 6, Some(Routing::ByChunk), None, Some("limited ≤10 s")),
        example_block(4, &[10], 1, None, Some("granularity"), None),
        example_block(5, &[11], 1, None, Some("granularity"), None),
    ];

    let boxes = stages
        .into_iter()
        .enumerate()
        .map(|(index, stage)| DiagramBox {
            node: NodeId::new(index),
            block: stage.block,
            name: stage.name.to_string(),
            identifier: stage.identifier.to_string(),
            kind: stage.kind,
            residency: Some(stage.residency),
            crossing: stage.crossing,
            width: blocks[stage.block].width,
            reason: None,
            provider: stage.provider,
            demands: stage.demands,
        })
        .collect();

    let exchanges = vec![
        Exchange {
            block: 1,
            kind: ExchangeKind::Disperse { slot: Some(0) },
            width: 8,
        },
        Exchange {
            block: 1,
            kind: ExchangeKind::Aggregate,
            width: 8,
        },
        Exchange {
            block: 3,
            kind: ExchangeKind::Disperse { slot: Some(0) },
            width: 6,
        },
        Exchange {
            block: 3,
            kind: ExchangeKind::Aggregate,
            width: 6,
        },
        Exchange {
            block: 5,
            kind: ExchangeKind::Disperse { slot: Some(0) },
            width: 1,
        },
    ];

    let local = None;
    let unbounded = Some(false);
    let bounded = Some(true);
    let links = vec![
        example_link(example_node(0), example_node(1), Some("clip"), local),
        example_link(example_node(1), example_node(2), None, local),
        example_link(example_node(2), Endpoint::Disperse(1), Some("clip"), unbounded),
        example_link(example_node(3), example_node(4), Some("clip"), local),
        example_link(example_node(3), example_node(5), Some("clip"), local),
        example_link(example_node(4), example_node(6), Some("base"), local),
        example_link(example_node(3), example_node(6), Some("overlay"), local),
        example_link(example_node(5), example_node(6), Some("mask"), local),
        example_link(example_node(6), example_node(7), None, local),
        example_link(Endpoint::Aggregate(1), example_node(8), Some("clip"), unbounded),
        example_link(example_node(8), Endpoint::Disperse(3), Some("clip"), unbounded),
        example_link(Endpoint::Aggregate(3), example_node(10), Some("video"), unbounded),
        example_link(
            example_node(0),
            Endpoint::Disperse(5),
            Some("reference"),
            unbounded,
        ),
        example_link(
            Endpoint::Aggregate(1),
            Endpoint::Disperse(5),
            Some("distorted"),
            unbounded,
        ),
        example_link(Endpoint::Disperse(1), example_node(3), None, unbounded),
        example_link(example_node(7), Endpoint::Aggregate(1), None, bounded),
        example_link(Endpoint::Disperse(3), example_node(9), None, unbounded),
        example_link(example_node(9), Endpoint::Aggregate(3), None, bounded),
        example_link(Endpoint::Disperse(5), example_node(11), None, unbounded),
    ];

    Diagram {
        blocks,
        boxes,
        exchanges,
        links,
        planned: true,
    }
}
