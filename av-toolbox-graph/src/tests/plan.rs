use av_toolbox_core::testing::FixtureCuts;
use av_toolbox_core::{Granularity, Parallelism, StageKind};
use av_toolbox_gpu::{ChunkRoute, LayerId};
use serde_json::json;

use crate::graph::{CompiledGraph, Node, NodeId};
use crate::plan::{ExecutionPlan, Narrowed, Region, RegionId, Routing, plan};
use crate::{
    DiagnosticCode,
    Diagnostics,
    Severity,
    Spanned,
    StageSpec,
    build,
    place_transfers,
    residency,
    satisfy_demands,
    solve,
    testing,
};

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

fn placed_graph_and_plan_of(specs: &[StageSpec]) -> (CompiledGraph, ExecutionPlan) {
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let mut graph = build(&registry, specs).expect("should build").into_graph();
    let mut formats = solve(&graph, &registry, &capabilities).expect("should solve");
    place_transfers(&mut graph, &registry, &mut formats).expect("should place");
    let made = plan(&graph, &registry).expect("should plan");

    (graph, made)
}

fn node_named(graph: &CompiledGraph, name: &str) -> NodeId {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .map(Node::id)
        .expect("a node with that name")
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

    assert_eq!(plan.node(NodeId::new(1)).granularity(), Granularity::Frame);
    assert_eq!(plan.node(NodeId::new(1)).width(), 4);
    assert_eq!(plan.node(NodeId::new(1)).narrowed(), None);
}

#[test]
fn a_stage_suggestion_widens_it_when_nothing_else_does() {
    let mut suggesting = temporal("suggesting", "input", 0);
    suggesting.args = Spanned::unknown(json!({ "radius": 0, "workers": 2 }));

    let plan = plan_of(&[source("input"), suggesting, sink("out", "suggesting")]);

    assert_eq!(plan.node(NodeId::new(1)).width(), 2);
    assert_eq!(plan.node(NodeId::new(1)).narrowed(), None);
}

#[test]
fn a_suggested_width_of_one_is_labelled_as_suggested() {
    let mut suggesting = temporal("suggesting", "input", 0);
    suggesting.args = Spanned::unknown(json!({ "radius": 0, "workers": 1 }));

    let plan = plan_of(&[source("input"), suggesting, sink("out", "suggesting")]);

    assert_eq!(plan.node(NodeId::new(1)).width(), 1);
    assert_eq!(plan.node(NodeId::new(1)).narrowed(), Some(Narrowed::Suggested));
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

/// A route on `layer` reading [FixtureChunk](av_toolbox_core::testing::FixtureChunk) values.
fn route(layer: u32) -> ChunkRoute {
    let parallelism = Parallelism::chunk::<FixtureCuts>(());
    let key = parallelism.chunk_key().expect("a chunk stage has a key");

    ChunkRoute {
        layer: LayerId(layer),
        key,
    }
}

#[test]
fn a_node_on_another_layer_starts_its_own_region() {
    let mut first = temporal("first", "input", 1);
    first.workers = Some(4);
    let second = temporal("second", "first", 1);

    let specs = [source("input"), first, second, sink("out", "second")];
    let registry = testing::registry();
    let mut graph = build(&registry, &specs).expect("should build").into_graph();
    let first_id = node_named(&graph, "first");
    let second_id = node_named(&graph, "second");
    graph.node_mut(first_id).set_route(route(0));
    graph.node_mut(second_id).set_route(route(1));

    let plan = plan(&graph, &registry).expect("should plan");

    let first_region = plan.node(first_id).region();
    let second_region = plan.node(second_id).region();
    assert_eq!(plan.node(first_id).granularity(), Granularity::Chunk);
    assert_eq!(plan.node(second_id).granularity(), Granularity::Chunk);
    assert_eq!(plan.node(second_id).width(), 4);
    assert_ne!(first_region, second_region);
    assert_eq!(plan.node(first_id).route(), Some(route(0)));
    assert_eq!(plan.node(second_id).route(), Some(route(1)));
    assert_eq!(plan.region(first_region).route(), Some(route(0)));
    assert_eq!(plan.region(second_region).route(), Some(route(1)));
}

#[test]
fn a_layered_node_joins_a_region_with_no_layer_and_routes_it() {
    let specs = [
        source("input"),
        temporal("denoise", "input", 1),
        sink("out", "denoise"),
    ];
    let registry = testing::registry();
    let mut graph = build(&registry, &specs).expect("should build").into_graph();
    let input_id = node_named(&graph, "input");
    let denoise_id = node_named(&graph, "denoise");
    graph.node_mut(denoise_id).set_route(route(0));

    let plan = plan(&graph, &registry).expect("should plan");

    let region = plan.node(denoise_id).region();
    assert_eq!(plan.node(input_id).region(), region);
    assert_eq!(plan.region(region).route(), Some(route(0)));
}

#[test]
fn a_provider_runs_in_its_own_region() {
    let specs = [
        source("input"),
        temporal("denoise", "input", 1),
        temporal("sharpen", "denoise", 0),
        sink("out", "sharpen"),
    ];
    let registry = testing::registry();
    let mut graph = build(&registry, &specs).expect("should build").into_graph();
    satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");

    let plan = plan(&graph, &registry).expect("should plan");

    let input_id = node_named(&graph, "input");
    let provider_id = node_named(&graph, "fixture-cuts(input)");
    let denoise_id = node_named(&graph, "denoise");
    let provider_region = plan.node(provider_id).region();
    assert_ne!(plan.node(input_id).region(), provider_region);
    assert_ne!(plan.node(denoise_id).region(), provider_region);
    assert_eq!(plan.region(provider_region).nodes(), &[provider_id]);
    assert_eq!(plan.region(provider_region).head(), provider_id);
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

#[test]
fn a_node_plan_carries_its_stages_chunk_authority() {
    let plan = plan_of(&[
        source("input"),
        temporal("spatial", "input", 0),
        sink("out", "spatial"),
    ]);

    assert_eq!(plan.node(NodeId::new(0)).chunk_authority(), None);
    assert_eq!(plan.node(NodeId::new(1)).chunk_authority(), Some(0));
}

/// `test/temporal-merge` declares `[base, overlay]`. Writing them the other way round by
/// name puts edge order and signature order at odds, so a chunk authority read by edge
/// position rather than by slot would route by `overlay`'s chunks instead of `base`'s.
#[test]
fn a_chunk_authority_survives_inputs_written_out_of_signature_order() {
    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::named("overlay", "right"), testing::named("base", "left")];
    merge.args = Spanned::unknown(json!({ "radius": 0 }));

    let (graph, plan) = graph_and_plan_of(&[source("left"), source("right"), merge, sink("out", "merge")]);

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
fn a_region_of_only_frame_nodes_routes_by_frame() {
    // a source into a frame mode node, widened, so the two share a region
    let mut wide_source = testing::spec(StageKind::Source, "input", "test/widening-source");
    wide_source.workers = Some(4);

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::positional("input"), testing::positional("input")];
    merge.args = Spanned::unknown(json!({ "radius": 0 }));

    let plan = plan_of(&[wide_source, merge, sink("out", "merge")]);

    let region = plan.node(NodeId::new(1)).region();
    assert_eq!(plan.region(region).routing(), Routing::ByFrame);
}

#[test]
fn one_chunk_node_makes_the_whole_region_route_by_chunk() {
    // a frame mode node and a chunk mode node at the same width in one region
    let mut wide_source = testing::spec(StageKind::Source, "input", "test/widening-source");
    wide_source.workers = Some(4);

    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::positional("input"), testing::positional("input")];
    merge.args = Spanned::unknown(json!({ "radius": 1 }));

    let plan = plan_of(&[wide_source, merge, sink("out", "merge")]);

    let region = plan.node(NodeId::new(1)).region();
    assert_eq!(plan.region(region).routing(), Routing::ByChunk);
}

#[test]
fn a_stream_region_routes_by_chunk() {
    // routing is meaningless at width one, and ByChunk is the answer that changes nothing
    let plan = plan_of(&[source("input"), sink("out", "input")]);

    let region = plan.node(NodeId::new(0)).region();
    assert_eq!(plan.region(region).routing(), Routing::ByChunk);
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

#[test]
fn a_crossing_beside_another_consumer_heads_a_region_of_its_own() {
    let mut wide = temporal("wide", "input", 0);
    wide.workers = Some(4);
    let mut host = testing::spec(StageKind::Sink, "host", "test/host-stdout");
    host.inputs = vec![testing::positional("wide")];

    let (graph, plan) = placed_graph_and_plan_of(&[source("input"), wide, sink("gpu", "wide"), host]);

    let wide_id = node_named(&graph, "wide");
    let unload_id = node_named(&graph, "unload(wide)");
    let wide_region = plan.node(wide_id).region();
    let unload = plan.node(unload_id);
    let unload_region = unload.region();

    assert_ne!(unload_region, wide_region);
    assert_eq!(plan.region(unload_region).head(), unload_id);
    assert_eq!(unload.width(), 4);
    assert_eq!(plan.region(wide_region).exits(), &[wide_id]);
}

#[test]
fn a_crossing_that_is_its_producers_only_consumer_joins_its_region() {
    let mut wide = temporal("wide", "input", 0);
    wide.workers = Some(4);
    let mut host = testing::spec(StageKind::Sink, "host", "test/host-stdout");
    host.inputs = vec![testing::positional("wide")];

    let (graph, plan) = placed_graph_and_plan_of(&[source("input"), wide, host]);

    let wide_id = node_named(&graph, "wide");
    let unload_id = node_named(&graph, "unload(wide)");
    assert_eq!(plan.node(unload_id).region(), plan.node(wide_id).region());
}

#[test]
fn planning_a_host_stage_before_transfers_are_placed_is_an_error() {
    let mut host = testing::spec(StageKind::Sink, "host", "test/host-stdout");
    host.inputs = vec![testing::positional("input")];

    let diagnostics = plan_error(&[source("input"), host]);

    let reported = diagnostics.iter().next().expect("a diagnostic");
    assert_eq!(reported.code(), DiagnosticCode::TransfersNotPlaced);
    assert_eq!(reported.severity(), Severity::Error);
    assert!(reported.message().contains("host"));
}

#[test]
fn a_graph_of_device_stages_plans_before_transfers_are_placed() {
    let plan = plan_of(&[source("input"), temporal("blur", "input", 0), sink("out", "blur")]);

    assert_eq!(plan.regions().len(), 1);
}

fn widened(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut spec = temporal(name, input, 0);
    spec.workers = Some(workers);

    spec
}

fn assert_one_exit_per_region(graph: &CompiledGraph, plan: &ExecutionPlan) {
    for region in plan.regions() {
        assert!(
            region.exits().len() <= 1,
            "the region headed by {} has exits {:?}",
            graph.node(region.head()).name(),
            region.exits(),
        );
    }
}

fn assert_every_node_in_one_region(graph: &CompiledGraph, plan: &ExecutionPlan) {
    for node in graph.nodes() {
        let holding: Vec<RegionId> = plan
            .regions()
            .iter()
            .filter(|region| region.nodes().contains(&node.id()))
            .map(Region::id)
            .collect();

        let expected = plan.node(node.id()).region();
        assert_eq!(holding, vec![expected], "{}", node.name());
    }
}

/// A region's head, members, width and exits, each node by name.
type RegionSummary = (String, Vec<String>, usize, Vec<String>);

fn region_summary(graph: &CompiledGraph, plan: &ExecutionPlan) -> Vec<RegionSummary> {
    let name_of = |id: &NodeId| graph.node(*id).name().to_string();

    plan.regions()
        .iter()
        .map(|region| {
            let head = name_of(&region.head());
            let members = region.nodes().iter().map(name_of).collect();
            let exits = region.exits().iter().map(name_of).collect();

            (head, members, region.width(), exits)
        })
        .collect()
}

fn summary_of(head: &str, members: &[&str], width: usize, exits: &[&str]) -> RegionSummary {
    let members = members.iter().map(|name| name.to_string()).collect();
    let exits = exits.iter().map(|name| name.to_string()).collect();

    (head.to_string(), members, width, exits)
}

#[test]
fn a_region_that_would_have_two_exits_is_split() {
    let specs = [
        source("input"),
        temporal("filter1", "input", 0),
        temporal("filter2", "filter1", 0),
        widened("filter3", "filter1", 4),
        widened("filter4", "filter2", 4),
        sink("sink3", "filter3"),
        sink("sink4", "filter4"),
    ];

    let (graph, plan) = graph_and_plan_of(&specs);

    assert_one_exit_per_region(&graph, &plan);
    assert_every_node_in_one_region(&graph, &plan);
}

fn mask_of(name: &str, clip: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Mask, name, "test/temporal-merge");
    spec.inputs = vec![testing::positional(clip), testing::positional(clip)];
    spec.args = Spanned::unknown(json!({ "radius": 0 }));

    spec
}

#[test]
fn a_clip_blended_with_a_mask_built_from_it_plans_one_exit_per_region() {
    let mut vote = testing::spec(StageKind::Mask, "vote", "test/combine");
    vote.inputs = vec![testing::positional("xdog"), testing::positional("adaptive")];

    let mut blend = testing::spec(StageKind::Filter, "blend", "test/plain-blend");
    blend.inputs = vec![
        testing::named("base", "clip"),
        testing::named("overlay", "overlay"),
        testing::named("mask", "vote"),
    ];

    let specs = [
        source("clip"),
        source("overlay"),
        mask_of("xdog", "clip"),
        mask_of("adaptive", "clip"),
        vote,
        blend,
        sink("out", "blend"),
    ];

    let (graph, plan) = graph_and_plan_of(&specs);

    assert_one_exit_per_region(&graph, &plan);
    assert_every_node_in_one_region(&graph, &plan);
}

#[test]
fn a_region_whose_exits_are_only_read_outside_it_is_split_at_its_branch() {
    let specs = [
        source("input"),
        widened("wide", "input", 4),
        temporal("left", "wide", 0),
        temporal("right", "wide", 0),
        sink("out_left", "left"),
        sink("out_right", "right"),
    ];

    let (graph, plan) = graph_and_plan_of(&specs);
    let summary = region_summary(&graph, &plan);

    let expected = vec![
        summary_of("input", &["input"], 1, &["input"]),
        summary_of("wide", &["wide"], 4, &["wide"]),
        summary_of("left", &["left"], 4, &["left"]),
        summary_of("right", &["right"], 4, &["right"]),
        summary_of("out_left", &["out_left"], 1, &[]),
        summary_of("out_right", &["out_right"], 1, &[]),
    ];
    assert_eq!(summary, expected);
}

#[test]
fn a_chain_with_one_exit_per_region_plans_as_before() {
    let chain = [
        source("input"),
        widened("first", "input", 4),
        temporal("second", "first", 0),
        sink("out", "second"),
    ];
    let (graph, plan) = graph_and_plan_of(&chain);
    let expected = vec![
        summary_of("input", &["input"], 1, &["input"]),
        summary_of("first", &["first", "second"], 4, &["second"]),
        summary_of("out", &["out"], 1, &[]),
    ];

    let summary = region_summary(&graph, &plan);
    assert_eq!(summary, expected);
}

#[test]
fn a_diamond_with_one_exit_per_region_plans_as_before() {
    let mut merge = testing::spec(StageKind::Mask, "merge", "test/temporal-merge");
    merge.inputs = vec![testing::positional("left"), testing::positional("right")];
    merge.args = Spanned::unknown(json!({ "radius": 0 }));
    let diamond = [
        source("input"),
        widened("spread", "input", 4),
        temporal("left", "spread", 0),
        temporal("right", "spread", 0),
        merge,
        sink("out", "merge"),
    ];
    let (graph, plan) = graph_and_plan_of(&diamond);
    let expected = vec![
        summary_of("input", &["input"], 1, &["input"]),
        summary_of("spread", &["spread", "left", "right", "merge"], 4, &["merge"]),
        summary_of("out", &["out"], 1, &[]),
    ];

    let summary = region_summary(&graph, &plan);
    assert_eq!(summary, expected);
}

#[test]
fn a_crossing_with_one_exit_per_region_plans_as_before() {
    let mut host = testing::spec(StageKind::Sink, "host", "test/host-stdout");
    host.inputs = vec![testing::positional("wide")];
    let crossing = [
        source("input"),
        widened("wide", "input", 4),
        sink("gpu", "wide"),
        host,
    ];
    let (graph, plan) = placed_graph_and_plan_of(&crossing);
    let expected = vec![
        summary_of("input", &["input"], 1, &["input"]),
        summary_of("wide", &["wide"], 4, &["wide"]),
        summary_of("gpu", &["gpu"], 1, &[]),
        summary_of("unload(wide)", &["unload(wide)"], 4, &["unload(wide)"]),
        summary_of("host", &["host"], 1, &[]),
    ];

    let summary = region_summary(&graph, &plan);
    assert_eq!(summary, expected);
}
