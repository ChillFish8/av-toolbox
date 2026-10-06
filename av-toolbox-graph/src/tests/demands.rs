use av_toolbox_core::testing::FixtureCuts;
use av_toolbox_core::{AnyDemand, DemandKind, LayerDemand, StageKind};
use av_toolbox_gpu::LayerId;
use serde_json::{Value, json};

use crate::graph::{CompiledGraph, Node, NodeRole};
use crate::testing::{BreaksRequest, FixtureBreaks, FixtureMarks, MarksRequest};
use crate::{
    Diagnostic,
    DiagnosticCode,
    Diagnostics,
    NodeDemand,
    ProviderPlacement,
    ServedLayer,
    Span,
    Spanned,
    StageSpec,
    build,
    residency,
    satisfy_demands,
    solve,
    testing,
};

fn source(name: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Source, name, "test/file");
    spec.args = Spanned::unknown(json!({ "path": "a.mkv" }));

    spec
}

fn sink(name: &str, input: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Sink, name, "test/stdout");
    spec.inputs = vec![testing::positional(input)];

    spec
}

fn blur(name: &str, input: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/blur");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(json!({ "sigma": 1.0 }));

    spec
}

/// A filter whose chunk request is the fixture cuts demand.
fn chunked(name: &str, input: &str) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/temporal");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(json!({ "radius": 1 }));

    spec
}

fn demanding_pair(name: &str, clip: &str, other: &str, args: Value) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/demanding-pair");
    spec.inputs = vec![testing::named("clip", clip), testing::named("other", other)];
    spec.args = Spanned::unknown(args);

    spec
}

fn demanding(name: &str, input: &str, args: Value) -> StageSpec {
    let mut spec = testing::spec(StageKind::Filter, name, "test/demanding");
    spec.inputs = vec![testing::positional(input)];
    spec.args = Spanned::unknown(args);

    spec
}

fn built(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();

    build(&registry, specs).expect("should build").into_graph()
}

fn satisfied(specs: &[StageSpec]) -> CompiledGraph {
    let registry = testing::registry();
    let mut graph = built(specs);

    satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");
    graph
}

fn refused(specs: &[StageSpec]) -> Vec<Diagnostic> {
    let registry = testing::registry();
    let mut graph = built(specs);

    let Err(diagnostics) = satisfy_demands(&mut graph, &registry) else {
        panic!("the demands should be refused");
    };

    diagnostics.into_inner()
}

fn named<'a>(graph: &'a CompiledGraph, name: &str) -> &'a Node {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .unwrap_or_else(|| panic!("the graph has no node named {name}"))
}

fn providers(graph: &CompiledGraph) -> Vec<&Node> {
    graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.role(), NodeRole::Provider(_)))
        .collect()
}

fn placement(node: &Node) -> &ProviderPlacement {
    match node.role() {
        NodeRole::Provider(placement) => placement,
        other => panic!("{} is not a provider but {other:?}", node.name()),
    }
}

/// The name of the node feeding `input` of the node named `name`.
fn reads(graph: &CompiledGraph, name: &str, input: usize) -> String {
    let node = named(graph, name);
    let source = node.inputs()[input].source();

    graph.node(source).name().to_string()
}

fn consumer_names(graph: &CompiledGraph, name: &str) -> Vec<String> {
    let node = named(graph, name);

    node.consumers()
        .iter()
        .map(|id| graph.node(*id).name().to_string())
        .collect()
}

fn cuts() -> AnyDemand {
    AnyDemand::new::<FixtureCuts>(())
}

fn breaks(every: u32, after: Option<&str>) -> AnyDemand {
    let request = BreaksRequest {
        every,
        after: after.map(str::to_string),
        refuse: false,
    };

    AnyDemand::new::<FixtureBreaks>(request)
}

fn marks(every: u32) -> AnyDemand {
    let request = MarksRequest { every, cycle: false };

    AnyDemand::new::<FixtureMarks>(request)
}

fn provider_names(graph: &CompiledGraph) -> Vec<&str> {
    let mut names: Vec<&str> = providers(graph).iter().map(|node| node.name()).collect();
    names.sort_unstable();

    names
}

fn served(layer: u32, demand: AnyDemand, description: &str, demanded_by: &[&str]) -> ServedLayer {
    ServedLayer {
        layer: LayerId(layer),
        demand,
        description: description.to_string(),
        demanded_by: demanded_by.iter().map(|name| name.to_string()).collect(),
    }
}

fn codes(diagnostics: &[Diagnostic]) -> Vec<DiagnosticCode> {
    diagnostics.iter().map(Diagnostic::code).collect()
}

#[test]
fn a_chunk_stage_gets_a_provider_after_its_source() {
    let specs = [
        source("input"),
        chunked("denoise", "input"),
        sink("out", "denoise"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);

    let provider = found[0];
    assert_eq!(provider.name(), "fixture-cuts(input)");
    assert_eq!(provider.kind(), StageKind::Provider);
    assert_eq!(provider.identifier(), "test/stamping");
    assert_eq!(reads(&graph, provider.name(), 0), "input");
    assert_eq!(consumer_names(&graph, "input"), vec!["fixture-cuts(input)"]);
    assert_eq!(reads(&graph, "denoise", 0), "fixture-cuts(input)");

    let expected = ProviderPlacement {
        kind: DemandKind::of::<FixtureCuts>(),
        node: named(&graph, "input").id(),
        layers: vec![served(0, cuts(), "fixture-cuts", &["denoise"])],
    };
    assert_eq!(*placement(provider), expected);

    let args = vec![LayerDemand {
        layer: LayerId(0),
        demand: cuts(),
    }];
    let expected_args = serde_json::to_value(args).expect("serialises");
    assert_eq!(*provider.args(), expected_args);

    let denoise = named(&graph, "denoise");
    let expected_demands = vec![NodeDemand {
        input: "reference",
        kind: DemandKind::of::<FixtureCuts>(),
        description: "fixture-cuts".to_string(),
        layer: LayerId(0),
        chunks: true,
    }];
    assert_eq!(denoise.route().map(|route| route.layer), Some(LayerId(0)));
    assert_eq!(denoise.demands(), expected_demands.as_slice());
}

#[test]
fn one_provider_serves_two_branches_of_one_source() {
    let specs = [
        source("input"),
        chunked("first", "input"),
        chunked("second", "input"),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(reads(&graph, "first", 0), "fixture-cuts(input)");
    assert_eq!(reads(&graph, "second", 0), "fixture-cuts(input)");
    assert_eq!(
        consumer_names(&graph, "fixture-cuts(input)"),
        vec!["first", "second"]
    );
    assert_eq!(
        placement(found[0]).layers,
        vec![served(0, cuts(), "fixture-cuts", &["first", "second"])]
    );
}

#[test]
fn equal_demands_share_a_layer() {
    let specs = [
        source("input"),
        demanding("first", "input", json!({ "every": 4 })),
        demanding("second", "input", json!({ "every": 4 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name(), "fixture-breaks(input)");
    assert_eq!(
        placement(found[0]).layers,
        vec![served(0, breaks(4, None), "every 4", &["first", "second"])]
    );
    assert_eq!(named(&graph, "first").demands()[0].layer, LayerId(0));
    assert_eq!(named(&graph, "second").demands()[0].layer, LayerId(0));
}

#[test]
fn distinct_demands_get_distinct_layers() {
    let specs = [
        source("input"),
        demanding("first", "input", json!({ "every": 4, "chunks": true })),
        demanding("second", "input", json!({ "every": 8, "chunks": true })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(
        placement(found[0]).layers,
        vec![
            served(0, breaks(4, None), "every 4", &["first"]),
            served(1, breaks(8, None), "every 8", &["second"]),
        ]
    );
    assert_eq!(
        named(&graph, "first").route().map(|route| route.layer),
        Some(LayerId(0))
    );
    assert_eq!(
        named(&graph, "second").route().map(|route| route.layer),
        Some(LayerId(1))
    );
    assert!(named(&graph, "second").demands()[0].chunks);
}

#[test]
fn only_the_source_a_demand_reaches_gets_a_provider() {
    let specs = [
        source("clip"),
        source("other"),
        chunked("denoise", "clip"),
        sink("clip_out", "denoise"),
        sink("other_out", "other"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name(), "fixture-cuts(clip)");
    assert_eq!(reads(&graph, "fixture-cuts(clip)", 0), "clip");
    assert_eq!(consumer_names(&graph, "other"), vec!["other_out"]);
    assert_eq!(named(&graph, "other_out").route(), None);
}

#[test]
fn two_sources_each_get_a_provider() {
    let specs = [
        source("clip"),
        source("other"),
        chunked("first", "clip"),
        chunked("second", "other"),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let mut names: Vec<&str> = providers(&graph).iter().map(|node| node.name()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["fixture-cuts(clip)", "fixture-cuts(other)"]);
    assert_eq!(reads(&graph, "first", 0), "fixture-cuts(clip)");
    assert_eq!(reads(&graph, "second", 0), "fixture-cuts(other)");
    assert_eq!(
        named(&graph, "first").route().map(|route| route.layer),
        Some(LayerId(0))
    );
    assert_eq!(
        named(&graph, "second").route().map(|route| route.layer),
        Some(LayerId(1))
    );
}

#[test]
fn a_demand_follows_the_chunk_authority_through_a_join() {
    let mut join = testing::spec(StageKind::Filter, "join", "test/host-join");
    join.inputs = vec![testing::named("input", "other"), testing::named("guide", "clip")];

    let specs = [
        source("clip"),
        source("other"),
        join,
        chunked("denoise", "join"),
        sink("out", "denoise"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name(), "fixture-cuts(clip)");
    assert_eq!(reads(&graph, "fixture-cuts(clip)", 0), "clip");
    assert_eq!(reads(&graph, "join", 0), "other");
    assert_eq!(reads(&graph, "join", 1), "fixture-cuts(clip)");
    assert_eq!(
        named(&graph, "denoise").route().map(|route| route.layer),
        Some(LayerId(0))
    );
}

#[test]
fn demands_through_a_shared_stage_go_after_the_source() {
    let specs = [
        source("input"),
        blur("soften", "input"),
        demanding("first", "soften", json!({ "every": 4 })),
        demanding("second", "soften", json!({ "every": 4 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
        sink("raw", "input"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "input");
    assert_eq!(reads(&graph, "soften", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "raw", 0), "input");
    assert_eq!(reads(&graph, "first", 0), "soften");
}

#[test]
fn an_after_places_the_provider_after_that_stage() {
    let mut deband = testing::spec(StageKind::Filter, "deband", "test/deband");
    deband.inputs = vec![testing::positional("soften")];

    let specs = [
        source("input"),
        blur("soften", "input"),
        deband,
        demanding("wants", "deband", json!({ "every": 4, "after": "soften" })),
        sink("out", "wants"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(placement(found[0]).node, named(&graph, "soften").id());
    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "soften");
    assert_eq!(reads(&graph, "deband", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "wants", 0), "deband");
}

#[test]
fn demands_placed_behind_one_node_merge_into_one_instance() {
    let specs = [
        source("input"),
        blur("soften", "input"),
        demanding("first", "soften", json!({ "every": 4, "after": "soften" })),
        demanding("second", "soften", json!({ "every": 8, "after": "soften" })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name(), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "soften");
    assert_eq!(reads(&graph, "first", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "second", 0), "fixture-breaks(input)");
    assert_eq!(
        placement(found[0]).layers,
        vec![
            served(0, breaks(4, Some("soften")), "every 4", &["first"]),
            served(1, breaks(8, Some("soften")), "every 8", &["second"]),
        ]
    );
}

#[test]
fn demands_from_two_inputs_behind_one_node_merge_into_one_instance() {
    let specs = [
        source("input"),
        demanding_pair("pair", "input", "input", json!({ "every": 4, "both": true })),
        sink("out", "pair"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(reads(&graph, "pair", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "pair", 1), "fixture-breaks(input)");
    assert_eq!(
        placement(found[0]).layers,
        vec![served(0, breaks(4, None), "every 4", &["pair"])]
    );
}

#[test]
fn demands_placed_behind_different_nodes_get_separate_instances() {
    let specs = [
        source("input"),
        blur("soften", "input"),
        demanding("first", "soften", json!({ "every": 4, "after": "soften" })),
        demanding("second", "input", json!({ "every": 4 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let mut names: Vec<&str> = providers(&graph).iter().map(|node| node.name()).collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec!["fixture-breaks(input after soften)", "fixture-breaks(input)"]
    );
    assert_eq!(reads(&graph, "fixture-breaks(input after soften)", 0), "soften");
    assert_eq!(reads(&graph, "first", 0), "fixture-breaks(input after soften)");
    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "input");
    assert_eq!(reads(&graph, "second", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "soften", 0), "input");
}

#[test]
fn different_after_stages_on_one_path_get_separate_instances() {
    let mut deband = testing::spec(StageKind::Filter, "deband", "test/deband");
    deband.inputs = vec![testing::positional("soften")];

    let specs = [
        source("input"),
        blur("soften", "input"),
        deband,
        demanding("first", "deband", json!({ "every": 4, "after": "soften" })),
        demanding("second", "deband", json!({ "every": 4, "after": "deband" })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(providers(&graph).len(), 2);
    assert_eq!(reads(&graph, "fixture-breaks(input after soften)", 0), "soften");
    assert_eq!(reads(&graph, "deband", 0), "fixture-breaks(input after soften)");
    assert_eq!(reads(&graph, "fixture-breaks(input after deband)", 0), "deband");
    assert_eq!(reads(&graph, "first", 0), "deband");
    assert_eq!(reads(&graph, "second", 0), "fixture-breaks(input after deband)");
}

#[test]
fn equal_requests_on_separate_instances_get_separate_layers() {
    let specs = [
        source("clip"),
        source("other"),
        demanding("first", "clip", json!({ "every": 4 })),
        demanding("second", "other", json!({ "every": 4 })),
        sink("first_out", "first"),
        sink("second_out", "second"),
    ];

    let graph = satisfied(&specs);

    let clip = named(&graph, "fixture-breaks(clip)");
    let other = named(&graph, "fixture-breaks(other)");
    assert_eq!(
        placement(clip).layers,
        vec![served(0, breaks(4, None), "every 4", &["first"])]
    );
    assert_eq!(
        placement(other).layers,
        vec![served(1, breaks(4, None), "every 4", &["second"])]
    );
    assert_eq!(named(&graph, "first").demands()[0].layer, LayerId(0));
    assert_eq!(named(&graph, "second").demands()[0].layer, LayerId(1));
}

#[test]
fn an_instance_only_takes_its_demanders_edges() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4 })),
        sink("out", "wants"),
        sink("raw", "input"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(reads(&graph, "wants", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "raw", 0), "input");
    assert_eq!(consumer_names(&graph, "fixture-breaks(input)"), vec!["wants"]);
}

#[test]
fn a_demand_on_one_input_leaves_the_other_input_alone() {
    let specs = [
        source("clip"),
        source("other"),
        demanding_pair("pair", "clip", "other", json!({ "every": 4 })),
        sink("out", "pair"),
    ];

    let graph = satisfied(&specs);

    let found = providers(&graph);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].name(), "fixture-breaks(clip)");
    assert_eq!(reads(&graph, "pair", 0), "fixture-breaks(clip)");
    assert_eq!(reads(&graph, "pair", 1), "other");
}

#[test]
fn a_place_refusal_names_the_demanding_stage() {
    let mut wants = demanding("wants", "input", json!({ "every": 4, "refuse": true }));
    wants.span = Span::new(20, 30);

    let specs = [source("input"), wants, sink("out", "wants")];

    let diagnostics = refused(&specs);

    assert_eq!(codes(&diagnostics), vec![DiagnosticCode::PlaceFailed]);
    assert_eq!(diagnostics[0].path(), "filter.wants");
    assert_eq!(diagnostics[0].span(), Span::new(20, 30));
    assert_eq!(
        diagnostics[0].message(),
        "`wants` demands fixture-breaks on `input`, but the fixture refuses to place this request",
    );
}

#[test]
fn a_demand_with_no_provider_is_refused() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "unprovided": true })),
        sink("out", "wants"),
    ];

    let diagnostics = refused(&specs);

    assert_eq!(codes(&diagnostics), vec![DiagnosticCode::UnsatisfiedDemand]);
    assert_eq!(diagnostics[0].path(), "filter.wants");
    assert!(diagnostics[0].message().contains("`wants`"), "{diagnostics:?}");
}

#[test]
fn no_demands_leave_the_graph_alone() {
    let specs = [source("input"), blur("soften", "input"), sink("out", "soften")];

    let graph = satisfied(&specs);

    assert_eq!(graph.len(), 3);
    assert!(providers(&graph).is_empty());
    assert!(graph.nodes().iter().all(|node| node.route().is_none()));
    assert!(graph.nodes().iter().all(|node| node.demands().is_empty()));
}

#[test]
fn a_provider_refusal_names_the_stage_that_demanded_it() {
    let specs = [
        source("input"),
        demanding("eager", "input", json!({ "every": 0 })),
        sink("out", "eager"),
    ];
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let graph = satisfied(&specs);

    let Err(diagnostics) = solve(&graph, &registry, &capabilities) else {
        panic!("a break every 0 frames should be refused");
    };
    let diagnostics: Vec<Diagnostic> = Diagnostics::into_inner(diagnostics);

    let refusal = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.path() == "provider.fixture-breaks(input)")
        .unwrap_or_else(|| panic!("no diagnostic names the provider: {diagnostics:?}"));
    let notes: Vec<&str> = refusal.notes().iter().map(|note| note.message.as_str()).collect();
    assert!(notes.iter().any(|note| note.contains("`eager`")), "{notes:?}");
}

#[test]
fn a_provider_refusal_points_at_the_demanding_stage_and_notes_its_source() {
    let mut input = source("input");
    input.span = Span::new(0, 10);
    let mut eager = demanding("eager", "input", json!({ "every": 0 }));
    eager.span = Span::new(20, 30);

    let specs = [input, eager, sink("out", "eager")];
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let graph = satisfied(&specs);

    let Err(diagnostics) = solve(&graph, &registry, &capabilities) else {
        panic!("a break every 0 frames should be refused");
    };
    let diagnostics: Vec<Diagnostic> = Diagnostics::into_inner(diagnostics);

    let refusal = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.path() == "provider.fixture-breaks(input)")
        .unwrap_or_else(|| panic!("no diagnostic names the provider: {diagnostics:?}"));
    assert_eq!(refusal.span(), Span::new(20, 30));

    let source_note = refusal
        .notes()
        .iter()
        .find(|note| note.span == Span::new(0, 10))
        .unwrap_or_else(|| panic!("no note points at the source: {refusal:?}"));
    assert!(source_note.message.contains("`input`"), "{refusal:?}");
}

#[test]
fn a_provider_demand_places_its_dependency() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4, "marks": true })),
        sink("out", "wants"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(
        provider_names(&graph),
        vec!["fixture-breaks(input)", "fixture-marks(input)"]
    );

    let marking = named(&graph, "fixture-marks(input)");
    assert_eq!(
        placement(marking).layers,
        vec![served(0, marks(4), "marks every 4", &["wants"])]
    );

    let breaking = named(&graph, "fixture-breaks(input)");
    assert_eq!(
        placement(breaking).layers,
        vec![served(1, breaks(4, None), "every 4", &["fixture-marks(input)"])]
    );

    let expected_demands = vec![NodeDemand {
        input: "input",
        kind: DemandKind::of::<FixtureBreaks>(),
        description: "every 4".to_string(),
        layer: LayerId(1),
        chunks: false,
    }];
    assert_eq!(marking.demands(), expected_demands.as_slice());
}

#[test]
fn a_provider_dependency_sits_upstream_in_its_stack() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4, "marks": true })),
        sink("out", "wants"),
        sink("raw", "input"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "input");
    assert_eq!(reads(&graph, "fixture-marks(input)", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "wants", 0), "fixture-marks(input)");
    assert_eq!(reads(&graph, "raw", 0), "input");
}

#[test]
fn a_stage_and_a_provider_demanding_one_request_share_a_layer() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4, "marks": true })),
        demanding("plain", "input", json!({ "every": 4 })),
        sink("wants_out", "wants"),
        sink("plain_out", "plain"),
    ];

    let graph = satisfied(&specs);

    let breaking = named(&graph, "fixture-breaks(input)");
    assert_eq!(
        placement(breaking).layers,
        vec![served(
            1,
            breaks(4, None),
            "every 4",
            &["plain", "fixture-marks(input)"]
        )]
    );
    assert_eq!(named(&graph, "plain").demands()[0].layer, LayerId(1));
    assert_eq!(
        named(&graph, "fixture-marks(input)").demands()[0].layer,
        LayerId(1)
    );
}

#[test]
fn each_instance_in_a_stack_takes_only_its_demanders_edges() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4, "marks": true })),
        demanding("plain", "input", json!({ "every": 8 })),
        sink("wants_out", "wants"),
        sink("plain_out", "plain"),
        sink("raw", "input"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(
        consumer_names(&graph, "input"),
        vec!["raw", "fixture-breaks(input)"]
    );
    assert_eq!(
        consumer_names(&graph, "fixture-breaks(input)"),
        vec!["plain", "fixture-marks(input)"]
    );
    assert_eq!(consumer_names(&graph, "fixture-marks(input)"), vec!["wants"]);
}

#[test]
fn unrelated_instances_in_a_stack_are_ordered_by_kind_name() {
    let specs = [
        source("input"),
        demanding("wants", "input", json!({ "every": 4, "cuts": true })),
        sink("out", "wants"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "input");
    assert_eq!(reads(&graph, "fixture-cuts(input)", 0), "fixture-breaks(input)");
    assert_eq!(reads(&graph, "wants", 0), "fixture-cuts(input)");
}

#[test]
fn a_provider_cycle_is_refused_with_its_chain() {
    let mut wants = demanding(
        "wants",
        "input",
        json!({ "every": 4, "marks": true, "cycle": true }),
    );
    wants.span = Span::new(20, 30);

    let specs = [source("input"), wants, sink("out", "wants")];

    let diagnostics = refused(&specs);

    assert_eq!(codes(&diagnostics), vec![DiagnosticCode::ProviderCycle]);
    assert_eq!(diagnostics[0].path(), "filter.wants");
    assert_eq!(diagnostics[0].span(), Span::new(20, 30));

    let message = diagnostics[0].message();
    assert!(
        message.contains("fixture-marks(input) → fixture-marks(input)"),
        "{message}"
    );
}

#[test]
fn a_provider_demand_error_notes_the_chain_that_led_to_it() {
    let mut wants = demanding(
        "wants",
        "input",
        json!({ "every": 4, "marks": true, "cycle": true }),
    );
    wants.span = Span::new(20, 30);

    let specs = [source("input"), wants, sink("out", "wants")];

    let diagnostics = refused(&specs);

    let notes: Vec<&str> = diagnostics[0]
        .notes()
        .iter()
        .map(|note| note.message.as_str())
        .collect();
    assert!(
        notes.contains(&"`fixture-marks` demanded by `fixture-marks(input)`, demanded by `wants`"),
        "{notes:?}"
    );
}

#[test]
fn a_refusal_by_a_demanded_provider_notes_its_chain_at_the_stage() {
    let mut eager = demanding("eager", "input", json!({ "every": 0, "marks": true }));
    eager.span = Span::new(20, 30);

    let specs = [source("input"), eager, sink("out", "eager")];
    let registry = testing::registry();
    let capabilities = residency::tests::capabilities();
    let graph = satisfied(&specs);

    let Err(diagnostics) = solve(&graph, &registry, &capabilities) else {
        panic!("a break every 0 frames should be refused");
    };
    let diagnostics: Vec<Diagnostic> = Diagnostics::into_inner(diagnostics);

    let refusal = diagnostics
        .iter()
        .find(|diagnostic| diagnostic.path() == "provider.fixture-breaks(input)")
        .unwrap_or_else(|| panic!("no diagnostic names the provider: {diagnostics:?}"));
    assert_eq!(refusal.span(), Span::new(20, 30));

    let chain = refusal
        .notes()
        .iter()
        .find(|note| {
            note.message
                .contains("demanded by `fixture-marks(input)`, demanded by `eager`")
        })
        .unwrap_or_else(|| panic!("no note spells out the chain: {refusal:?}"));
    assert_eq!(chain.span, Span::new(20, 30));
}

#[test]
fn a_provider_on_a_wider_edge_set_sits_upstream_of_a_narrower_one() {
    let specs = [
        source("input"),
        demanding("wide", "input", json!({ "every": 4, "cuts": true })),
        chunked("plain", "input"),
        sink("wide_out", "wide"),
        sink("plain_out", "plain"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(reads(&graph, "fixture-cuts(input)", 0), "input");
    assert_eq!(reads(&graph, "plain", 0), "fixture-cuts(input)");
    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "fixture-cuts(input)");
    assert_eq!(reads(&graph, "wide", 0), "fixture-breaks(input)");
    assert_eq!(
        consumer_names(&graph, "fixture-cuts(input)"),
        vec!["plain", "fixture-breaks(input)"]
    );
    assert_eq!(consumer_names(&graph, "fixture-breaks(input)"), vec!["wide"]);
}

#[test]
fn partly_overlapping_instances_in_a_stack_form_one_chain() {
    let specs = [
        source("input"),
        demanding("first", "input", json!({ "every": 4 })),
        demanding("both", "input", json!({ "every": 4, "cuts": true })),
        chunked("last", "input"),
        sink("first_out", "first"),
        sink("both_out", "both"),
        sink("last_out", "last"),
    ];

    let graph = satisfied(&specs);

    assert_eq!(reads(&graph, "fixture-breaks(input)", 0), "input");
    assert_eq!(
        consumer_names(&graph, "fixture-breaks(input)"),
        vec!["first", "fixture-cuts(input)"]
    );
    assert_eq!(
        consumer_names(&graph, "fixture-cuts(input)"),
        vec!["both", "last"]
    );
}

#[test]
fn a_regrown_provider_replaces_its_earlier_demands() {
    let specs = [
        source("input"),
        demanding("direct", "input", json!({ "every": 8, "marks": true })),
        demanding("relayed", "input", json!({ "every": 4, "relays": true })),
        sink("direct_out", "direct"),
        sink("relayed_out", "relayed"),
    ];

    let graph = satisfied(&specs);

    let marking = named(&graph, "fixture-marks(input)");
    let expected_demands = vec![NodeDemand {
        input: "input",
        kind: DemandKind::of::<FixtureBreaks>(),
        description: "every 4".to_string(),
        layer: LayerId(4),
        chunks: false,
    }];
    assert_eq!(marking.demands(), expected_demands.as_slice());

    let breaking = named(&graph, "fixture-breaks(input)");
    assert_eq!(
        placement(breaking).layers,
        vec![served(4, breaks(4, None), "every 4", &["fixture-marks(input)"])]
    );
}

#[test]
fn a_failed_provider_demand_is_reported_once_when_asked_again() {
    let specs = [
        source("input"),
        demanding(
            "direct",
            "input",
            json!({ "every": 8, "marks": true, "cycle": true }),
        ),
        demanding("relayed", "input", json!({ "every": 4, "relays": true })),
        sink("direct_out", "direct"),
        sink("relayed_out", "relayed"),
    ];

    let diagnostics = refused(&specs);

    assert_eq!(codes(&diagnostics), vec![DiagnosticCode::ProviderCycle]);
}
