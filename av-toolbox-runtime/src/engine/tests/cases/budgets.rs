use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use av_toolbox_graph::{CompiledGraph, Diagnostic, DiagnosticCode, Diagnostics, RegionId, Routing};

use crate::engine::plan::{default_source_share, holds_nothing_in_frame_region, inherited_budgets};
use crate::engine::tests::fixtures::gated::{SLOW_GATES, SLOW_PARKED};
use crate::engine::tests::fixtures::sources::{SOURCE_BUDGET, SOURCE_BUDGETS, SOURCE_DONE};
use crate::engine::tests::harness::{
    NOTHING_MORE_COMING,
    WATCHDOG,
    assert_every_thread_exited,
    build_engine,
    clear_recorded,
    node_id,
    planned_graph,
    planned_width,
    recorded,
    routing_of,
    run_driven_with_watchdog,
    run_result_with_watchdog,
    run_with_watchdog,
    share_a_region,
};
use crate::engine::tests::specs::{
    allocating_spec,
    delay_spec,
    filter_spec,
    join_spec,
    masking_spec,
    sink_spec,
    slow_spec,
    source_spec,
};
use crate::engine::tests::support::{format, mask_format};
use crate::{RunError, Settings};

/// Four allocating filters between a source and its sink, every one of them at width one, so all
/// six nodes run in the source's own region and share its one budgeted context.
///
/// [RegionRunner](crate::RegionRunner) holds every node's output until it returns, so this region
/// needs five items resident at once. The derived budget used to be four per worker, which is four
/// for a plan of one worker, and the fifth allocation waited on room only its own caller could
/// ever free.
#[test]
fn a_source_region_that_allocates_at_every_node_runs_on_the_default_budget() {
    let property = "a_source_region_that_allocates_at_every_node_runs_on_the_default_budget";
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0, 0], 0),
        allocating_spec("first", "input"),
        allocating_spec("second", "first"),
        allocating_spec("third", "second"),
        allocating_spec("fourth", "third"),
        sink_spec("out", "fourth"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    assert_eq!(recorded(), vec![0, 1, 2, 3]);
    assert_eq!(report.frames, 4);
}

/// A budget with room for two frames, and a region that allocates three before it releases any.
///
/// Nothing but a frame being dropped can free room, and every frame this region holds is dropped
/// only once the pass holding them all returns, so the third allocation waits on room that is
/// never coming. The deadline is what ends it, and the error says what it was waiting under.
#[test]
fn a_source_waiting_for_room_nothing_will_free_fails_the_run() {
    let property = "a_source_waiting_for_room_nothing_will_free_fails_the_run";
    let budget = 2 * format().total_bytes();
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        allocating_spec("first", "input"),
        allocating_spec("second", "first"),
        sink_spec("out", "second"),
    ];
    let engine = build_engine(
        &specs,
        Settings {
            frame_budget: Some(budget),
            stall_deadline: Some(Duration::from_millis(100)),
            frame_limit: None,
            warmup: 0,
            profiling: None,
        },
    );

    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("a source that waited on room nothing could free should have failed the run");
    };
    let RunError::BudgetStalled { limit, outstanding } = error else {
        panic!("the run reported {error}");
    };

    assert_eq!(limit, budget);
    assert_eq!(
        outstanding, budget,
        "{property}: the budget was not full when it gave up"
    );
}

/// A plan whose largest item is a mask rather than a frame.
///
/// The source, the mask stage and the sink all run in one region, so the source's share has to
/// cover one item per node with the mask's own bytes, not the frame's. A budget that only ever
/// measured frames was eleven times too small for this plan, and deadlocked on the mask.
#[test]
fn a_budget_derived_for_a_plan_with_masks_counts_them() {
    let property = "a_budget_derived_for_a_plan_with_masks_counts_them";
    SOURCE_BUDGETS.lock().expect("not poisoned").clear();

    let specs = vec![
        source_spec("input", &[0, 0], 0),
        masking_spec("mask", "input"),
        sink_spec("out", "mask"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, property);

    let limit = SOURCE_BUDGETS
        .lock()
        .expect("not poisoned")
        .iter()
        .find(|(id, _)| *id == 0)
        .and_then(|(_, limit)| *limit)
        .expect("the source should have started with a budget");
    let region_holds = 3 * mask_format().bytes();

    assert!(
        limit >= region_holds,
        "{property}: a share of {limit} bytes cannot hold the {region_holds} bytes its region does",
    );
}

#[test]
fn a_source_share_makes_room_for_held_frames() {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        filter_spec("pass", "input", None),
        sink_spec("out", "pass"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");
    let regions = plan.regions();
    let none_held = vec![0; graph.len()];
    let mut some_held = none_held.clone();
    let pass = node_id(&graph, "pass");
    some_held[pass.index()] = 20;

    let without = default_source_share(&graph, &solved, regions, &none_held);
    let with = default_source_share(&graph, &solved, regions, &some_held);

    assert_eq!(with - without, 20 * format().total_bytes());
}

#[test]
fn a_source_share_counts_a_providers_held_frames() {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        delay_spec("delay", "input", &[0], 1, false),
        sink_spec("out", "delay"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");
    let regions = plan.regions();
    let none_held = vec![0; graph.len()];
    let mut some_held = none_held.clone();
    let provider = node_id(&graph, "fixture-cuts(input)");
    some_held[provider.index()] = 12;

    let without = default_source_share(&graph, &solved, regions, &none_held);
    let with = default_source_share(&graph, &solved, regions, &some_held);

    assert_eq!(with - without, 12 * format().total_bytes());
}

#[test]
fn a_source_share_counts_held_frames_once_per_worker() {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        filter_spec("pass", "input", Some(4)),
        sink_spec("out", "pass"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");
    let regions = plan.regions();
    let none_held = vec![0; graph.len()];
    let mut some_held = none_held.clone();
    let pass = node_id(&graph, "pass");
    some_held[pass.index()] = 20;

    assert_eq!(planned_width(&graph, "pass"), 4);

    let without = default_source_share(&graph, &solved, regions, &none_held);
    let with = default_source_share(&graph, &solved, regions, &some_held);

    assert_eq!(with - without, 20 * 4 * format().total_bytes());
}

#[test]
fn a_frame_routed_node_declaring_held_frames_is_refused() {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        filter_spec("pass", "input", Some(4)),
        sink_spec("out", "pass"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "pass"), Routing::ByFrame);

    let plan = graph.execution_plan().expect("a planned graph");
    let pass = node_id(&graph, "pass");
    let region = plan.region(plan.node(pass).region());
    let mut held = vec![0; graph.len()];
    held[pass.index()] = 1;
    let mut diagnostics = Diagnostics::new();

    let accepted = holds_nothing_in_frame_region(&graph, region, &held, &mut diagnostics);

    assert!(!accepted);

    let codes: Vec<DiagnosticCode> = diagnostics.iter().map(Diagnostic::code).collect();
    assert_eq!(codes, vec![DiagnosticCode::HeldInFrameRegion]);
}

#[test]
fn a_frame_routed_node_holding_nothing_is_accepted() {
    let specs = vec![
        source_spec("input", &[0, 0], 0),
        filter_spec("pass", "input", Some(4)),
        sink_spec("out", "pass"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert_eq!(routing_of(&graph, "pass"), Routing::ByFrame);

    let plan = graph.execution_plan().expect("a planned graph");
    let pass = node_id(&graph, "pass");
    let region = plan.region(plan.node(pass).region());
    let held = vec![0; graph.len()];
    let mut diagnostics = Diagnostics::new();

    let accepted = holds_nothing_in_frame_region(&graph, region, &held, &mut diagnostics);

    assert!(accepted);
    assert!(diagnostics.is_empty());
}

/// The region `name` runs in.
fn region_of(graph: &CompiledGraph, name: &str) -> RegionId {
    let plan = graph.execution_plan().expect("a planned graph");
    let node = node_id(graph, name);

    plan.node(node).region()
}

/// `fresh` allocates a new frame for every input in a region of its own after the provider, and
/// worker zero of `slow` is held on the first chunk so nothing reaches the sink. The frames `fresh`
/// allocates pile up behind that chunk, and the source can only finish while they do if they are
/// not charged to its share.
///
/// `fresh` charges the share without waiting for room, so the peak may pass the share by the one
/// frame it allocates between the source's waits.
#[test]
fn frames_allocated_after_a_provider_are_charged_to_the_source_share() {
    let property = "frames_allocated_after_a_provider_are_charged_to_the_source_share";
    clear_recorded();

    let (parked_sender, parked_receiver) = mpsc::channel();
    let (gate_sender, gate_receiver) = mpsc::channel();
    let (done_sender, done_receiver) = mpsc::channel();

    *SLOW_PARKED[0].lock().expect("not poisoned") = Some(parked_sender);
    *SLOW_GATES[0].lock().expect("not poisoned") = Some(gate_receiver);
    *SOURCE_DONE.lock().expect("not poisoned") = Some(done_sender);

    let chunks: Vec<u64> = (0..48).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        allocating_spec("fresh", "input"),
        slow_spec("slow", "fresh", 2),
        sink_spec("out", "slow"),
    ];
    let (graph, _solved) = planned_graph(&specs);

    assert!(!share_a_region(&graph, "input", "fresh"));
    assert!(!share_a_region(&graph, "fresh", "slow"));

    let engine = build_engine(&specs, Settings::default());

    let report = run_driven_with_watchdog(engine, property, |_cancel| {
        parked_receiver
            .recv_timeout(WATCHDOG)
            .expect("the held worker should take its chunk and park inside it");

        let finished = done_receiver.recv_timeout(NOTHING_MORE_COMING);
        assert!(
            finished.is_err(),
            "{property}: the source emitted every frame while the frames after it were held",
        );

        gate_sender.send(()).ok();
    })
    .expect("the run should finish once the held worker is released");

    let budget = SOURCE_BUDGET
        .lock()
        .expect("not poisoned")
        .clone()
        .expect("the source should have started with a budget");

    let allocating_stages_after_the_source = 1;
    let allowance = allocating_stages_after_the_source * format().total_bytes();
    let ceiling = budget.limit() + allowance;
    assert!(
        budget.peak() <= ceiling,
        "{property}: {} bytes were outstanding against a {} byte share plus {allowance} for `fresh`",
        budget.peak(),
        budget.limit(),
    );
    assert_eq!(report.frames, 48, "{property}: not every chunk reached the sink");
    assert_every_thread_exited(property);
}

#[test]
fn a_chain_of_single_worker_regions_after_a_source_inherits_its_budget() {
    let specs = vec![
        source_spec("input", &[0, 1], 0),
        allocating_spec("fresh", "input"),
        slow_spec("slow", "fresh", 2),
        sink_spec("out", "slow"),
    ];
    let (graph, _solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");

    let inherited = inherited_budgets(&graph, plan.regions());

    let source = region_of(&graph, "input");
    let provider = region_of(&graph, "fixture-cuts(input)");
    let fresh = region_of(&graph, "fresh");
    let expected = HashMap::from([(provider, source), (fresh, source)]);
    assert_eq!(inherited, expected);
}

#[test]
fn a_region_after_a_disperse_does_not_inherit_a_budget() {
    let specs = vec![
        source_spec("input", &[0, 1], 0),
        slow_spec("slow", "input", 2),
        filter_spec("narrow", "slow", Some(1)),
        sink_spec("out", "narrow"),
    ];
    let (graph, _solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");

    let inherited = inherited_budgets(&graph, plan.regions());

    let source = region_of(&graph, "input");
    let provider = region_of(&graph, "fixture-cuts(input)");
    let expected = HashMap::from([(provider, source)]);
    assert!(!share_a_region(&graph, "slow", "narrow"));
    assert_eq!(inherited, expected);
}

#[test]
fn a_region_fed_by_two_sources_does_not_inherit_a_budget() {
    let specs = vec![
        source_spec("base", &[0, 1], 0),
        source_spec("overlay", &[0, 1], 1),
        join_spec("join", "base", "overlay"),
        sink_spec("out", "join"),
    ];
    let (graph, _solved) = planned_graph(&specs);
    let plan = graph.execution_plan().expect("a planned graph");

    let inherited = inherited_budgets(&graph, plan.regions());

    let base = region_of(&graph, "base");
    let provider = region_of(&graph, "fixture-cuts(base)");
    let expected = HashMap::from([(provider, base)]);
    assert_eq!(inherited, expected);
}
