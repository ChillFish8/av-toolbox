use std::time::Duration;

use av_toolbox_graph::{Diagnostic, DiagnosticCode, Diagnostics, Routing};

use crate::engine::plan::{default_source_share, holds_nothing_in_frame_region};
use crate::engine::tests::fixtures::sources::SOURCE_BUDGETS;
use crate::engine::tests::harness::{
    build_engine,
    clear_recorded,
    node_id,
    planned_graph,
    planned_width,
    recorded,
    routing_of,
    run_result_with_watchdog,
    run_with_watchdog,
};
use crate::engine::tests::specs::{
    allocating_spec,
    delay_spec,
    filter_spec,
    masking_spec,
    sink_spec,
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
