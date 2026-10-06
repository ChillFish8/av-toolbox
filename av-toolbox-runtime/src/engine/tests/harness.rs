use std::collections::HashSet;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use av_toolbox_core::testing::{HostStamping, Stamping, StampingInstance};
use av_toolbox_core::{LayerDemand, PipelineRegistry};
use av_toolbox_gpu::testing_context;
use av_toolbox_graph::{
    CompiledGraph,
    Diagnostics,
    NodeId,
    NodeRole,
    Region,
    Routing,
    SolvedFormats,
    StageBox,
    StageSpec,
    build,
    place_transfers,
    plan,
    satisfy_demands,
    solve,
};

use crate::engine::tests::fixtures::delay::{Delay, DelayArgs, DelayInstance};
use crate::engine::tests::fixtures::double::{Double, DoubleInstance};
use crate::engine::tests::fixtures::failing::{
    Failing,
    FailingArgs,
    FailingInstance,
    Panicking,
    PanickingInstance,
};
use crate::engine::tests::fixtures::filters::{
    Allocating,
    AllocatingInstance,
    Flushing,
    FlushingInstance,
    HostPass,
    HostPassInstance,
    PASS_BUILD_CALLS,
    Pass,
    PassInstance,
};
use crate::engine::tests::fixtures::gated::{
    HostSlow,
    HostSlowInstance,
    MATCHED,
    Matching,
    MatchingInstance,
    Slow,
    SlowInstance,
    Tracked,
    TrackedInstance,
};
use crate::engine::tests::fixtures::joins::{Join, JoinInstance, Referring, ReferringInstance};
use crate::engine::tests::fixtures::shaping::{
    Masking,
    MaskingInstance,
    Multiplying,
    MultiplyingInstance,
    RestampDefinition,
    RestampInstance,
};
use crate::engine::tests::fixtures::sinks::{
    Gated,
    GatedInstance,
    Hoarding,
    HoardingInstance,
    HostRecording,
    HostRecordingInstance,
    MetaRecording,
    MetaRecordingInstance,
    Pairing,
    PairingInstance,
    RECORDED,
    Recording,
    RecordingInstance,
};
use crate::engine::tests::fixtures::sources::{
    HostListed,
    HostListedInstance,
    Listed,
    ListedArgs,
    ListedInstance,
};
use crate::engine::tests::support::{ASSIGNED, STAGES_BUILT, STAGES_DROPPED, testing_capabilities};
use crate::{Cancel, Engine, RunError, RunReport, Settings};

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();
    registry.register_source::<Listed>().expect("should register");
    registry.register_source::<HostListed>().expect("should register");
    registry.register_filter::<Pass>().expect("should register");
    registry.register_filter::<HostPass>().expect("should register");
    registry.register_filter::<Allocating>().expect("should register");
    registry.register_filter::<Flushing>().expect("should register");
    registry.register_filter::<Delay>().expect("should register");
    registry.register_filter::<Double>().expect("should register");
    registry.register_mask::<Masking>().expect("should register");
    registry
        .register_transform::<Multiplying>()
        .expect("should register");
    registry.register_filter::<Join>().expect("should register");
    registry.register_filter::<Referring>().expect("should register");
    registry.register_filter::<Slow>().expect("should register");
    registry.register_filter::<HostSlow>().expect("should register");
    registry.register_filter::<Tracked>().expect("should register");
    registry.register_filter::<Matching>().expect("should register");
    registry.register_filter::<Failing>().expect("should register");
    registry.register_filter::<Panicking>().expect("should register");
    registry.register_sink::<Recording>().expect("should register");
    registry
        .register_sink::<HostRecording>()
        .expect("should register");
    registry.register_sink::<Hoarding>().expect("should register");
    registry.register_sink::<Pairing>().expect("should register");
    registry.register_sink::<Gated>().expect("should register");
    registry
        .register_meta_transform::<RestampDefinition>()
        .expect("should register");
    registry
        .register_meta_sink::<MetaRecording>()
        .expect("should register");
    registry.register_provider::<Stamping>().expect("should register");
    registry
        .register_provider::<HostStamping>()
        .expect("should register");
    registry
}

/// Builds one node's stage directly from its identifier, the same way
/// [crate::worker::region]'s own tests build a [StageBox] by hand.
///
/// This is what [Engine::build_with] calls instead of `instantiate_region`, so the whole engine
/// runs against these fixtures with no backend and no registry lookup at run time. `worker` is
/// this region's own worker index, read by `test/slow` for its own gates and by every fixture
/// carrying a [WorkerGate](crate::engine::tests::support::WorkerGate) for theirs. Holding a
/// worker is how a test forces where work lands, since a held worker is never drained and so is
/// never claimed again.
fn build_stage(graph: &CompiledGraph, id: NodeId, worker: usize) -> (NodeId, StageBox) {
    let node = graph.node(id);

    if let NodeRole::Crossing(crossing) = node.role() {
        return (id, StageBox::Crossing(*crossing));
    }

    let stage = match node.identifier() {
        "test/listed" => {
            let args: ListedArgs = serde_json::from_value(node.args().clone()).expect("valid listed args");
            StageBox::Source(Box::new(ListedInstance::from_args(args)))
        },
        "test/host-listed" => {
            let args: ListedArgs = serde_json::from_value(node.args().clone()).expect("valid listed args");
            StageBox::HostSource(Box::new(HostListedInstance::from_args(args)))
        },
        "test/pass" => {
            PASS_BUILD_CALLS.fetch_add(1, Ordering::SeqCst);
            StageBox::Filter(Box::new(PassInstance::new()))
        },
        "test/host-pass" => StageBox::HostFilter(Box::new(HostPassInstance::new())),
        "test/allocating" => StageBox::Filter(Box::new(AllocatingInstance::new())),
        "test/flushing" => StageBox::Filter(Box::new(FlushingInstance::new())),
        "test/delay" => {
            let args: DelayArgs = serde_json::from_value(node.args().clone()).expect("valid delay args");
            StageBox::Filter(Box::new(DelayInstance::from_args(args)))
        },
        "test/double" => StageBox::Filter(Box::new(DoubleInstance::new())),
        "test/masking" => StageBox::Mask(Box::new(MaskingInstance::new())),
        "test/multiplying" => StageBox::Transform(Box::new(MultiplyingInstance::new())),
        "test/join" => StageBox::Filter(Box::new(JoinInstance::new())),
        "test/referring" => StageBox::Filter(Box::new(ReferringInstance::new())),
        "test/slow" => StageBox::Filter(Box::new(SlowInstance::for_worker(worker))),
        "test/host-slow" => StageBox::HostFilter(Box::new(HostSlowInstance::for_worker(worker))),
        "test/tracked" => StageBox::Filter(Box::new(TrackedInstance::for_worker(worker))),
        "test/matching" => StageBox::Filter(Box::new(MatchingInstance::for_worker(worker))),
        "test/failing" => {
            let args: FailingArgs = serde_json::from_value(node.args().clone()).expect("valid failing args");
            StageBox::Filter(Box::new(FailingInstance::for_worker(args, worker)))
        },
        "test/panicking" => StageBox::Filter(Box::new(PanickingInstance::new())),
        "test/recording" => StageBox::Sink(Box::new(RecordingInstance::new())),
        "test/host-recording" => StageBox::HostSink(Box::new(HostRecordingInstance::new())),
        "test/hoarding" => StageBox::Sink(Box::new(HoardingInstance::new())),
        "test/pairing" => StageBox::Sink(Box::new(PairingInstance::new())),
        "test/gated" => StageBox::Sink(Box::new(GatedInstance::new())),
        "test/restamp" => StageBox::MetaTransform(Box::new(RestampInstance)),
        "test/meta-recording" => StageBox::MetaSink(Box::new(MetaRecordingInstance::new())),
        "test/stamping" => {
            let args: Vec<LayerDemand> =
                serde_json::from_value(node.args().clone()).expect("valid stamping args");
            let instance = StampingInstance::new(&args);
            StageBox::Provider(Box::new(instance))
        },
        "test/host-stamping" => {
            let args: Vec<LayerDemand> =
                serde_json::from_value(node.args().clone()).expect("valid stamping args");
            let instance = StampingInstance::new(&args);
            StageBox::HostProvider(Box::new(instance))
        },
        other => panic!("no test fixture knows how to build {other}"),
    };

    (id, stage)
}

pub(crate) fn instantiate(
    graph: &CompiledGraph,
    region: &Region,
    worker: usize,
) -> Result<Vec<(NodeId, StageBox)>, Diagnostics> {
    Ok(region
        .nodes()
        .iter()
        .map(|&id| build_stage(graph, id, worker))
        .collect())
}

/// Builds, places providers, solves, places crossings and plans `specs`, without ever touching a backend.
pub(crate) fn planned_graph(specs: &[StageSpec]) -> (CompiledGraph, SolvedFormats) {
    let registry = registry();
    let mut graph = build(&registry, specs).expect("should build").into_graph();
    satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");

    let capabilities = testing_capabilities();
    let mut solved = solve(&graph, &registry, &capabilities).expect("should solve");
    place_transfers(&mut graph, &registry, &mut solved).expect("should place crossings");

    let made_plan = plan(&graph, &registry).expect("should plan");
    graph.set_execution_plan(made_plan);

    (graph, solved)
}

/// Builds, plans, solves and instantiates an engine from `specs`, entirely through the fake
/// allocator seam: no `PipelineRegistry::prepare`-driven build, no backend.
pub(crate) fn build_engine(specs: &[StageSpec], settings: Settings) -> Engine {
    let (graph, solved) = planned_graph(specs);

    engine_from(&graph, &solved, settings)
}

/// The same, for a test that needs the graph itself as well, to name the node it asserts on.
pub(crate) fn engine_from(graph: &CompiledGraph, solved: &SolvedFormats, settings: Settings) -> Engine {
    let registry = registry();
    let held = graph
        .held_frames(&registry, solved)
        .expect("every fixture prepares");

    Engine::build_with(
        graph,
        solved,
        &held,
        settings,
        |region, worker| instantiate(graph, region, worker),
        testing_context,
    )
    .expect("should build the engine")
}

/// The frames each node declares it holds, read the way [engine_from] reads them.
pub(crate) fn declared_held(graph: &CompiledGraph, solved: &SolvedFormats) -> Vec<usize> {
    graph
        .held_frames(&registry(), solved)
        .expect("every fixture prepares")
}

/// Solves `specs` the way [planned_graph] does, handing back whatever diagnostics it refused with.
pub(crate) fn solve_result(specs: &[StageSpec]) -> Result<SolvedFormats, Diagnostics> {
    let registry = registry();
    let mut graph = build(&registry, specs).expect("should build").into_graph();
    satisfy_demands(&mut graph, &registry).expect("should satisfy every demand");

    solve(&graph, &registry, &testing_capabilities())
}

pub(crate) fn node_id(graph: &CompiledGraph, name: &str) -> NodeId {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .map(|node| node.id())
        .unwrap_or_else(|| panic!("the graph has no node named {name}"))
}

/// The routing the plan settled on for the region `head` leads.
pub(crate) fn routing_of(graph: &CompiledGraph, head: &str) -> Routing {
    let plan = graph.execution_plan().expect("a planned graph");
    let node = node_id(graph, head);

    plan.region(plan.node(node).region()).routing()
}

/// The width the plan advertises for the region `head` leads.
pub(crate) fn planned_width(graph: &CompiledGraph, head: &str) -> usize {
    let plan = graph.execution_plan().expect("a planned graph");
    let node = node_id(graph, head);

    plan.region(plan.node(node).region()).width()
}

/// Whether the nodes named `first` and `second` sit in the same region of the plan.
pub(crate) fn share_a_region(graph: &CompiledGraph, first: &str, second: &str) -> bool {
    let plan = graph.execution_plan().expect("a planned graph");
    let first_region = plan.node(node_id(graph, first)).region();
    let second_region = plan.node(node_id(graph, second)).region();

    first_region == second_region
}

/// The chunk authority the plan settled on for the region `head` leads.
pub(crate) fn authority_of(graph: &CompiledGraph, head: &str) -> Option<usize> {
    let plan = graph.execution_plan().expect("a planned graph");
    let node = node_id(graph, head);

    plan.region(plan.node(node).region()).chunk_authority()
}

/// How long any of these tests will wait for a run before calling it hung.
pub(crate) const WATCHDOG: Duration = Duration::from_secs(10);

/// How long to wait when a test is proving something does **not** arrive.
///
/// A [WATCHDOG] length wait costs its full ten seconds every run, since timing out is the passing
/// path rather than the failing one. This is long enough that an over eager dispatch would have
/// been seen and short enough not to dominate the suite.
pub(crate) const NOTHING_MORE_COMING: Duration = Duration::from_millis(250);

/// Runs `engine` on its own thread and waits for it with a generous timeout, so a design mistake
/// that hangs the exchange reports as a named failure instead of a stalled suite.
pub(crate) fn run_with_watchdog(engine: Engine, property: &str) -> RunReport {
    match run_result_with_watchdog(engine, property) {
        Ok(report) => report,
        Err(error) => panic!("{property}: the engine run failed: {error}"),
    }
}

/// The same, handing back whatever the run answered with, for a test about a run that fails.
pub(crate) fn run_result_with_watchdog(engine: Engine, property: &str) -> Result<RunReport, RunError> {
    run_driven_with_watchdog(engine, property, |_cancel| {})
}

/// The same again, with `drive` running on the test's own thread while the engine runs.
///
/// A test that cancels part way through does its waiting and its cancelling in there, given the
/// signal that stops this run. Everything `drive` waits for is a channel, never a delay.
pub(crate) fn run_driven_with_watchdog<Drive>(
    engine: Engine,
    property: &str,
    drive: Drive,
) -> Result<RunReport, RunError>
where
    Drive: FnOnce(&Cancel),
{
    let cancel = engine.cancel_handle();
    let (result_sender, result_receiver) = mpsc::channel();

    thread::spawn(move || {
        result_sender.send(engine.run()).ok();
    });

    drive(&cancel);

    match result_receiver.recv_timeout(WATCHDOG) {
        Ok(result) => result,
        Err(_) => panic!("{property}: the engine run did not finish within the watchdog timeout"),
    }
}

/// Asserts every fixture stage has been dropped, which it only is on the way out of the thread
/// that owned it.
pub(crate) fn assert_every_thread_exited(property: &str) {
    let built = STAGES_BUILT.load(Ordering::SeqCst);
    let dropped = STAGES_DROPPED.load(Ordering::SeqCst);

    assert_eq!(
        built, dropped,
        "{property}: {built} stages were built and {dropped} dropped, so a thread is still holding one",
    );
}

pub(crate) fn recorded() -> Vec<u64> {
    RECORDED
        .lock()
        .expect("not poisoned")
        .iter()
        .map(|(timestamp, _)| *timestamp)
        .collect()
}

/// The sibling of [recorded], reading the index half of the same recorded pairs.
pub(crate) fn recorded_indices() -> Vec<u64> {
    RECORDED
        .lock()
        .expect("not poisoned")
        .iter()
        .map(|(_, index)| *index)
        .collect()
}

pub(crate) fn clear_recorded() {
    RECORDED.lock().expect("not poisoned").clear();
}

/// How many distinct fixture instances processed at least one frame, which is the width the run
/// actually achieved.
pub(crate) fn achieved_width() -> usize {
    let assigned = ASSIGNED.lock().expect("not poisoned");
    let instances: HashSet<u64> = assigned.iter().map(|(instance, _)| *instance).collect();

    instances.len()
}

pub(crate) fn clear_assigned() {
    ASSIGNED.lock().expect("not poisoned").clear();
}

/// Every pair a `Matching` instance was given, in the order it was given them.
pub(crate) fn matched() -> Vec<(u64, u64)> {
    MATCHED.lock().expect("not poisoned").clone()
}

pub(crate) fn clear_matched() {
    MATCHED.lock().expect("not poisoned").clear();
}
