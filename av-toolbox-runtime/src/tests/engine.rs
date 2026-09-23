use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;
use std::time::Duration;

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Mask,
    MaskDefinition,
    Parallelism,
    PipelineRegistry,
    Sink,
    SinkDefinition,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
    StageKind,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourInfo,
    FormatSet,
    Frame,
    FrameBudget,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Mask as GpuMask,
    MaskFormat,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamItem,
    StreamTemplate,
    Subsampling,
    default_accelerators,
    testing_context,
};
use av_toolbox_graph::{
    CompiledGraph,
    Diagnostics,
    InputRef,
    NodeId,
    Region,
    SolvedFormats,
    Span,
    Spanned,
    StageBox,
    StageSpec,
    build,
    plan,
    solve,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use crate::engine::{AggregateSide, WorkerFreedom, run_aggregate};
use crate::{
    Aggregate,
    Assignment,
    Cancel,
    Engine,
    ExchangeError,
    RunError,
    RunReport,
    Settings,
    WorkResult,
};

pub(super) fn format() -> FrameFormat {
    FrameFormat {
        width: 64,
        height: 64,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
    }
}

/// A device-free set of capabilities, exactly as [testing_context] builds its own, so
/// `solve` never has to touch a real backend either.
fn testing_capabilities() -> Capabilities {
    Capabilities::builder(default_accelerators()[0])
        .plane_size(16, 16)
        .max_units_per_cube(1024)
        .max_cube_count(65535, 65535, 65535)
        .build()
}

// Every static below is process wide, not per test, so it is only safe because the `Justfile`
// runs `cargo nextest run`, which gives each test its own process. Running this file's tests
// through plain `cargo test` instead puts them all in one process and one set of statics, and
// they will race each other.

/// How many fixture stages were built, and how many have since been dropped.
///
/// Every stage belongs to exactly one thread of a run and is dropped as that thread's own frame
/// goes away, so equal counts once a run has returned mean no thread is still holding one.
static STAGES_BUILT: AtomicU32 = AtomicU32::new(0);
static STAGES_DROPPED: AtomicU32 = AtomicU32::new(0);

/// Counts one fixture stage from the moment it is built to the moment it is dropped.
struct StageLife;

impl StageLife {
    fn new() -> Self {
        STAGES_BUILT.fetch_add(1, Ordering::SeqCst);
        Self
    }
}

impl Drop for StageLife {
    fn drop(&mut self) {
        STAGES_DROPPED.fetch_add(1, Ordering::SeqCst);
    }
}

// A source that plays back a fixed list of chunk ids, one frame each, timestamped by the
// position it was built at, so a test can tell frames apart after the aggregate renumbers
// their indices.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct ListedArgs {
    #[serde(default)]
    chunks: Vec<u64>,
    #[serde(default)]
    id: u32,
    /// The frame index the first frame carries, so a test can tell a stream that was renumbered
    /// from one that was not.
    #[serde(default)]
    index_base: u64,
}

/// The frame budget limit every `Listed` source instance saw, by its `id`.
///
/// Read only after a run, since it is written from inside `start_of_stream` on the source's own
/// worker thread.
static SOURCE_BUDGETS: Mutex<Vec<(u32, Option<usize>)>> = Mutex::new(Vec::new());

/// Signalled once, from inside `next`, the moment a `Listed` source discovers it has nothing
/// left. By then every real frame has already been forwarded to every downstream consumer, from
/// the loop iteration before this one, so a test reading this signal knows production is
/// completely finished without guessing at timing.
static SOURCE_DONE: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Signalled once, from inside `next`, just before a `Listed` source allocates a frame that its
/// own budget's current state says would have to wait for room. Checked against that state
/// rather than assumed from a fixed position, so it fires at whatever allocation genuinely
/// saturates a share of any size, not only the smallest one. The budget's release is sticky, so
/// a test cancelling on this signal is right whether or not this thread has reached the wait yet.
pub(super) static SOURCE_WAITING: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// The highest outstanding byte count any `Listed` source has observed on its own budget,
/// sampled right after it charges each frame.
pub(super) static SOURCE_BUDGET_PEAK: AtomicUsize = AtomicUsize::new(0);

fn record_budget_peak(outstanding: usize) {
    SOURCE_BUDGET_PEAK.fetch_max(outstanding, Ordering::SeqCst);
}

/// Whether allocating `bytes` against `budget` right now would have to wait for room, mirroring
/// the rule `FrameBudget::wait_for_room` applies internally.
fn would_wait_for_room(budget: &FrameBudget, bytes: usize) -> bool {
    let outstanding = budget.outstanding();
    outstanding != 0 && outstanding + bytes > budget.limit()
}

struct ListedInstance {
    _life: StageLife,
    remaining: VecDeque<u64>,
    position: u64,
    index_base: u64,
    id: u32,
}

impl ListedInstance {
    fn from_args(args: ListedArgs) -> Self {
        Self {
            _life: StageLife::new(),
            remaining: args.chunks.into_iter().collect(),
            position: 0,
            index_base: args.index_base,
            id: args.id,
        }
    }
}

impl StageInstance for ListedInstance {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        let limit = context.budget().map(|budget| budget.limit());
        SOURCE_BUDGETS
            .lock()
            .expect("not poisoned")
            .push((self.id, limit));
        Ok(())
    }
}

impl Source for ListedInstance {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        let Some(chunk) = self.remaining.pop_front() else {
            if let Some(sender) = SOURCE_DONE.lock().expect("not poisoned").take() {
                sender.send(()).ok();
            }
            return Ok(None);
        };

        let mut meta = FrameMeta::at(self.index_base + self.position).in_chunk(chunk);
        meta.timestamp = Some(Duration::from_secs(self.position));

        if let Some(budget) = context.budget()
            && would_wait_for_room(budget, format().total_bytes())
            && let Some(sender) = SOURCE_WAITING.lock().expect("not poisoned").take()
        {
            sender.send(()).ok();
        }

        self.position += 1;

        let frame = context.alloc_frame(&format(), meta);
        if let Some(budget) = context.budget() {
            record_budget_peak(budget.outstanding());
        }

        Ok(Some(frame))
    }
}

struct Listed;

impl StageDefinition for Listed {
    type Args = ListedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "listed"
    }

    // Per frame, like `av-toolbox-graph`'s own `WideningSource` fixture, so a widened source
    // reaches the planner's width decision rather than being narrowed by its granularity before
    // the engine ever sees it. `a_widened_source_is_rejected` needs exactly that to exercise the
    // engine's own refusal rather than the planner's separate, earlier one.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Listed {
    fn produces(_capabilities: &Capabilities, _args: &Self::Args) -> Result<OutputFormats, StageError> {
        let stream_format = StreamFormat::Frames(format());

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&stream_format),
            samples: FormatSet::of([stream_format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(ListedInstance::from_args(args)))
    }
}

// A filter that passes its one input through unchanged.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct PassArgs {}

/// How many times a `Pass` instance was built, across every worker of every region.
static PASS_BUILD_CALLS: AtomicU32 = AtomicU32::new(0);

/// Whether the last `Pass` instance started saw a budget on its context.
static PASS_HAD_BUDGET: Mutex<Option<bool>> = Mutex::new(None);

/// How many times a `Pass` instance was told a stream started, and how many times one ended,
/// across every worker of every region.
static PASS_STARTS: AtomicU32 = AtomicU32::new(0);
static PASS_ENDS: AtomicU32 = AtomicU32::new(0);

struct PassInstance {
    _life: StageLife,
}

impl PassInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for PassInstance {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        *PASS_HAD_BUDGET.lock().expect("not poisoned") = Some(context.budget().is_some());
        PASS_STARTS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

impl Filter for PassInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        PASS_ENDS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }
}

struct Pass;

impl StageDefinition for Pass {
    type Args = PassArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "pass"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Pass {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        PASS_BUILD_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(PassInstance::new()))
    }
}

// A filter that allocates its output instead of passing its input through, standing in for every
// real filter that writes somewhere new. A region of several of these holds one frame per node at
// once, all of them charged to whichever budget the region's context carries.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct AllocatingArgs {}

struct AllocatingInstance {
    _life: StageLife,
}

impl AllocatingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for AllocatingInstance {}

impl Filter for AllocatingInstance {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let input = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(context.alloc_frame(&format(), *input.meta()))
    }
}

struct Allocating;

impl StageDefinition for Allocating {
    type Args = AllocatingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "allocating"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Allocating {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(AllocatingInstance::new()))
    }
}

// A mask stage, whose output is far larger than any frame in these tests, so a budget that only
// measures frames is visibly too small for it.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct MaskingArgs {}

pub(super) fn mask_format() -> MaskFormat {
    MaskFormat {
        width: 512,
        height: 512,
        sample: SampleFormat::U8,
    }
}

struct MaskingInstance {
    _life: StageLife,
}

impl MaskingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for MaskingInstance {}

impl Mask for MaskingInstance {
    fn select(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<GpuMask, StageError> {
        let input = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(context.alloc_mask(&mask_format(), *input.meta()))
    }
}

struct Masking;

impl StageDefinition for Masking {
    type Args = MaskingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "masking"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl MaskDefinition for Masking {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::Mask(mask_format()),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Mask>, StageError> {
        Ok(Box::new(MaskingInstance::new()))
    }
}

// A filter that reads two inputs and passes the first through, standing in for a mask-and-frame
// filter, an overlay, or a composite: any two input stage whose producers can end up in
// different regions.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct JoinArgs {}

struct JoinInstance {
    _life: StageLife,
}

impl JoinInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for JoinInstance {}

impl Filter for JoinInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let base = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

        Ok(base)
    }
}

struct Join;

impl StageDefinition for Join {
    type Args = JoinArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "join"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base").accepting(StreamType::Frames),
            InputParam::required("overlay").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Join {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(JoinInstance::new()))
    }
}

// A filter that passes its input through, but whose first call blocks until a test releases it.
// Standing in for a producer whose path to a join is genuinely, deliberately slower than the
// path the other side of that join takes, rather than merely one thread hop.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct SlowArgs {}

/// Taken by the `Slow` worker of the matching index. `None` means run unblocked, which is every
/// worker of every test that never sets one.
pub(super) static SLOW_GATES: [Mutex<Option<mpsc::Receiver<()>>>; 2] = [Mutex::new(None), Mutex::new(None)];

/// Signalled by a `Slow` worker the moment before it waits on its gate.
pub(super) static SLOW_PARKED: [Mutex<Option<mpsc::Sender<()>>>; 2] = [Mutex::new(None), Mutex::new(None)];

struct SlowInstance {
    _life: StageLife,
    parked: Option<mpsc::Sender<()>>,
    gate: Option<mpsc::Receiver<()>>,
}

impl SlowInstance {
    /// An instance nothing holds back, for the registry build path no test drives.
    fn unblocked() -> Self {
        Self {
            _life: StageLife::new(),
            parked: None,
            gate: None,
        }
    }

    /// The instance for `worker`, holding whichever gate and signal that worker was given.
    fn for_worker(worker: usize) -> Self {
        let parked = SLOW_PARKED
            .get(worker)
            .and_then(|parked| parked.lock().expect("not poisoned").take());
        let gate = SLOW_GATES
            .get(worker)
            .and_then(|gate| gate.lock().expect("not poisoned").take());

        Self {
            _life: StageLife::new(),
            parked,
            gate,
        }
    }
}

impl StageInstance for SlowInstance {}

impl Filter for SlowInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        if let Some(parked) = self.parked.take() {
            parked.send(()).ok();
        }

        if let Some(gate) = self.gate.take() {
            gate.recv().ok();
        }

        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

struct Slow;

impl StageDefinition for Slow {
    type Args = SlowArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "slow"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Slow {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(SlowInstance::unblocked()))
    }
}

// A sink that records the timestamp of every frame it consumes, in the order it consumed them.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct RecordingArgs {}

/// Every timestamp a `Recording` sink consumed, across the whole run, in consumption order.
static RECORDED: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// The frame index of every item a `Recording` sink consumed, in the same order.
///
/// A timestamp is the source's own and survives everything, which is what makes it the right
/// thing to check a stream's contents against. An index is renumbered by every aggregate it
/// passes through, which is what makes it the right thing to check renumbering against.
static RECORDED_INDICES: Mutex<Vec<u64>> = Mutex::new(Vec::new());

/// Signalled once, when a `Recording` sink is told its stream has ended. Its region only reaches
/// that once whatever feeds it has closed the stream, so a test waiting here knows the region
/// upstream has finished, however it finished.
static RECORDING_ENDED: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

struct RecordingInstance {
    _life: StageLife,
}

impl RecordingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for RecordingInstance {}

impl Sink for RecordingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
        let timestamp = item
            .meta()
            .timestamp
            .expect("every test frame carries a timestamp")
            .as_secs();

        RECORDED.lock().expect("not poisoned").push(timestamp);
        RECORDED_INDICES
            .lock()
            .expect("not poisoned")
            .push(item.meta().index);

        Ok(())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        if let Some(sender) = RECORDING_ENDED.lock().expect("not poisoned").take() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

struct Recording;

impl StageDefinition for Recording {
    type Args = RecordingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "recording"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for Recording {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(RecordingInstance::new()))
    }
}

// A filter that fails on a chosen call, and that a test can hold back until it says so, which
// is what lets two failures in one run be ordered against each other rather than raced.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct FailingArgs {
    /// Which of the two gate, signal and counter sets this stage uses.
    #[serde(default)]
    id: usize,
    /// The call this stage starts failing on, counting every call to it across every worker.
    /// Zero never fails.
    #[serde(default)]
    fail_at: u32,
    /// The only worker that fails, when a test needs one worker to fail while another carries on.
    /// Every worker fails when this is unset.
    #[serde(default)]
    failing_worker: Option<usize>,
}

/// How many times each `Failing` stage has been called, by its `id`.
static FAILING_CALLS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

/// Taken by the worker of the matching stage and index. A worker holding one waits there on its
/// first call, before it decides whether to fail, so a test can hold back one worker of a widened
/// stage, one of two separate stages, or every one of them.
static FAILING_GATES: [[Mutex<Option<mpsc::Receiver<()>>>; 2]; 2] = [
    [Mutex::new(None), Mutex::new(None)],
    [Mutex::new(None), Mutex::new(None)],
];

/// Signalled by a `Failing` worker the moment before it waits on its gate.
static FAILING_PARKED: [[Mutex<Option<mpsc::Sender<()>>>; 2]; 2] = [
    [Mutex::new(None), Mutex::new(None)],
    [Mutex::new(None), Mutex::new(None)],
];

/// How many times each worker of a `Failing` stage was told a stream started, and how many times
/// one ended.
static FAILING_STARTS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
static FAILING_ENDS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

struct FailingInstance {
    _life: StageLife,
    id: usize,
    worker: usize,
    fail_at: u32,
    fails: bool,
    parked: Option<mpsc::Sender<()>>,
    gate: Option<mpsc::Receiver<()>>,
}

impl FailingInstance {
    /// The instance for `worker` of the stage `args` names.
    fn for_worker(args: FailingArgs, worker: usize) -> Self {
        let parked = FAILING_PARKED[args.id]
            .get(worker)
            .and_then(|parked| parked.lock().expect("not poisoned").take());
        let gate = FAILING_GATES[args.id]
            .get(worker)
            .and_then(|gate| gate.lock().expect("not poisoned").take());

        Self {
            _life: StageLife::new(),
            id: args.id,
            worker,
            fail_at: args.fail_at,
            fails: args.failing_worker.is_none_or(|only| only == worker),
            parked,
            gate,
        }
    }
}

impl StageInstance for FailingInstance {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        if let Some(starts) = FAILING_STARTS.get(self.worker) {
            starts.fetch_add(1, Ordering::SeqCst);
        }

        Ok(())
    }
}

impl Filter for FailingInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        if let Some(parked) = self.parked.take() {
            parked.send(()).ok();
        }

        if let Some(gate) = self.gate.take() {
            gate.recv().ok();
        }

        let call = FAILING_CALLS[self.id].fetch_add(1, Ordering::SeqCst) + 1;
        if self.fails && self.fail_at > 0 && call >= self.fail_at {
            return Err(StageError::msg(
                "the failing filter was asked for one call too many",
            ));
        }

        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        if let Some(ends) = FAILING_ENDS.get(self.worker) {
            ends.fetch_add(1, Ordering::SeqCst);
        }

        Ok(Vec::new())
    }
}

struct Failing;

impl StageDefinition for Failing {
    type Args = FailingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "failing"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Failing {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(FailingInstance::for_worker(args, usize::MAX)))
    }
}

// A filter that panics rather than returning, standing in for a stage with a bug in it.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct PanickingArgs {}

struct PanickingInstance {
    _life: StageLife,
}

impl PanickingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for PanickingInstance {}

impl Filter for PanickingInstance {
    fn process(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<Frame, StageError> {
        panic!("the panicking filter panicked");
    }
}

struct Panicking;

impl StageDefinition for Panicking {
    type Args = PanickingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "panicking"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Panicking {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = chosen.iter().flatten().next().copied().expect("an input");

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PanickingInstance::new()))
    }
}

// A sink that holds on to every item it is given, so nothing it consumes is ever credited back
// to the budget the source charged it to.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct HoardingArgs {}

/// Every item a `Hoarding` sink consumed, still alive.
static HOARDED: Mutex<Vec<StreamItem>> = Mutex::new(Vec::new());

/// Signalled once, the first time a `Hoarding` sink takes an item. From then on that item is
/// never dropped, so whatever budget it was charged to is never credited back.
static HOARDED_ONE: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

struct HoardingInstance {
    _life: StageLife,
}

impl HoardingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for HoardingInstance {}

impl Sink for HoardingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        HOARDED.lock().expect("not poisoned").push(item);

        if let Some(sender) = HOARDED_ONE.lock().expect("not poisoned").take() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

struct Hoarding;

impl StageDefinition for Hoarding {
    type Args = HoardingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "hoarding"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for Hoarding {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(HoardingInstance::new()))
    }
}

// A sink that reads two inputs, so one step of it consumes two items at once.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct PairingArgs {}

/// How many times a `Pairing` sink was asked to consume, which is half of what it consumed.
static PAIRING_STEPS: AtomicU32 = AtomicU32::new(0);

struct PairingInstance {
    _life: StageLife,
}

impl PairingInstance {
    fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for PairingInstance {}

impl Sink for PairingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
        inputs
            .take(1)
            .ok_or_else(|| StageError::msg("expected an item in slot 1"))?;

        PAIRING_STEPS.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct Pairing;

impl StageDefinition for Pairing {
    type Args = PairingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "pairing"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("base"), InputParam::required("overlay")]);
        SIGNATURE
    }
}

impl SinkDefinition for Pairing {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(PairingInstance::new()))
    }
}

// A sink that takes one item at a time and holds each one until the test releases it, so a
// source's budget only gets its bytes back on the test's own schedule.
#[derive(Debug, Default, Deserialize, JsonSchema)]
struct GatedArgs {}

/// Taken once, by the `Gated` sink built for a run. Sent after every item is taken, before that
/// item is held for release.
pub(super) static GATED_GOT: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Taken once, by the `Gated` sink built for a run. One receive per item, so a test releases
/// them one at a time.
pub(super) static GATED_RELEASE: Mutex<Option<mpsc::Receiver<()>>> = Mutex::new(None);

struct GatedInstance {
    _life: StageLife,
    got: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl GatedInstance {
    fn new() -> Self {
        let got = GATED_GOT
            .lock()
            .expect("not poisoned")
            .take()
            .expect("a test sets the gated sender before building the engine");
        let release = GATED_RELEASE
            .lock()
            .expect("not poisoned")
            .take()
            .expect("a test sets the gated receiver before building the engine");

        Self {
            _life: StageLife::new(),
            got,
            release,
        }
    }
}

impl StageInstance for GatedInstance {}

impl Sink for GatedInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<(), StageError> {
        let _item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;

        self.got.send(()).ok();
        self.release.recv().ok();

        Ok(())
    }
}

struct Gated;

impl StageDefinition for Gated {
    type Args = GatedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "gated"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for Gated {
    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([offered.sample()]))
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(GatedInstance::new()))
    }
}

fn registry() -> PipelineRegistry {
    let mut registry = PipelineRegistry::new();
    registry.register_source::<Listed>().expect("should register");
    registry.register_filter::<Pass>().expect("should register");
    registry.register_filter::<Allocating>().expect("should register");
    registry.register_mask::<Masking>().expect("should register");
    registry.register_filter::<Join>().expect("should register");
    registry.register_filter::<Slow>().expect("should register");
    registry.register_filter::<Failing>().expect("should register");
    registry.register_filter::<Panicking>().expect("should register");
    registry.register_sink::<Recording>().expect("should register");
    registry.register_sink::<Hoarding>().expect("should register");
    registry.register_sink::<Pairing>().expect("should register");
    registry.register_sink::<Gated>().expect("should register");
    registry
}

/// Builds one node's stage directly from its identifier, the same way
/// [crate::worker::region]'s own tests build a [StageBox] by hand.
///
/// This is what [Engine::build_with] calls instead of `instantiate_region`, so the whole engine
/// runs against these fixtures with no backend and no registry lookup at run time. `worker` is
/// this region's own worker index, which only `test/slow` reads: the first idle dispatch policy
/// always hands a region's very first chunk to its worker `0`, so gating worker `0` alone is
/// enough to hold up a whole widened branch until a test releases it.
fn build_stage(graph: &CompiledGraph, id: NodeId, worker: usize) -> (NodeId, StageBox) {
    let node = graph.node(id);

    let stage = match node.identifier() {
        "test/listed" => {
            let args: ListedArgs = serde_json::from_value(node.args().clone()).expect("valid listed args");
            StageBox::Source(Box::new(ListedInstance::from_args(args)))
        },
        "test/pass" => {
            PASS_BUILD_CALLS.fetch_add(1, Ordering::SeqCst);
            StageBox::Filter(Box::new(PassInstance::new()))
        },
        "test/allocating" => StageBox::Filter(Box::new(AllocatingInstance::new())),
        "test/masking" => StageBox::Mask(Box::new(MaskingInstance::new())),
        "test/join" => StageBox::Filter(Box::new(JoinInstance::new())),
        "test/slow" => StageBox::Filter(Box::new(SlowInstance::for_worker(worker))),
        "test/failing" => {
            let args: FailingArgs = serde_json::from_value(node.args().clone()).expect("valid failing args");
            StageBox::Filter(Box::new(FailingInstance::for_worker(args, worker)))
        },
        "test/panicking" => StageBox::Filter(Box::new(PanickingInstance::new())),
        "test/recording" => StageBox::Sink(Box::new(RecordingInstance::new())),
        "test/hoarding" => StageBox::Sink(Box::new(HoardingInstance::new())),
        "test/pairing" => StageBox::Sink(Box::new(PairingInstance::new())),
        "test/gated" => StageBox::Sink(Box::new(GatedInstance::new())),
        other => panic!("no test fixture knows how to build {other}"),
    };

    (id, stage)
}

fn instantiate(
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

fn spec(kind: StageKind, name: &str, stage_type: &str) -> StageSpec {
    StageSpec {
        name: Spanned::unknown(name.to_string()),
        kind,
        stage_type: Spanned::unknown(stage_type.to_string()),
        inputs: Vec::new(),
        args: Spanned::unknown(json!({})),
        workers: None,
        span: Span::unknown(),
    }
}

fn positional(target: &str) -> Spanned<InputRef> {
    Spanned::unknown(InputRef {
        name: None,
        target: target.to_string(),
    })
}

pub(super) fn source_spec(name: &str, chunks: &[u64], id: u32) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id }));
    source
}

/// A source whose frame indices start at `index_base` rather than zero.
fn offset_source_spec(name: &str, chunks: &[u64], id: u32, index_base: u64) -> StageSpec {
    let mut source = spec(StageKind::Source, name, "test/listed");
    source.args = Spanned::unknown(json!({ "chunks": chunks, "id": id, "index_base": index_base }));
    source
}

/// A source declared to run wider than one worker, which this engine has to refuse.
fn widened_source_spec(name: &str, chunks: &[u64], id: u32, workers: usize) -> StageSpec {
    let mut source = source_spec(name, chunks, id);
    source.workers = Some(workers);
    source
}

pub(super) fn filter_spec(name: &str, input: &str, workers: Option<usize>) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/pass");
    filter.inputs = vec![positional(input)];
    filter.workers = workers;
    filter
}

/// A filter that allocates its own output rather than passing its input through.
fn allocating_spec(name: &str, input: &str) -> StageSpec {
    let mut filter = spec(StageKind::Filter, name, "test/allocating");
    filter.inputs = vec![positional(input)];
    filter
}

fn masking_spec(name: &str, input: &str) -> StageSpec {
    let mut mask = spec(StageKind::Mask, name, "test/masking");
    mask.inputs = vec![positional(input)];
    mask
}

pub(super) fn join_spec(name: &str, base: &str, overlay: &str) -> StageSpec {
    let mut join = spec(StageKind::Filter, name, "test/join");
    join.inputs = vec![positional(base), positional(overlay)];
    join
}

pub(super) fn slow_spec(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut slow = spec(StageKind::Filter, name, "test/slow");
    slow.inputs = vec![positional(input)];
    slow.workers = Some(workers);
    slow
}

fn failing_spec(name: &str, input: &str, id: usize, fail_at: u32, workers: Option<usize>) -> StageSpec {
    let mut failing = spec(StageKind::Filter, name, "test/failing");
    failing.inputs = vec![positional(input)];
    failing.args = Spanned::unknown(json!({ "id": id, "fail_at": fail_at }));
    failing.workers = workers;
    failing
}

/// A widened failing stage where only `failing_worker` ever fails, on its own first call.
fn failing_worker_spec(
    name: &str,
    input: &str,
    id: usize,
    failing_worker: usize,
    workers: usize,
) -> StageSpec {
    let mut failing = spec(StageKind::Filter, name, "test/failing");
    failing.inputs = vec![positional(input)];
    failing.args = Spanned::unknown(json!({ "id": id, "fail_at": 1, "failing_worker": failing_worker }));
    failing.workers = Some(workers);
    failing
}

fn panicking_spec(name: &str, input: &str, workers: usize) -> StageSpec {
    let mut panicking = spec(StageKind::Filter, name, "test/panicking");
    panicking.inputs = vec![positional(input)];
    panicking.workers = Some(workers);
    panicking
}

fn hoarding_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/hoarding");
    sink.inputs = vec![positional(input)];
    sink
}

/// A sink reading two inputs, so one step of it consumes two items.
fn pairing_spec(name: &str, base: &str, overlay: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/pairing");
    sink.inputs = vec![positional(base), positional(overlay)];
    sink
}

pub(super) fn sink_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/recording");
    sink.inputs = vec![positional(input)];
    sink
}

pub(super) fn gated_spec(name: &str, input: &str) -> StageSpec {
    let mut sink = spec(StageKind::Sink, name, "test/gated");
    sink.inputs = vec![positional(input)];
    sink
}

/// Builds and plans `specs` into a graph carrying a solved [av_toolbox_graph::ExecutionPlan],
/// without ever touching a backend.
fn planned_graph(specs: &[StageSpec]) -> (CompiledGraph, SolvedFormats) {
    let registry = registry();
    let mut graph = build(&registry, specs).expect("should build").into_graph();

    let made_plan = plan(&graph, &registry).expect("should plan");
    graph.set_execution_plan(made_plan);

    let capabilities = testing_capabilities();
    let solved = solve(&graph, &registry, &capabilities).expect("should solve");

    (graph, solved)
}

/// Builds, plans, solves and instantiates an engine from `specs`, entirely through the fake
/// allocator seam: no `PipelineRegistry::prepare`-driven build, no backend.
pub(super) fn build_engine(specs: &[StageSpec], settings: Settings) -> Engine {
    let (graph, solved) = planned_graph(specs);

    engine_from(&graph, &solved, settings)
}

/// The same, for a test that needs the graph itself as well, to name the node it asserts on.
fn engine_from(graph: &CompiledGraph, solved: &SolvedFormats, settings: Settings) -> Engine {
    Engine::build_with(
        graph,
        solved,
        settings,
        |region, worker| instantiate(graph, region, worker),
        testing_context,
    )
    .expect("should build the engine")
}

fn node_id(graph: &CompiledGraph, name: &str) -> NodeId {
    graph
        .nodes()
        .iter()
        .find(|node| node.name() == name)
        .map(|node| node.id())
        .unwrap_or_else(|| panic!("the graph has no node named {name}"))
}

/// How long any of these tests will wait for a run before calling it hung.
pub(super) const WATCHDOG: Duration = Duration::from_secs(10);

/// Runs `engine` on its own thread and waits for it with a generous timeout, so a design mistake
/// that hangs the exchange reports as a named failure instead of a stalled suite.
pub(super) fn run_with_watchdog(engine: Engine, property: &str) -> RunReport {
    match run_result_with_watchdog(engine, property) {
        Ok(report) => report,
        Err(error) => panic!("{property}: the engine run failed: {error}"),
    }
}

/// The same, handing back whatever the run answered with, for a test about a run that fails.
fn run_result_with_watchdog(engine: Engine, property: &str) -> Result<RunReport, RunError> {
    run_driven_with_watchdog(engine, property, |_cancel| {})
}

/// The same again, with `drive` running on the test's own thread while the engine runs.
///
/// A test that cancels part way through does its waiting and its cancelling in there, given the
/// signal that stops this run. Everything `drive` waits for is a channel, never a delay.
pub(super) fn run_driven_with_watchdog<Drive>(
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
pub(super) fn assert_every_thread_exited(property: &str) {
    let built = STAGES_BUILT.load(Ordering::SeqCst);
    let dropped = STAGES_DROPPED.load(Ordering::SeqCst);

    assert_eq!(
        built, dropped,
        "{property}: {built} stages were built and {dropped} dropped, so a thread is still holding one",
    );
}

pub(super) fn recorded() -> Vec<u64> {
    RECORDED.lock().expect("not poisoned").clone()
}

fn recorded_indices() -> Vec<u64> {
    RECORDED_INDICES.lock().expect("not poisoned").clone()
}

pub(super) fn clear_recorded() {
    RECORDED.lock().expect("not poisoned").clear();
    RECORDED_INDICES.lock().expect("not poisoned").clear();
}

#[test]
fn a_single_region_graph_runs_end_to_end() {
    clear_recorded();

    let specs = vec![
        source_spec("input", &[0, 0, 0], 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "a_single_region_graph_runs_end_to_end");

    assert_eq!(recorded(), vec![0, 1, 2]);
    assert_eq!(report.frames, 3);
}

#[test]
fn a_widened_region_runs_every_frame_exactly_once() {
    clear_recorded();

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_widened_region_runs_every_frame_exactly_once");

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..8).collect::<Vec<u64>>());
}

#[test]
fn output_comes_back_in_input_order_at_any_width() {
    let chunks = vec![0, 0, 1, 1, 2, 2, 3];

    clear_recorded();
    let narrow_specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", None),
        sink_spec("out", "filter"),
    ];
    let narrow = build_engine(&narrow_specs, Settings::default());
    run_with_watchdog(
        narrow,
        "output_comes_back_in_input_order_at_any_width (width one)",
    );
    let narrow_order = recorded();

    clear_recorded();
    let wide_specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let wide = build_engine(&wide_specs, Settings::default());
    run_with_watchdog(wide, "output_comes_back_in_input_order_at_any_width (width four)");
    let wide_order = recorded();

    assert_eq!(narrow_order, vec![0, 1, 2, 3, 4, 5, 6]);
    assert_eq!(narrow_order, wide_order);
}

#[test]
fn every_worker_of_a_region_gets_its_own_instances() {
    PASS_BUILD_CALLS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "every_worker_of_a_region_gets_its_own_instances");

    assert_eq!(PASS_BUILD_CALLS.load(Ordering::SeqCst), 4);
}

#[test]
fn a_source_gets_its_own_share_of_the_budget() {
    SOURCE_BUDGETS.lock().expect("not poisoned").clear();

    let specs = vec![
        source_spec("left", &[], 0),
        sink_spec("left_out", "left"),
        source_spec("right", &[], 1),
        sink_spec("right_out", "right"),
    ];
    let engine = build_engine(
        &specs,
        Settings {
            frame_budget: Some(1000),
            ..Settings::default()
        },
    );

    run_with_watchdog(engine, "a_source_gets_its_own_share_of_the_budget");

    let recorded = SOURCE_BUDGETS.lock().expect("not poisoned").clone();
    let left = recorded
        .iter()
        .find(|(id, _)| *id == 0)
        .map(|(_, limit)| *limit)
        .expect("the left source should have started");
    let right = recorded
        .iter()
        .find(|(id, _)| *id == 1)
        .map(|(_, limit)| *limit)
        .expect("the right source should have started");

    assert_eq!(left, Some(500));
    assert_eq!(right, Some(500));
}

#[test]
fn a_worker_that_is_not_a_source_has_no_budget() {
    *PASS_HAD_BUDGET.lock().expect("not poisoned") = None;

    // Widened so the filter lands in a region of its own, separate from the source's, since a
    // filter left at the source's own width would share its worker's context and therefore its
    // budget, proving nothing about a non-source worker.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_worker_that_is_not_a_source_has_no_budget");

    let had_budget = PASS_HAD_BUDGET
        .lock()
        .expect("not poisoned")
        .expect("the filter should have started");

    assert!(!had_budget);
}

#[test]
fn a_graph_with_no_frames_still_finishes() {
    clear_recorded();

    // Widened, so the filter lands in a region with a disperse, and this run reaches
    // `Disperse::finish` with no chunk ever having been in flight. Left at width one, all
    // three nodes collapse into a single region with no disperse at all, and never touches
    // that path.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter", "input", Some(2)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, "a_graph_with_no_frames_still_finishes");

    assert!(recorded().is_empty());
    assert_eq!(report.frames, 0);
}

#[test]
fn a_widened_source_is_rejected() {
    let specs = vec![widened_source_spec("input", &[], 0, 4), sink_spec("out", "input")];
    let (graph, solved) = planned_graph(&specs);

    let result = Engine::build_with(
        &graph,
        &solved,
        Settings::default(),
        |region, worker| instantiate(&graph, region, worker),
        testing_context,
    );

    let Err(diagnostics) = result else {
        panic!("a source asked to run wider than one worker should have been refused");
    };
    // "workers" rather than the stage's own name "input", which would also match a message
    // about literally anything else that happened to mention the stage.
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("workers"))
    );
}

#[test]
fn a_region_with_more_than_one_exit_is_rejected() {
    // `filter1` and `filter2` chain off the source at width one, so they join its region. Each
    // is then read by its own widened filter, forcing both to be exits of that one region, which
    // this engine does not know how to give one aggregate each.
    let specs = vec![
        source_spec("input", &[], 0),
        filter_spec("filter1", "input", None),
        filter_spec("filter2", "filter1", None),
        filter_spec("filter3", "filter1", Some(4)),
        filter_spec("filter4", "filter2", Some(4)),
        sink_spec("sink3", "filter3"),
        sink_spec("sink4", "filter4"),
    ];
    let (graph, solved) = planned_graph(&specs);

    let result = Engine::build_with(
        &graph,
        &solved,
        Settings::default(),
        |region, worker| instantiate(&graph, region, worker),
        testing_context,
    );

    let Err(diagnostics) = result else {
        panic!("a region with two exits should have been refused");
    };
    // "exits" rather than just `has_errors()`, which would also pass if the build failed for
    // any unrelated reason.
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message().contains("exits"))
    );
}

/// A source, a widened branch off it, and a two input join reading both directly.
///
/// `join`'s two inputs sit in different regions (the source's own, and the branch's widened
/// one), so its disperse has two bound slots fed by two independent upstream channels. Before
/// the fix, offering each message the instant it arrived meant a fast slot receiving a second
/// item before its sibling received a first tripped `Disperse`'s own one-pending-item-per-slot
/// limit: `SlotRanAhead` on a single thread, or the same panic raised nondeterministically by
/// two threads racing into one `Disperse` behind a mutex before that. Either way it cascaded into
/// `Engine::run`'s own join panicking on the crashed thread.
///
/// The fix keeps its own queue per bound slot and only offers once every slot has something
/// waiting, so a fast slot queues rather than panics. This runs with the plan's own default
/// budget, `Settings::default()`, deliberately: it is what proves the fix does not
/// depend on a hand-picked size pinning the two paths together, the way the first attempt at
/// this test did.
#[test]
fn a_two_input_join_reads_both_producers_without_racing_the_disperse() {
    clear_recorded();

    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("branch", "input", Some(4)),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(
        engine,
        "a_two_input_join_reads_both_producers_without_racing_the_disperse",
    );

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..4).collect::<Vec<u64>>());
    assert_eq!(report.frames, 4);
}

/// The same shape, but `branch` is deliberately held back rather than merely slower by one
/// thread hop, so the fast slot genuinely accumulates several frames rather than at most one.
///
/// `branch`'s first worker blocks on [SLOW_GATE] until the source has finished producing and
/// forwarding every frame down both paths (signalled through [SOURCE_DONE], set the moment the
/// source discovers it has nothing left, which is always after the last real frame was already
/// forwarded). By the time the gate is released, every frame the source ever produced is
/// already queued against `join`'s direct slot, and none has reached its other slot at all.
/// Releasing the gate then lets `branch` drain its own backlog and catch up. Nothing here is
/// timed: both the hold and the release are channel sends, never a delay.
#[test]
fn a_slow_producer_still_joins_correctly_once_a_fast_ones_backlog_has_built_up() {
    clear_recorded();

    let (source_done_sender, source_done_receiver) = mpsc::channel();
    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    let (gate_sender, gate_receiver) = mpsc::channel();
    *SLOW_GATES[0].lock().expect("not poisoned") = Some(gate_receiver);

    let chunks: Vec<u64> = (0..6).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        slow_spec("branch", "input", 2),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let (run_result_sender, run_result_receiver) = mpsc::channel();
    thread::spawn(move || {
        run_result_sender.send(engine.run()).ok();
    });

    source_done_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("the source should finish producing before the gate is ever released");

    gate_sender.send(()).ok();

    let report = run_result_receiver
        .recv_timeout(Duration::from_secs(10))
        .expect("a_slow_producer_still_joins_correctly_once_a_fast_ones_backlog_has_built_up: the engine run did not finish within the watchdog timeout")
        .expect("the engine run failed");

    let mut seen = recorded();
    seen.sort_unstable();

    assert_eq!(seen, (0..6).collect::<Vec<u64>>());
    assert_eq!(report.frames, 6);
}

/// The failing filter is widened, so the stage that fails is one worker of four and the other
/// three are left waiting on work that is never coming.
#[test]
fn a_stage_failure_stops_the_run_and_names_its_node() {
    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 3, Some(4)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_stops_the_run_and_names_its_node");

    let Err(error) = result else {
        panic!("a stage that failed should have failed the run");
    };
    assert_eq!(error.node(), Some(failing), "the run reported {error}");
}

#[test]
fn a_stage_failure_joins_every_thread() {
    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 3, Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_joins_every_thread");

    assert!(result.is_err(), "a stage that failed should have failed the run");
    assert_every_thread_exited("a_stage_failure_joins_every_thread");
}

/// A budget of one byte and a sink that never lets go of a frame, so the source waits for room
/// that will never appear.
///
/// The test waits for the sink to take the first frame before it cancels, because a frame still
/// on its way somewhere is a frame that could yet be dropped, and dropping one credits the budget
/// and frees the source without any of this being involved. Once the sink is holding it, only
/// releasing the budget can get the source out of its wait.
///
/// The source signals before the allocation that finds the budget full. Nothing here depends on
/// the source having reached the wait by then, because releasing a budget is sticky rather than a
/// one-off nudge.
#[test]
fn a_run_cancelled_while_a_source_waits_on_the_budget_joins_every_thread() {
    let property = "a_run_cancelled_while_a_source_waits_on_the_budget_joins_every_thread";
    let (waiting_sender, waiting_receiver) = mpsc::channel();
    let (hoarded_sender, hoarded_receiver) = mpsc::channel();
    *SOURCE_WAITING.lock().expect("not poisoned") = Some(waiting_sender);
    *HOARDED_ONE.lock().expect("not poisoned") = Some(hoarded_sender);

    // Far more frames than this run will ever get through, so the source is still going when the
    // budget stops it.
    let chunks: Vec<u64> = (0..64).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(2)),
        hoarding_spec("out", "filter"),
    ];
    let engine = build_engine(
        &specs,
        Settings {
            frame_budget: Some(1),
            ..Settings::default()
        },
    );

    let result = run_driven_with_watchdog(engine, property, |cancel| {
        hoarded_receiver
            .recv_timeout(WATCHDOG)
            .expect("the sink should take the first frame and keep it");
        waiting_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should reach the frame that has to wait for room");

        cancel.stop();
    });

    let report = result.expect("a cancelled run is not a failed one");

    assert!(report.cancelled);
    assert_every_thread_exited(property);
}

#[test]
fn a_panicking_worker_becomes_an_error_rather_than_a_hang() {
    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        panicking_spec("filter", "input", 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, "a_panicking_worker_becomes_an_error_rather_than_a_hang");

    let Err(error) = result else {
        panic!("a stage that panicked should have failed the run");
    };
    assert!(
        matches!(error, RunError::Panicked { .. }),
        "the run reported {error}",
    );
}

#[test]
fn a_panicking_worker_still_lets_every_other_thread_be_joined() {
    let chunks: Vec<u64> = (0..4).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        panicking_spec("filter", "input", 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(
        engine,
        "a_panicking_worker_still_lets_every_other_thread_be_joined",
    );

    assert!(
        result.is_err(),
        "a stage that panicked should have failed the run"
    );
    assert_every_thread_exited("a_panicking_worker_still_lets_every_other_thread_be_joined");
}

/// Two stages that both fail, ordered against each other rather than raced.
///
/// `second` announces that it is parked and waits. Only then is `first` let go, so it fails while
/// `second` is already sitting inside its own call, and `second` is only released once the run has
/// stopped, which is something that only happens after the first failure has been recorded.
#[test]
fn the_first_error_is_the_one_reported() {
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();

    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_PARKED[1][0].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[1][0].lock().expect("not poisoned") = Some(second_gate_receiver);

    let specs = vec![
        source_spec("first_input", &[0, 0], 0),
        failing_spec("first", "first_input", 0, 1, None),
        sink_spec("first_out", "first"),
        source_spec("second_input", &[0, 0], 1),
        failing_spec("second", "second_input", 1, 1, None),
        sink_spec("second_out", "second"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let first = node_id(&graph, "first");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_driven_with_watchdog(engine, "the_first_error_is_the_one_reported", |cancel| {
        let (stopped_sender, stopped_receiver) = mpsc::channel();
        cancel.on_stop(move || {
            stopped_sender.send(()).ok();
        });

        parked_receiver
            .recv_timeout(WATCHDOG)
            .expect("the second stage should park before the first is let go");
        first_gate_sender.send(()).ok();

        stopped_receiver
            .recv_timeout(WATCHDOG)
            .expect("the first failure should have stopped the run");
        second_gate_sender.send(()).ok();
    });

    let Err(error) = result else {
        panic!("both stages failed, so the run should have failed");
    };

    assert_eq!(error.node(), Some(first), "the run reported {error}");
    assert_eq!(FAILING_CALLS[0].load(Ordering::SeqCst), 1);
    assert_eq!(
        FAILING_CALLS[1].load(Ordering::SeqCst),
        1,
        "the second stage never reached the failure it was released for",
    );
}

/// A worker whose aggregate has already gone still has to end what it started.
///
/// Worker zero fails while worker one is held inside its own first frame. That failure is what
/// makes the aggregate give up, and the sink being told its stream ended is how this test knows
/// it has, since an aggregate closes the stream downstream on its way out. Only then is worker
/// one let go, so the results it comes back with have nowhere to go, which is the path that used
/// to leave the loop early with a start still open.
#[test]
fn a_worker_whose_aggregate_has_gone_still_ends_what_it_started() {
    let property = "a_worker_whose_aggregate_has_gone_still_ends_what_it_started";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (sink_ended_sender, sink_ended_receiver) = mpsc::channel();

    *FAILING_PARKED[0][0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *FAILING_PARKED[0][1].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_GATES[0][1].lock().expect("not poisoned") = Some(second_gate_receiver);
    *RECORDING_ENDED.lock().expect("not poisoned") = Some(sink_ended_sender);

    let specs = vec![
        source_spec("input", &[0, 1], 0),
        failing_worker_spec("filter", "input", 0, 0, 2),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        first_gate_sender.send(()).ok();
        sink_ended_receiver
            .recv_timeout(WATCHDOG)
            .expect("the aggregate should give up on the failed worker and close the stream");

        second_gate_sender.send(()).ok();
    });

    assert!(
        result.is_err(),
        "the first worker failed, so the run should have failed"
    );

    let starts = FAILING_STARTS[1].load(Ordering::SeqCst);
    let ends = FAILING_ENDS[1].load(Ordering::SeqCst);

    assert_eq!(ends, 1, "{property}: the surviving worker never ended its chunk");
    assert_eq!(starts, ends, "{property}: {starts} starts against {ends} ends");
}

/// Chunk ids starting at five, so the one reported cannot be a default or a frame index.
#[test]
fn a_stage_failure_names_the_chunk_it_came_from() {
    let specs = vec![
        source_spec("input", &[5, 6, 7], 0),
        failing_worker_spec("filter", "input", 0, 0, 2),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_result_with_watchdog(engine, "a_stage_failure_names_the_chunk_it_came_from");

    let Err(error) = result else {
        panic!("a stage that failed should have failed the run");
    };

    assert_eq!(error.node(), Some(failing), "the run reported {error}");
    assert_eq!(error.chunk(), Some(5), "the run reported {error}");
}

/// The inherited hang, as the catalogue describes it: a worker fails, its aggregate gives up on
/// it, and nothing will ever mark a worker free again while the disperse is waiting for one.
///
/// Both workers are held inside their own first frame and then fail the moment they are let go,
/// so neither ever closes the chunk it was given and nothing ever frees a worker. The only thing
/// that can get the disperse out of its wait is the stop the first failure raises. Six more
/// frames are produced behind the two the workers took, so by the time they are let go the
/// disperse has long since taken the next bundle and found nobody free for it.
#[test]
fn a_failure_while_the_disperse_waits_for_a_worker_still_ends_the_run() {
    let property = "a_failure_while_the_disperse_waits_for_a_worker_still_ends_the_run";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (source_done_sender, source_done_receiver) = mpsc::channel();

    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    *FAILING_PARKED[0][0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *FAILING_PARKED[0][1].lock().expect("not poisoned") = Some(parked_sender);
    *FAILING_GATES[0][0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *FAILING_GATES[0][1].lock().expect("not poisoned") = Some(second_gate_receiver);

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        failing_spec("filter", "input", 0, 1, Some(2)),
        sink_spec("out", "filter"),
    ];
    let (graph, solved) = planned_graph(&specs);
    let failing = node_id(&graph, "filter");
    let engine = engine_from(&graph, &solved, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |_cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        source_done_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should finish producing every frame");

        first_gate_sender.send(()).ok();
        second_gate_sender.send(()).ok();
    });

    let Err(error) = result else {
        panic!("both workers failed, so the run should have failed");
    };

    assert_eq!(error.node(), Some(failing), "the run reported {error}");
    assert_every_thread_exited(property);
}

/// Three chunks and two workers, both held inside their own first frame, so the third chunk's
/// bundle completes with nobody free to take it and the disperse is left waiting.
///
/// Cancelling is what gets the disperse out of that wait, and the bundle it was still holding is
/// what the report has to account for. The workers are released afterwards, because a stage
/// blocked inside a call of its own is the one place cancellation cannot reach.
#[test]
fn a_bundle_dropped_at_shutdown_is_reported_rather_than_discarded() {
    let property = "a_bundle_dropped_at_shutdown_is_reported_rather_than_discarded";
    let (parked_sender, parked_receiver) = mpsc::channel();
    let (first_gate_sender, first_gate_receiver) = mpsc::channel();
    let (second_gate_sender, second_gate_receiver) = mpsc::channel();
    let (source_done_sender, source_done_receiver) = mpsc::channel();

    *SOURCE_DONE.lock().expect("not poisoned") = Some(source_done_sender);
    *SLOW_PARKED[0].lock().expect("not poisoned") = Some(parked_sender.clone());
    *SLOW_PARKED[1].lock().expect("not poisoned") = Some(parked_sender);
    *SLOW_GATES[0].lock().expect("not poisoned") = Some(first_gate_receiver);
    *SLOW_GATES[1].lock().expect("not poisoned") = Some(second_gate_receiver);

    let specs = vec![
        source_spec("input", &[0, 1, 2], 0),
        slow_spec("branch", "input", 2),
        sink_spec("out", "branch"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_driven_with_watchdog(engine, property, |cancel| {
        for _ in 0..2 {
            parked_receiver
                .recv_timeout(WATCHDOG)
                .expect("both workers should take a chunk and park inside it");
        }

        source_done_receiver
            .recv_timeout(WATCHDOG)
            .expect("the source should finish producing every frame");

        cancel.stop();
        first_gate_sender.send(()).ok();
        second_gate_sender.send(()).ok();
    });

    let report = result.expect("a cancelled run is not a failed one");

    assert!(report.cancelled);
    assert_eq!(
        report.dropped, 1,
        "the bundle held when the run stopped was not accounted for",
    );
    assert_eq!(report.frames, 2, "the two chunks already with a worker still ran");
    assert_every_thread_exited(property);
}

#[test]
fn a_stage_gets_no_start_without_a_matching_end() {
    PASS_STARTS.store(0, Ordering::SeqCst);
    PASS_ENDS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..8).collect();
    let specs = vec![
        source_spec("input", &chunks, 0),
        filter_spec("filter", "input", Some(4)),
        sink_spec("out", "filter"),
    ];
    let engine = build_engine(&specs, Settings::default());

    run_with_watchdog(engine, "a_stage_gets_no_start_without_a_matching_end");

    let starts = PASS_STARTS.load(Ordering::SeqCst);
    let ends = PASS_ENDS.load(Ordering::SeqCst);

    assert_eq!(starts, ends, "{starts} starts against {ends} ends");
}

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

/// Three sources whose frames start at index one hundred, and two joins chained off them.
///
/// Each join heads a region of one worker with two inputs, which the plan gives a disperse to
/// pair them and no aggregate. Nothing in either region renumbers, so the indices the sources
/// chose reach the sink untouched. Building an aggregate for the first join would renumber its
/// output from zero, and the second join would then be pairing frame nought with frame one
/// hundred, which is not a pairing at all.
#[test]
fn a_width_one_region_with_a_two_input_head_does_not_renumber() {
    let property = "a_width_one_region_with_a_two_input_head_does_not_renumber";
    clear_recorded();

    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        offset_source_spec("base_input", &chunks, 0, 100),
        offset_source_spec("overlay_input", &chunks, 1, 100),
        join_spec("first_join", "base_input", "overlay_input"),
        offset_source_spec("third_input", &chunks, 2, 100),
        join_spec("second_join", "first_join", "third_input"),
        sink_spec("out", "second_join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    let mut indices = recorded_indices();
    indices.sort_unstable();

    assert_eq!(indices, vec![100, 101, 102]);
    assert_eq!(report.frames, 3);
}

/// A join whose two branches disagree about which frame they are carrying.
///
/// `branch` is a widened region, so its aggregate renumbers what it emits from zero, while the
/// source's own output reaches the join still numbered from one hundred. Pairing them is what
/// `OutOfStep` names, and it used to surface as a panic about a slot running ahead instead.
#[test]
fn a_join_whose_inputs_carry_different_frames_fails_the_run() {
    let property = "a_join_whose_inputs_carry_different_frames_fails_the_run";
    let specs = vec![
        offset_source_spec("input", &[0, 1, 2], 0, 100),
        filter_spec("branch", "input", Some(2)),
        join_spec("join", "input", "branch"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("a join given two different frames should have failed the run");
    };
    assert!(
        matches!(
            error,
            RunError::Exchange {
                source: ExchangeError::OutOfStep { .. }
            }
        ),
        "the run reported {error}",
    );
}

/// A worker ending while it still owes a chunk, on a run nothing has stopped.
///
/// The tolerance for this is guarded on the run having stopped, and that is a race rather than an
/// invariant: a thread that fails drops its senders a moment before the stop it raised is
/// recorded, and a worker can flush and end inside that window. Reporting rather than panicking
/// is what leaves the real cause the first error of the run.
#[test]
fn an_aggregate_whose_worker_ends_owing_a_chunk_reports_rather_than_panics() {
    let (result_sender, result_receiver) = mpsc::sync_channel(1);
    let (assignment_sender, assignment_receiver) = mpsc::channel();
    let mut aggregate = Aggregate::new();
    aggregate.assign(Assignment { chunk: 0, worker: 0 });

    result_sender.send(WorkResult::EndOfStream).expect("should send");
    drop(assignment_sender);

    let side = AggregateSide {
        aggregate,
        assignment_receiver,
    };
    let freedom = WorkerFreedom::new(1);
    let cancel = Cancel::new();

    let result = run_aggregate(side, vec![result_receiver], &freedom, &cancel, Vec::new());

    let Err(error) = result else {
        panic!("a worker ending while it owed a chunk should have been reported");
    };
    assert!(
        matches!(
            error,
            RunError::Exchange {
                source: ExchangeError::WorkerEndedOwingChunk { .. }
            }
        ),
        "the aggregate reported {error}",
    );
}

/// Two sources of different lengths feeding one join, which can only pair as many as the shorter
/// one has.
///
/// The five frames left over never reach a worker at all. A run that answered `Ok` for that would
/// leave a caller checking only the result with no sign that anything went missing.
#[test]
fn a_run_that_lost_items_without_being_cancelled_fails() {
    let property = "a_run_that_lost_items_without_being_cancelled_fails";
    clear_recorded();

    let long: Vec<u64> = (0..8).collect();
    let short: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &long, 0),
        source_spec("overlay_input", &short, 1),
        join_spec("join", "base_input", "overlay_input"),
        sink_spec("out", "join"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let result = run_result_with_watchdog(engine, property);

    let Err(error) = result else {
        panic!("a run that lost five frames should not have reported success");
    };
    let RunError::DroppedItems { dropped } = error else {
        panic!("the run reported {error}");
    };

    assert_eq!(dropped, 5);
}

/// A sink of two inputs consumes two items every step, and the report counts both of them.
#[test]
fn a_report_counts_every_input_slot_a_sink_consumed() {
    let property = "a_report_counts_every_input_slot_a_sink_consumed";
    PAIRING_STEPS.store(0, Ordering::SeqCst);

    let chunks: Vec<u64> = (0..3).collect();
    let specs = vec![
        source_spec("base_input", &chunks, 0),
        source_spec("overlay_input", &chunks, 1),
        pairing_spec("out", "base_input", "overlay_input"),
    ];
    let engine = build_engine(&specs, Settings::default());

    let report = run_with_watchdog(engine, property);

    assert_eq!(PAIRING_STEPS.load(Ordering::SeqCst), 3);
    assert_eq!(
        report.frames, 6,
        "{property}: both of the sink's slots should count"
    );
}

#[test]
fn settings_and_reports_carry_the_derives_a_caller_needs() {
    let settings = Settings::default();
    let report = RunReport {
        frames: 2,
        cancelled: false,
        dropped: 0,
    };

    let copied_settings = settings.clone();
    let copied_report = report.clone();

    assert!(copied_settings.frame_budget.is_none());
    assert!(copied_settings.stall_deadline.is_none());
    assert_eq!(copied_report.frames, 2);
    assert!(format!("{report:?}").contains("frames"));
}
