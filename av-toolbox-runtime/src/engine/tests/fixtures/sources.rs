use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::time::Duration;

use av_toolbox_core::testing::fixture_chunk;
use av_toolbox_core::{
    BoundInputs,
    InputSignature,
    Parallelism,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameBudget,
    FrameContext,
    FrameMeta,
    FrameRate,
    HostFrame,
    HostSide,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::engine::tests::support::{StageLife, format};

// A source that plays back a fixed list of chunk ids, one frame each, timestamped by the
// position it was built at, so a test can tell frames apart after the aggregate renumbers
// their indices.
#[derive(Debug, Default, Clone, Deserialize, JsonSchema)]
pub(crate) struct ListedArgs {
    #[serde(default)]
    chunks: Vec<u64>,
    #[serde(default)]
    id: u32,
    /// The frame index the first frame carries, so a test can tell a stream that was renumbered
    /// from one that was not.
    #[serde(default)]
    index_base: u64,
    /// The frame rate the source declares, as numerator and denominator. Unknown when absent.
    #[serde(default)]
    frame_rate: Option<[u32; 2]>,
}

/// The frame budget limit every `Listed` source instance saw, by its `id`.
///
/// Read only after a run, since it is written from inside `start_of_stream` on the source's own
/// worker thread.
pub(crate) static SOURCE_BUDGETS: Mutex<Vec<(u32, Option<usize>)>> = Mutex::new(Vec::new());

/// Signalled once, from inside `next`, the moment a `Listed` source discovers it has nothing
/// left. By then every real frame has already been forwarded to every downstream consumer, from
/// the loop iteration before this one, so a test reading this signal knows production is
/// completely finished without guessing at timing.
pub(crate) static SOURCE_DONE: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Signalled once, from inside `next`, just before a `Listed` source allocates a frame that its
/// own budget's current state says would have to wait for room. Checked against that state
/// rather than assumed from a fixed position, so it fires at whatever allocation genuinely
/// saturates a share of any size, not only the smallest one. The budget's release is sticky, so
/// a test cancelling on this signal is right whether or not this thread has reached the wait yet.
pub(crate) static SOURCE_WAITING: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// The highest outstanding byte count any `Listed` source has observed on its own budget,
/// sampled right after it charges each frame.
pub(crate) static SOURCE_BUDGET_PEAK: AtomicUsize = AtomicUsize::new(0);

fn record_budget_peak(outstanding: usize) {
    SOURCE_BUDGET_PEAK.fetch_max(outstanding, Ordering::SeqCst);
}

/// Whether allocating `bytes` against `budget` right now would have to wait for room, mirroring
/// the rule `FrameBudget::wait_for_room` applies internally.
fn would_wait_for_room(budget: &FrameBudget, bytes: usize) -> bool {
    let outstanding = budget.outstanding();
    outstanding != 0 && outstanding + bytes > budget.limit()
}

pub(crate) struct ListedInstance {
    _life: StageLife,
    remaining: VecDeque<u64>,
    position: u64,
    index_base: u64,
    id: u32,
}

impl ListedInstance {
    pub(crate) fn from_args(args: ListedArgs) -> Self {
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

        let at = FrameMeta::at(self.index_base + self.position);
        let mut meta = fixture_chunk(at, chunk);
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

pub(crate) struct Listed;

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
    type Side = DeviceSide;
    type Opened = ListedArgs;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(args.clone())
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let mut frame_format = format();
        frame_format.frame_rate = opened
            .frame_rate
            .and_then(|[numerator, denominator]| FrameRate::new(numerator, denominator));
        let stream_format = StreamFormat::Frames(frame_format);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&stream_format),
            samples: FormatSet::of([stream_format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(ListedInstance::from_args(opened)))
    }
}

pub(crate) struct HostListedInstance {
    _life: StageLife,
    remaining: VecDeque<u64>,
    position: u64,
    index_base: u64,
    id: u32,
}

impl HostListedInstance {
    pub(crate) fn from_args(args: ListedArgs) -> Self {
        Self {
            _life: StageLife::new(),
            remaining: args.chunks.into_iter().collect(),
            position: 0,
            index_base: args.index_base,
            id: args.id,
        }
    }
}

impl StageInstance for HostListedInstance {
    fn start_of_stream(&mut self, context: &mut FrameContext) -> Result<(), StageError> {
        let limit = context.budget().map(|budget| budget.limit());
        SOURCE_BUDGETS
            .lock()
            .expect("not poisoned")
            .push((self.id, limit));
        Ok(())
    }
}

impl Source<HostSide> for HostListedInstance {
    fn next(&mut self, context: &mut FrameContext) -> Result<Option<HostFrame>, StageError> {
        let Some(chunk) = self.remaining.pop_front() else {
            if let Some(sender) = SOURCE_DONE.lock().expect("not poisoned").take() {
                sender.send(()).ok();
            }
            return Ok(None);
        };

        let at = FrameMeta::at(self.index_base + self.position);
        let mut meta = fixture_chunk(at, chunk);
        meta.timestamp = Some(Duration::from_secs(self.position));

        if let Some(budget) = context.budget()
            && would_wait_for_room(budget, format().total_bytes())
            && let Some(sender) = SOURCE_WAITING.lock().expect("not poisoned").take()
        {
            sender.send(()).ok();
        }

        self.position += 1;

        let frame = context.alloc_host_frame(&format(), meta).freeze();
        if let Some(budget) = context.budget() {
            record_budget_peak(budget.outstanding());
        }

        Ok(Some(frame))
    }
}

pub(crate) struct HostListed;

impl StageDefinition for HostListed {
    type Args = ListedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-listed"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for HostListed {
    type Side = HostSide;
    type Opened = ListedArgs;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(args.clone())
    }

    fn produces(_capabilities: &Capabilities, _opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let stream_format = StreamFormat::Frames(format());

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&stream_format),
            samples: FormatSet::of([stream_format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        Ok(Box::new(HostListedInstance::from_args(opened)))
    }
}
