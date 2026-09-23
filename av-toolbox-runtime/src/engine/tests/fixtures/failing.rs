use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, mpsc};

use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    StageDefinition,
    StageError,
    StageInstance,
    StreamType,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::engine::tests::support::StageLife;

// A filter that fails on a chosen call, and that a test can hold back until it says so, which
// is what lets two failures in one run be ordered against each other rather than raced.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct FailingArgs {
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
pub(crate) static FAILING_CALLS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

/// Taken by the worker of the matching stage and index. A worker holding one waits there on its
/// first call, before it decides whether to fail, so a test can hold back one worker of a widened
/// stage, one of two separate stages, or every one of them.
pub(crate) static FAILING_GATES: [[Mutex<Option<mpsc::Receiver<()>>>; 2]; 2] = [
    [Mutex::new(None), Mutex::new(None)],
    [Mutex::new(None), Mutex::new(None)],
];

/// Signalled by a `Failing` worker the moment before it waits on its gate.
pub(crate) static FAILING_PARKED: [[Mutex<Option<mpsc::Sender<()>>>; 2]; 2] = [
    [Mutex::new(None), Mutex::new(None)],
    [Mutex::new(None), Mutex::new(None)],
];

/// How many times each worker of a `Failing` stage was told a stream started, and how many times
/// one ended.
pub(crate) static FAILING_STARTS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];
pub(crate) static FAILING_ENDS: [AtomicU32; 2] = [AtomicU32::new(0), AtomicU32::new(0)];

pub(crate) struct FailingInstance {
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
    pub(crate) fn for_worker(args: FailingArgs, worker: usize) -> Self {
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

pub(crate) struct Failing;

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
    type Side = DeviceSide;

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
pub(crate) struct PanickingArgs {}

pub(crate) struct PanickingInstance {
    _life: StageLife,
}

impl PanickingInstance {
    pub(crate) fn new() -> Self {
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

pub(crate) struct Panicking;

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
    type Side = DeviceSide;

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
