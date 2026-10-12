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
    HostFrame,
    HostSide,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{chunked, host_chunked};
use crate::engine::tests::support::{InstanceId, StageLife, WorkerGate, frame_second};

// A filter that passes its input through, but whose first call blocks until a test releases it.
// Standing in for a producer whose path to a join is genuinely, deliberately slower than the
// path the other side of that join takes, rather than merely one thread hop.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct SlowArgs {}

/// Taken by the `Slow` worker of the matching index. `None` means run unblocked, which is every
/// worker of every test that never sets one.
pub(crate) static SLOW_GATES: [Mutex<Option<mpsc::Receiver<()>>>; 2] = [Mutex::new(None), Mutex::new(None)];

/// Signalled by a `Slow` worker the moment before it waits on its gate.
pub(crate) static SLOW_PARKED: [Mutex<Option<mpsc::Sender<()>>>; 2] = [Mutex::new(None), Mutex::new(None)];

/// The signal and gate set for the slow worker at `worker`, taken so no other instance sees them.
fn take_slow_hooks(worker: usize) -> (Option<mpsc::Sender<()>>, Option<mpsc::Receiver<()>>) {
    let parked = SLOW_PARKED
        .get(worker)
        .and_then(|parked| parked.lock().expect("not poisoned").take());
    let gate = SLOW_GATES
        .get(worker)
        .and_then(|gate| gate.lock().expect("not poisoned").take());

    (parked, gate)
}

pub(crate) struct SlowInstance {
    _life: StageLife,
    parked: Option<mpsc::Sender<()>>,
    gate: Option<mpsc::Receiver<()>>,
}

impl SlowInstance {
    /// An instance nothing holds back, for the registry build path no test drives.
    pub(crate) fn unblocked() -> Self {
        Self {
            _life: StageLife::new(),
            parked: None,
            gate: None,
        }
    }

    /// The instance for `worker`, holding whichever gate and signal that worker was given.
    pub(crate) fn for_worker(worker: usize) -> Self {
        let (parked, gate) = take_slow_hooks(worker);

        Self {
            _life: StageLife::new(),
            parked,
            gate,
        }
    }
}

impl StageInstance for SlowInstance {}

impl Filter for SlowInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        if let Some(parked) = self.parked.take() {
            parked.send(()).ok();
        }

        if let Some(gate) = self.gate.take() {
            gate.recv().ok();
        }

        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

pub(crate) struct Slow;

impl StageDefinition for Slow {
    type Args = SlowArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "slow"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    // Per chunk, so a held worker is holding a whole chunk. Every test that gates one of these
    // is written about a chunk taking far longer than the rest.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        chunked()
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Slow {
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
        Ok(Box::new(SlowInstance::unblocked()))
    }
}

// The host side twin of `Slow`, held by the same gates.
pub(crate) struct HostSlowInstance {
    _life: StageLife,
    parked: Option<mpsc::Sender<()>>,
    gate: Option<mpsc::Receiver<()>>,
}

impl HostSlowInstance {
    pub(crate) fn unblocked() -> Self {
        Self {
            _life: StageLife::new(),
            parked: None,
            gate: None,
        }
    }

    /// The instance for `worker`, holding the same gate and signal a `Slow` worker would.
    pub(crate) fn for_worker(worker: usize) -> Self {
        let (parked, gate) = take_slow_hooks(worker);

        Self {
            _life: StageLife::new(),
            parked,
            gate,
        }
    }
}

impl StageInstance for HostSlowInstance {}

impl Filter<HostSide> for HostSlowInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostFrame>, StageError> {
        if let Some(parked) = self.parked.take() {
            parked.send(()).ok();
        }

        if let Some(gate) = self.gate.take() {
            gate.recv().ok();
        }

        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

pub(crate) struct HostSlow;

impl StageDefinition for HostSlow {
    type Args = SlowArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-slow"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        host_chunked()
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for HostSlow {
    type Side = HostSide;

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
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        Ok(Box::new(HostSlowInstance::unblocked()))
    }
}

// A filter that passes its input through and records which of its own instances processed each
// frame, so a test can count how many workers a region really used. Declared per chunk or per
// frame by its args, so one fixture covers both routings.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct TrackedArgs {
    /// Declares the stage per chunk rather than per frame.
    #[serde(default)]
    per_chunk: bool,
}

pub(crate) struct TrackedInstance {
    _life: StageLife,
    id: InstanceId,
    gate: WorkerGate,
}

impl TrackedInstance {
    pub(crate) fn unblocked() -> Self {
        Self {
            _life: StageLife::new(),
            id: InstanceId::new(),
            gate: WorkerGate::open(),
        }
    }

    pub(crate) fn for_worker(worker: usize) -> Self {
        Self {
            _life: StageLife::new(),
            id: InstanceId::new(),
            gate: WorkerGate::for_worker(worker),
        }
    }
}

impl StageInstance for TrackedInstance {}

impl Filter for TrackedInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        self.gate.wait();

        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        self.id.processed(frame_second(frame.meta()));
        WorkerGate::processed();

        Ok(vec![frame])
    }
}

pub(crate) struct Tracked;

impl StageDefinition for Tracked {
    type Args = TrackedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "tracked"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_chunk {
            true => chunked(),
            false => Parallelism::Frame,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Tracked {
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
        Ok(Box::new(TrackedInstance::unblocked()))
    }
}

// A frame mode filter of two inputs, recording the pair each bundle handed it. A bundle whose
// slots crossed would still produce a frame, so the pair itself is the only place it shows.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct MatchingArgs {}

/// Every pair a `Matching` instance was given, as `(base, overlay)` timestamps.
pub(crate) static MATCHED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

pub(crate) struct MatchingInstance {
    _life: StageLife,
    id: InstanceId,
    gate: WorkerGate,
}

impl MatchingInstance {
    pub(crate) fn unblocked() -> Self {
        Self {
            _life: StageLife::new(),
            id: InstanceId::new(),
            gate: WorkerGate::open(),
        }
    }

    pub(crate) fn for_worker(worker: usize) -> Self {
        Self {
            _life: StageLife::new(),
            id: InstanceId::new(),
            gate: WorkerGate::for_worker(worker),
        }
    }
}

impl StageInstance for MatchingInstance {}

impl Filter for MatchingInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        self.gate.wait();

        let base = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;
        let overlay = inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

        let base_second = frame_second(base.meta());
        let overlay_second = frame_second(overlay.meta());

        MATCHED
            .lock()
            .expect("not poisoned")
            .push((base_second, overlay_second));
        self.id.processed(base_second);
        WorkerGate::processed();

        Ok(vec![base])
    }
}

pub(crate) struct Matching;

impl StageDefinition for Matching {
    type Args = MatchingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "matching"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("base").accepting(StreamType::Frames),
            InputParam::required("overlay").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Matching {
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
        Ok(Box::new(MatchingInstance::unblocked()))
    }
}
