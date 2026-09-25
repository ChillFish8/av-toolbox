use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, mpsc};

use av_toolbox_core::{
    BoundInputs,
    InputParam,
    InputSignature,
    Inputs,
    MetaInputs,
    MetaSink,
    MetaSinkDefinition,
    Sink,
    SinkDefinition,
    StageDefinition,
    StageError,
    StageInstance,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    FrameContext,
    HostSide,
    Offered,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamItem,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::engine::tests::support::StageLife;

// A sink that records the timestamp of every frame it consumes, in the order it consumed them.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct RecordingArgs {}

/// Every frame a `Recording` sink consumed, as `(timestamp, index)` pairs, in consumption order.
///
/// Kept as one pair per frame rather than two separate lists, so the timestamp and the index
/// recorded alongside it always describe the same frame.
///
/// A timestamp is the source's own and survives everything, which is what makes it the right
/// thing to check a stream's contents against. An index is renumbered by every aggregate it
/// passes through, which is what makes it the right thing to check renumbering against.
pub(crate) static RECORDED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

/// Signalled once, when a `Recording` sink is told its stream has ended. Its region only reaches
/// that once whatever feeds it has closed the stream, so a test waiting here knows the region
/// upstream has finished, however it finished.
pub(crate) static RECORDING_ENDED: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

pub(crate) struct RecordingInstance {
    _life: StageLife,
}

impl RecordingInstance {
    pub(crate) fn new() -> Self {
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

        RECORDED
            .lock()
            .expect("not poisoned")
            .push((timestamp, item.meta().index));

        Ok(())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        if let Some(sender) = RECORDING_ENDED.lock().expect("not poisoned").take() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

pub(crate) struct Recording;

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

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(RecordingInstance::new()))
    }
}

pub(crate) struct HostRecordingInstance {
    _life: StageLife,
}

impl HostRecordingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for HostRecordingInstance {}

impl Sink<HostSide> for HostRecordingInstance {
    fn consume(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<(), StageError> {
        let item = inputs
            .take(0)
            .ok_or_else(|| StageError::msg("expected an item in slot 0"))?;
        let timestamp = item
            .meta()
            .timestamp
            .expect("every test frame carries a timestamp")
            .as_secs();

        RECORDED
            .lock()
            .expect("not poisoned")
            .push((timestamp, item.meta().index));

        Ok(())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        if let Some(sender) = RECORDING_ENDED.lock().expect("not poisoned").take() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

pub(crate) struct HostRecording;

impl StageDefinition for HostRecording {
    type Args = RecordingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-recording"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for HostRecording {
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

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink<HostSide>>, StageError> {
        Ok(Box::new(HostRecordingInstance::new()))
    }
}

// A sink that holds on to every item it is given, so nothing it consumes is ever credited back
// to the budget the source charged it to.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct HoardingArgs {}

/// Every item a `Hoarding` sink consumed, still alive.
static HOARDED: Mutex<Vec<StreamItem>> = Mutex::new(Vec::new());

/// Signalled once, the first time a `Hoarding` sink takes an item. From then on that item is
/// never dropped, so whatever budget it was charged to is never credited back.
pub(crate) static HOARDED_ONE: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Signalled every time a `Hoarding` sink takes an item, so a test can wait for a specific count
/// of items to have been permanently held rather than only the first.
pub(crate) static HOARDED_EVERY: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

pub(crate) struct HoardingInstance {
    _life: StageLife,
}

impl HoardingInstance {
    pub(crate) fn new() -> Self {
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

        if let Some(sender) = HOARDED_EVERY.lock().expect("not poisoned").as_ref() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

pub(crate) struct Hoarding;

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
pub(crate) struct PairingArgs {}

/// How many times a `Pairing` sink was asked to consume, which is half of what it consumed.
pub(crate) static PAIRING_STEPS: AtomicU32 = AtomicU32::new(0);

pub(crate) struct PairingInstance {
    _life: StageLife,
}

impl PairingInstance {
    pub(crate) fn new() -> Self {
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

pub(crate) struct Pairing;

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
pub(crate) struct GatedArgs {}

/// Taken once, by the `Gated` sink built for a run. Sent after every item is taken, before that
/// item is held for release.
pub(crate) static GATED_GOT: Mutex<Option<mpsc::Sender<()>>> = Mutex::new(None);

/// Taken once, by the `Gated` sink built for a run. One receive per item, so a test releases
/// them one at a time.
pub(crate) static GATED_RELEASE: Mutex<Option<mpsc::Receiver<()>>> = Mutex::new(None);

pub(crate) struct GatedInstance {
    _life: StageLife,
    got: mpsc::Sender<()>,
    release: mpsc::Receiver<()>,
}

impl GatedInstance {
    pub(crate) fn new() -> Self {
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

pub(crate) struct Gated;

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

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(GatedInstance::new()))
    }
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct MetaRecordingArgs {}

// A metadata-only sink that records the `(timestamp, index)` of every item it consumes, without
// touching a plane, so it runs on whichever side its input already sits on.
pub(crate) struct MetaRecordingInstance {
    _life: StageLife,
}

impl MetaRecordingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for MetaRecordingInstance {}

impl MetaSink for MetaRecordingInstance {
    fn consume(&mut self, _context: &mut FrameContext, mut inputs: MetaInputs) -> Result<(), StageError> {
        let item = inputs.take(0).expect("an item");
        let timestamp = item
            .meta()
            .timestamp
            .expect("every test frame carries a timestamp")
            .as_secs();

        RECORDED
            .lock()
            .expect("not poisoned")
            .push((timestamp, item.meta().index));

        Ok(())
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        if let Some(sender) = RECORDING_ENDED.lock().expect("not poisoned").take() {
            sender.send(()).ok();
        }

        Ok(())
    }
}

pub(crate) struct MetaRecording;

impl StageDefinition for MetaRecording {
    type Args = MetaRecordingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "meta-recording"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl MetaSinkDefinition for MetaRecording {
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
    ) -> Result<Box<dyn MetaSink>, StageError> {
        Ok(Box::new(MetaRecordingInstance::new()))
    }
}
