use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

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
    FrameMeta,
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

use crate::engine::tests::support::{StageLife, format};

// A filter that passes its one input through unchanged.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct PassArgs {}

/// How many times a `Pass` instance was built, across every worker of every region.
pub(crate) static PASS_BUILD_CALLS: AtomicU32 = AtomicU32::new(0);

/// Whether the last `Pass` instance started saw a budget on its context.
pub(crate) static PASS_HAD_BUDGET: Mutex<Option<bool>> = Mutex::new(None);

/// How many times a `Pass` instance was told a stream started, and how many times one ended,
/// across every worker of every region.
pub(crate) static PASS_STARTS: AtomicU32 = AtomicU32::new(0);
pub(crate) static PASS_ENDS: AtomicU32 = AtomicU32::new(0);

pub(crate) struct PassInstance {
    _life: StageLife,
}

impl PassInstance {
    pub(crate) fn new() -> Self {
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
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        PASS_ENDS.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }
}

pub(crate) struct Pass;

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
        PASS_BUILD_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(PassInstance::new()))
    }
}

pub(crate) struct HostPassInstance {
    _life: StageLife,
}

impl HostPassInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for HostPassInstance {}

impl Filter<HostSide> for HostPassInstance {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<Vec<HostFrame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

pub(crate) struct HostPass;

impl StageDefinition for HostPass {
    type Args = PassArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-pass"
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

impl FilterDefinition for HostPass {
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
        Ok(Box::new(HostPassInstance::new()))
    }
}

// A pass through filter that also emits one frame of its own from `end_of_stream`, so a run that
// never delivers a region's flush loses a frame a test can see. Declared per chunk or per frame
// by its args, so one fixture covers both routings.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct FlushingArgs {
    /// Declares the stage per chunk rather than per frame.
    #[serde(default)]
    per_chunk: bool,
}

/// The timestamp every frame a `Flushing` instance emits from `end_of_stream` carries.
///
/// It sits far above anything a source in these tests produces, so a flushed frame is
/// unmistakable wherever it lands in the recorded output.
pub(crate) const FLUSHED_TIMESTAMP: u64 = 999;

pub(crate) struct FlushingInstance {
    _life: StageLife,
}

impl FlushingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for FlushingInstance {}

impl Filter for FlushingInstance {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let mut meta = FrameMeta::at(0);
        meta.timestamp = Some(Duration::from_secs(FLUSHED_TIMESTAMP));

        Ok(vec![context.alloc_frame(&format(), meta)])
    }
}

pub(crate) struct Flushing;

impl StageDefinition for Flushing {
    type Args = FlushingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "flushing"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.per_chunk {
            true => Parallelism::Chunk,
            false => Parallelism::Frame,
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Flushing {
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
        Ok(Box::new(FlushingInstance::new()))
    }
}

// A filter that allocates its output instead of passing its input through, standing in for every
// real filter that writes somewhere new. A region of several of these holds one frame per node at
// once, all of them charged to whichever budget the region's context carries.
#[derive(Debug, Default, Deserialize, JsonSchema)]
pub(crate) struct AllocatingArgs {}

pub(crate) struct AllocatingInstance {
    _life: StageLife,
}

impl AllocatingInstance {
    pub(crate) fn new() -> Self {
        Self {
            _life: StageLife::new(),
        }
    }
}

impl StageInstance for AllocatingInstance {}

impl Filter for AllocatingInstance {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let input = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        let output = context.alloc_frame(&format(), input.meta().clone());

        Ok(vec![output])
    }
}

pub(crate) struct Allocating;

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
        Ok(Box::new(AllocatingInstance::new()))
    }
}
