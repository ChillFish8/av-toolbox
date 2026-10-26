use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

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
    Depth,
    DeviceSide,
    ElementType,
    FormatSet,
    Frame,
    FrameContext,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::chunked;
use super::shared::{BlurArgs, NoArgs, TemporalArgs, first_bound, frames_template};

fixture_stage!(
    Blur,
    BlurStage,
    Filter,
    FilterDefinition,
    "blur",
    BlurArgs,
    &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]),
    frames_template
);

fixture_stage!(
    Deband,
    DebandStage,
    Filter,
    FilterDefinition,
    "deband",
    NoArgs,
    &InputSignature::new(&[
        InputParam::required("reference").accepting(StreamType::Frames),
        InputParam::optional("overlay").accepting(StreamType::Mask),
    ]),
    frames_template
);

pub(super) struct Failing;

impl StageDefinition for Failing {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "failing"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
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
        let format = first_bound(chosen)?;

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
        Err(StageError::msg("this stage always fails"))
    }
}

/// The width [Recording] was built with, read from the first of its resolved inputs.
///
/// Stays `0` when the stage never built, and when that input was not a frame stream.
pub(crate) static RECORDED_INPUT_WIDTH: AtomicU32 = AtomicU32::new(0);

struct RecordingStage;

impl StageInstance for RecordingStage {}

impl Filter for RecordingStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

pub(super) struct Recording;

impl StageDefinition for Recording {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "recording"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for Recording {
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
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let width = match formats.input(0) {
            Some(StreamFormat::Frames(format)) => format.width,
            _ => 0,
        };
        RECORDED_INPUT_WIDTH.store(width, Ordering::SeqCst);

        Ok(Box::new(RecordingStage))
    }
}

opinionated_filter!(
    Precise,
    PreciseStage,
    "precise",
    FormatSet::of([SampleFormat::F32])
);
opinionated_filter!(
    Either,
    EitherStage,
    "either",
    FormatSet::of([SampleFormat::U8, SampleFormat::F32])
);
opinionated_filter!(Narrow, NarrowStage, "narrow", FormatSet::of([SampleFormat::U8]));

struct Prefers16Stage;

impl StageInstance for Prefers16Stage {}

impl Filter for Prefers16Stage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A filter that would rather work in a ten-bit integer, and asks the device before offering it.
pub(super) struct Prefers16;

impl StageDefinition for Prefers16 {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "prefers16"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Prefers16 {
    type Side = DeviceSide;

    fn accepts_input(
        capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        _offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let mut formats = FormatSet::new();
        if capabilities.can_compute(ElementType::U16) {
            formats.push(SampleFormat::U16 { depth: Depth::TEN });
        }
        formats.push(SampleFormat::F32);

        Ok(formats)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;

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
        Ok(Box::new(Prefers16Stage))
    }
}

/// What [Sparse] was told about its inputs, recorded so a test can compare the three places a
/// stage hears about them.
///
/// The slot it was asked about and how many it was offered, then how many formats it was given to
/// produce from and which of those slots were filled, as a bit per slot.
pub(crate) static SPARSE_ASKED_SLOT: AtomicUsize = AtomicUsize::new(usize::MAX);
pub(crate) static SPARSE_OFFERED_SLOTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static SPARSE_CHOSEN_SLOTS: AtomicUsize = AtomicUsize::new(0);
pub(crate) static SPARSE_FILLED_SLOTS: AtomicUsize = AtomicUsize::new(0);

struct SparseStage;

impl StageInstance for SparseStage {}

impl Filter for SparseStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(1)
            .ok_or_else(|| StageError::msg("expected a frame in slot 1"))?;

        Ok(vec![frame])
    }
}

/// A filter whose first input is optional, so a pipeline can leave a hole before a bound slot.
pub(super) struct Sparse;

impl StageDefinition for Sparse {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "sparse"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::optional("first").accepting(StreamType::Frames),
            InputParam::required("second").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Sparse {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        index: usize,
        offered: StreamFormat,
        inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        SPARSE_ASKED_SLOT.store(index, Ordering::SeqCst);
        SPARSE_OFFERED_SLOTS.store(inputs.len(), Ordering::SeqCst);

        Ok(FormatSet::of([offered.sample()]))
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let filled: usize = chosen
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(index, _)| 1 << index)
            .sum();

        SPARSE_CHOSEN_SLOTS.store(chosen.len(), Ordering::SeqCst);
        SPARSE_FILLED_SLOTS.store(filled, Ordering::SeqCst);

        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(SparseStage))
    }
}

/// How many times [Temporal]'s stage was actually built.
///
/// A cached or shared build would only ever move this once, so a test proving two builds are
/// independent reads this rather than the stage's own state, which is empty.
pub(crate) static TEMPORAL_BUILD_CALLS: AtomicU32 = AtomicU32::new(0);

struct TemporalFilter;

impl StageInstance for TemporalFilter {}

impl Filter for TemporalFilter {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A filter whose granularity comes from its arguments, unlike every other fixture.
pub(super) struct Temporal;

impl StageDefinition for Temporal {
    type Args = TemporalArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "temporal"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        match args.radius {
            0 => Parallelism::Frame,
            _ => chunked(),
        }
    }

    fn workers(args: &Self::Args) -> Option<usize> {
        args.workers
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for Temporal {
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
        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        TEMPORAL_BUILD_CALLS.fetch_add(1, Ordering::SeqCst);

        Ok(Box::new(TemporalFilter))
    }
}

/// The sample a [ResampleDepth] converts to, its bits or `"f32"`.
#[derive(Debug, Deserialize, JsonSchema, Copy, Clone)]
#[serde(untagged)]
pub(super) enum DepthTarget {
    Bits(u8),
    Float(FloatTarget),
}

#[derive(Debug, Deserialize, JsonSchema, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub(super) enum FloatTarget {
    F32,
}

#[derive(Debug, Deserialize, JsonSchema, Copy, Clone)]
#[serde(rename_all = "snake_case")]
pub(super) enum DitherMode {
    None,
    Ordered,
    BlueNoise,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct ResampleDepthArgs {
    depth: DepthTarget,
    #[serde(default)]
    at_least: bool,
    #[serde(default)]
    #[expect(dead_code, reason = "the fixture only checks the argument parses")]
    dither: Option<DitherMode>,
}

fn bad_depth(depth: u8) -> StageError {
    let message = format!("resample/depth needs depth between 8 and 16, got {depth}");

    StageError::msg(message)
}

fn integer_sample(bits: u8) -> Result<SampleFormat, StageError> {
    if bits == 8 {
        return Ok(SampleFormat::U8);
    }

    let Ok(depth) = Depth::new(bits) else {
        return Err(bad_depth(bits));
    };

    Ok(SampleFormat::U16 { depth })
}

/// The samples `args` lets the output take, narrowest first.
fn depth_samples(args: &ResampleDepthArgs) -> Result<FormatSet, StageError> {
    let bits = match args.depth {
        DepthTarget::Float(FloatTarget::F32) => return Ok(FormatSet::of([SampleFormat::F32])),
        DepthTarget::Bits(bits) => bits,
    };

    let last = if args.at_least { 16 } else { bits };
    let mut samples = FormatSet::new();
    for deeper in bits..=last {
        let sample = integer_sample(deeper)?;
        samples.push(sample);
    }

    Ok(samples)
}

struct ResampleDepthStage;

impl StageInstance for ResampleDepthStage {}

impl Filter for ResampleDepthStage {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

/// A stand-in for `resample/depth`, taking any sample and emitting the depth it is asked for.
pub(super) struct ResampleDepth;

impl StageDefinition for ResampleDepth {
    type Args = ResampleDepthArgs;

    fn package() -> &'static str {
        "resample"
    }

    fn identifier() -> &'static str {
        "depth"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for ResampleDepth {
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
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let format = first_bound(chosen)?;
        let samples = depth_samples(args)?;

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples,
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(ResampleDepthStage))
    }
}
