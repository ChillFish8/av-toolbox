use std::sync::atomic::{AtomicUsize, Ordering};

use av_toolbox_core::{
    AnyProbe,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputProbe,
    InputSignature,
    Inputs,
    ProbeDefinition,
    Probed,
    Seeker,
    Source,
    SourceDefinition,
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
    FrameFormat,
    FrameMeta,
    HostFrame,
    HostPool,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::shared::{NoArgs, first_bound, fixture_format, frame_format, frames_template};

/// How many seekers [Seekable] has opened.
pub(crate) static SEEKABLE_OPENS: AtomicUsize = AtomicUsize::new(0);
/// How many frames the seekers of [Seekable] have read.
pub(crate) static SEEKER_READS: AtomicUsize = AtomicUsize::new(0);
/// How many times [FixtureProbe] or [PlainProbe] has probed a source.
pub(crate) static PROBE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Asks a probe to answer `mark` plus the source's frame count.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, JsonSchema)]
pub(crate) struct MarkRequest {
    pub(crate) mark: u32,
}

/// A probe kind answering `mark` plus the frame count, merging several sources by the largest answer.
pub(crate) struct FixtureProbe;

impl ProbeDefinition for FixtureProbe {
    type Request = MarkRequest;
    type Output = u32;

    fn kind() -> &'static str {
        "fixture-probe"
    }

    fn describe(request: &MarkRequest) -> String {
        format!("mark {}", request.mark)
    }

    fn probe(seeker: &mut dyn Seeker, requests: &[MarkRequest]) -> Result<Vec<u32>, StageError> {
        answer_marks(seeker, requests)
    }

    fn combine(outputs: Vec<u32>) -> Result<u32, StageError> {
        let largest = outputs.into_iter().max();

        largest.ok_or_else(nothing_to_combine)
    }
}

/// The same as [FixtureProbe] under its own kind, refusing to merge several sources.
pub(crate) struct PlainProbe;

impl ProbeDefinition for PlainProbe {
    type Request = MarkRequest;
    type Output = u32;

    fn kind() -> &'static str {
        "plain-probe"
    }

    fn describe(request: &MarkRequest) -> String {
        format!("mark {}", request.mark)
    }

    fn probe(seeker: &mut dyn Seeker, requests: &[MarkRequest]) -> Result<Vec<u32>, StageError> {
        answer_marks(seeker, requests)
    }
}

/// Counts the call, reads frame 0 and answers each request with its mark plus the frame count.
fn answer_marks(seeker: &mut dyn Seeker, requests: &[MarkRequest]) -> Result<Vec<u32>, StageError> {
    PROBE_CALLS.fetch_add(1, Ordering::SeqCst);

    seeker.read(0)?;

    let frame_count = seeker.frame_count() as u32;
    let answers = requests
        .iter()
        .map(|request| request.mark + frame_count)
        .collect();

    Ok(answers)
}

fn nothing_to_combine() -> StageError {
    StageError::msg("there was nothing to combine")
}

struct FixtureSeeker {
    format: FrameFormat,
    pool: HostPool,
}

impl Seeker for FixtureSeeker {
    fn format(&self) -> &FrameFormat {
        &self.format
    }

    fn frame_count(&self) -> usize {
        8
    }

    fn read(&mut self, index: usize) -> Result<HostFrame, StageError> {
        SEEKER_READS.fetch_add(1, Ordering::SeqCst);

        let meta = FrameMeta::at(index as u64);
        let frame = self.pool.alloc_frame(&self.format, meta);

        Ok(frame.freeze())
    }
}

struct SeekableStage;

impl StageInstance for SeekableStage {}

impl Source for SeekableStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A source with a seeker over 8 frames.
pub(super) struct Seekable;

impl StageDefinition for Seekable {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "seekable"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Seekable {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(())
    }

    fn seeker(_opened: &Self::Opened) -> Result<Option<Box<dyn Seeker>>, StageError> {
        SEEKABLE_OPENS.fetch_add(1, Ordering::SeqCst);

        let seeker = FixtureSeeker {
            format: frame_format(),
            pool: HostPool::new(),
        };

        Ok(Some(Box::new(seeker)))
    }

    fn produces(_capabilities: &Capabilities, _opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        let format = fixture_format();

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build(
        _context: &BuildContext,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(SeekableStage))
    }
}

fixture_stage!(
    Pair,
    PairStage,
    Filter,
    FilterDefinition,
    "pair",
    NoArgs,
    &InputSignature::new(&[
        InputParam::required("clip").accepting(StreamType::Frames),
        InputParam::required("other").accepting(StreamType::Frames),
    ]),
    frames_template
);

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct ProbingArgs {
    mark: u32,
    /// Whether to probe with [PlainProbe] rather than [FixtureProbe].
    #[serde(default)]
    plain: bool,
    /// Whether to probe the `extra` input rather than `input`.
    #[serde(default)]
    on_extra: bool,
    /// Whether settling fails.
    #[serde(default)]
    refuse: bool,
    #[serde(skip)]
    found: Option<u32>,
}

/// A filter probing one of its inputs, refusing to produce until it has settled.
pub(super) struct Probing;

impl Probing {
    fn probed_input(args: &ProbingArgs) -> &'static str {
        match args.on_extra {
            true => "extra",
            false => "input",
        }
    }
}

impl StageDefinition for Probing {
    type Args = ProbingArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "probing"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn probes(args: &Self::Args) -> Vec<InputProbe> {
        let request = MarkRequest { mark: args.mark };
        let probe = match args.plain {
            true => AnyProbe::new::<PlainProbe>(request),
            false => AnyProbe::new::<FixtureProbe>(request),
        };
        let input = Self::probed_input(args);

        vec![InputProbe { input, probe }]
    }

    fn settle(args: &mut Self::Args, probed: &Probed) -> Result<(), StageError> {
        if args.refuse {
            return Err(StageError::msg("the fixture refuses to settle"));
        }

        let input = Self::probed_input(args);
        let found = match args.plain {
            true => probed.get::<PlainProbe>(input),
            false => probed.get::<FixtureProbe>(input),
        };
        args.found = found.copied();

        Ok(())
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("input").accepting(StreamType::Frames),
            InputParam::optional("extra").accepting(StreamType::Frames),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for Probing {
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
        if args.found.is_none() {
            return Err(StageError::msg("not settled"));
        }

        let format = first_bound(chosen)?;

        Ok(OutputFormats {
            template: frames_template(format),
            samples: FormatSet::of([format.sample()]),
        })
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PairStage))
    }
}
