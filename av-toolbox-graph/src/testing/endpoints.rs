use av_toolbox_core::{
    AnyDemand,
    BoundInputs,
    InputDemand,
    InputParam,
    InputSignature,
    Inputs,
    Parallelism,
    Sink,
    SinkDefinition,
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
    Depth,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::providers::{BreaksRequest, FixtureBreaks};
use super::shared::{FileArgs, NoArgs, fixture_format, frame_format};

fixture_source!(File, FileStage, "file", FileArgs, &InputSignature::EMPTY);

fixture_sink!(
    Stdout,
    StdoutStage,
    "stdout",
    NoArgs,
    &InputSignature::new(&[InputParam::required("input")])
);

struct FloatMaskStage;

impl StageInstance for FloatMaskStage {}

impl Sink for FloatMaskStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        Ok(())
    }
}

/// A sink reading a mask, taking it only as F32.
pub(super) struct FloatMask;

impl StageDefinition for FloatMask {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "float-mask"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("mask").accepting(StreamType::Mask)]);
        SIGNATURE
    }
}

impl SinkDefinition for FloatMask {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        _offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        Ok(FormatSet::of([SampleFormat::F32]))
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(FloatMaskStage))
    }
}

opinionated_source!(
    Wide,
    WideStage,
    "wide",
    FormatSet::of([SampleFormat::U8, SampleFormat::F32])
);
opinionated_source!(
    Deep,
    DeepStage,
    "deep",
    FormatSet::of([SampleFormat::U16 { depth: Depth::TEN }, SampleFormat::F32])
);
opinionated_source!(Float, FloatStage, "float", FormatSet::of([SampleFormat::F32]));
// FloatFirst would rather emit the wider of its two samples.
opinionated_source!(
    FloatFirst,
    FloatFirstStage,
    "floatfirst",
    FormatSet::of([SampleFormat::F32, SampleFormat::U8])
);

opinionated_source!(Eight, EightStage, "eight", FormatSet::of([SampleFormat::U8]));
opinionated_source!(
    Twelve,
    TwelveStage,
    "twelve",
    FormatSet::of([SampleFormat::U16 { depth: Depth::TWELVE }])
);
opinionated_source!(
    Sixteen,
    SixteenStage,
    "sixteen",
    FormatSet::of([SampleFormat::U16 {
        depth: Depth::SIXTEEN
    }])
);
opinionated_source!(
    Grey,
    GreyStage,
    "grey",
    FormatSet::of([SampleFormat::U8]),
    grey_format()
);

fn grey_format() -> FrameFormat {
    FrameFormat {
        layout: FrameLayout::Grey,
        ..frame_format()
    }
}

struct WideningSourceStage;

impl StageInstance for WideningSourceStage {}

impl Source for WideningSourceStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

/// A source whose granularity is per frame, unlike every other fixture source.
pub(super) struct WideningSource;

impl StageDefinition for WideningSource {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "widening-source"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::EMPTY;
        SIGNATURE
    }
}

impl SourceDefinition for WideningSource {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Ok(())
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
        Ok(Box::new(WideningSourceStage))
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub(crate) struct Y4mArgs {
    /// The stage to demand breaks after, which places a provider between it and the sink.
    #[serde(default)]
    breaks_after: Option<String>,
}

struct Y4mStage;

impl StageInstance for Y4mStage {}

impl Sink for Y4mStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs) -> Result<(), StageError> {
        Ok(())
    }
}

/// A stand-in for a y4m sink, wanting the output depth and taking only the samples y4m can carry.
pub(super) struct Y4m;

impl StageDefinition for Y4m {
    type Args = Y4mArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "y4m"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn wants_output_depth() -> bool {
        true
    }

    fn demands(args: &Self::Args) -> Vec<InputDemand> {
        let Some(after) = &args.breaks_after else {
            return Vec::new();
        };

        let request = BreaksRequest {
            every: 4,
            after: Some(after.clone()),
            refuse: false,
        };

        vec![InputDemand {
            input: "input",
            demand: AnyDemand::new::<FixtureBreaks>(request),
        }]
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for Y4m {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let StreamFormat::Frames(format) = offered else {
            return Ok(FormatSet::new());
        };

        let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
        let twelve_bit = SampleFormat::U16 { depth: Depth::TWELVE };
        let samples = match format.layout {
            FrameLayout::Grey => FormatSet::of([SampleFormat::U8, twelve_bit]),
            _ => FormatSet::of([SampleFormat::U8, ten_bit, twelve_bit]),
        };

        Ok(samples)
    }

    fn build(
        _context: &BuildContext,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Sink>, StageError> {
        Ok(Box::new(Y4mStage))
    }
}
