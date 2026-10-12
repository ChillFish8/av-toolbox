use av_toolbox_core::{
    BoundInputs,
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
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Depth,
    DeviceSide,
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

use super::shared::{FileArgs, NoArgs, fixture_format, frame_format};

fixture_source!(File, FileStage, "file", FileArgs, &InputSignature::EMPTY);

fixture_sink!(
    Stdout,
    StdoutStage,
    "stdout",
    NoArgs,
    &InputSignature::new(&[InputParam::required("input")])
);

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
// FloatFirst would rather emit the wider of its two samples, so its preference and its narrowest
// sample are not the same.
opinionated_source!(
    FloatFirst,
    FloatFirstStage,
    "floatfirst",
    FormatSet::of([SampleFormat::F32, SampleFormat::U8])
);

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

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _opened: Self::Opened,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Source>, StageError> {
        Ok(Box::new(WideningSourceStage))
    }
}
