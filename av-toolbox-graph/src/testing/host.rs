use av_toolbox_core::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputParam,
    InputSignature,
    Inputs,
    MetaInputs,
    MetaItem,
    MetaSink,
    MetaSinkDefinition,
    MetaTransform,
    MetaTransformDefinition,
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
    FormatSet,
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

use super::shared::{NoArgs, first_bound, fixture_format, frames_template};

struct HostFileStage;

impl StageInstance for HostFileStage {}

impl Source<HostSide> for HostFileStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<HostFrame>, StageError> {
        Ok(None)
    }
}

pub(super) struct HostFile;

impl StageDefinition for HostFile {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-file"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for HostFile {
    type Side = HostSide;
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
    ) -> Result<Box<dyn Source<HostSide>>, StageError> {
        Ok(Box::new(HostFileStage))
    }
}

struct HostBlurStage;

impl StageInstance for HostBlurStage {}

impl Filter<HostSide> for HostBlurStage {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: Inputs<HostSide>,
    ) -> Result<HostFrame, StageError> {
        inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))
    }
}

pub(super) struct HostBlur;

impl StageDefinition for HostBlur {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-blur"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("reference").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for HostBlur {
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
    ) -> Result<Box<dyn Filter<HostSide>>, StageError> {
        Ok(Box::new(HostBlurStage))
    }
}

struct HostStdoutStage;

impl StageInstance for HostStdoutStage {}

impl Sink<HostSide> for HostStdoutStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: Inputs<HostSide>) -> Result<(), StageError> {
        Ok(())
    }
}

pub(super) struct HostStdout;

impl StageDefinition for HostStdout {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "host-stdout"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("input")]);
        SIGNATURE
    }
}

impl SinkDefinition for HostStdout {
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
        Ok(Box::new(HostStdoutStage))
    }
}

struct RestampStage;

impl StageInstance for RestampStage {}

impl MetaTransform for RestampStage {
    fn process(
        &mut self,
        _context: &mut FrameContext,
        mut inputs: MetaInputs,
    ) -> Result<Vec<MetaItem>, StageError> {
        Ok(inputs.take(0).into_iter().collect())
    }
}

pub(super) struct Restamp;

impl StageDefinition for Restamp {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "restamp"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Frame
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("input").accepting(StreamType::Frames),
            InputParam::optional("guide")
                .accepting(StreamType::Frames)
                .chunk_authority(),
        ]);
        SIGNATURE
    }
}

impl MetaTransformDefinition for Restamp {
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
    ) -> Result<Box<dyn MetaTransform>, StageError> {
        Ok(Box::new(RestampStage))
    }
}

struct LogStage;

impl StageInstance for LogStage {}

impl MetaSink for LogStage {
    fn consume(&mut self, _context: &mut FrameContext, _inputs: MetaInputs) -> Result<(), StageError> {
        Ok(())
    }
}

pub(super) struct Log;

impl StageDefinition for Log {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "log"
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

impl MetaSinkDefinition for Log {
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
        Ok(Box::new(LogStage))
    }
}
