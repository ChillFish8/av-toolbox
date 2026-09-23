use std::sync::atomic::{AtomicUsize, Ordering};

use av_toolbox_core::{
    BoundInputs,
    InputSignature,
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
    FrameContext,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamTemplate,
};

use super::shared::{NoArgs, fixture_format};

/// How many times [Counted](crate::testing::counted::Counted) has opened its input.
pub(crate) static COUNTED_OPENS: AtomicUsize = AtomicUsize::new(0);

struct CountedStage;

impl StageInstance for CountedStage {}

impl Source for CountedStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

pub(super) struct Counted;

impl StageDefinition for Counted {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "counted"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Counted {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        COUNTED_OPENS.fetch_add(1, Ordering::SeqCst);
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
        Ok(Box::new(CountedStage))
    }
}

struct UnopenableStage;

impl StageInstance for UnopenableStage {}

impl Source for UnopenableStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

pub(super) struct Unopenable;

impl StageDefinition for Unopenable {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "unopenable"
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Unopenable {
    type Side = DeviceSide;
    type Opened = ();

    fn open(_args: &Self::Args) -> Result<Self::Opened, StageError> {
        Err(StageError::msg("the input could not be opened"))
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
        Ok(Box::new(UnopenableStage))
    }
}
