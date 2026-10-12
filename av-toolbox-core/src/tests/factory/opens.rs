use std::sync::atomic::{AtomicUsize, Ordering};

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
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{capabilities, frame_format};
use crate::factory::Definition;
use crate::{
    BoundInputs,
    InputSignature,
    OpenSlot,
    PreparedSource,
    Source,
    SourceDefinition,
    StageDefinition,
    StageError,
    StageFactory,
    StageInstance,
    StageKind,
};

/// How many times each counter has been opened, one per test so tests running at once never share.
static COUNTED_OPENS: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];

const OPENED_ONCE: usize = 0;

const OPENED_AGAIN: usize = 1;

const FAILED_OPEN: usize = 2;

#[derive(Debug, Deserialize, JsonSchema)]
struct CountedArgs {
    fail: bool,
    counter: usize,
}

struct CountedStage;

impl StageInstance for CountedStage {}

impl Source for CountedStage {
    fn next(&mut self, _context: &mut FrameContext) -> Result<Option<Frame>, StageError> {
        Ok(None)
    }
}

struct Counted;

impl StageDefinition for Counted {
    type Args = CountedArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "counted"
    }

    fn description() -> &'static str {
        "A stage used by tests."
    }

    fn inputs() -> &'static InputSignature {
        &InputSignature::EMPTY
    }
}

impl SourceDefinition for Counted {
    type Side = DeviceSide;
    type Opened = StreamFormat;

    fn open(args: &Self::Args) -> Result<Self::Opened, StageError> {
        COUNTED_OPENS[args.counter].fetch_add(1, Ordering::SeqCst);

        if args.fail {
            return Err(StageError::msg("the input could not be opened"));
        }

        Ok(StreamFormat::Frames(frame_format()))
    }

    fn produces(_capabilities: &Capabilities, opened: &Self::Opened) -> Result<OutputFormats, StageError> {
        Ok(OutputFormats {
            template: StreamTemplate::from_format(opened),
            samples: FormatSet::of([opened.sample()]),
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

fn counted(slot: &OpenSlot, fail: bool, counter: usize) -> Box<dyn PreparedSource> {
    let definition = Definition::<Counted>::new(StageKind::Source).expect("should build");
    let args = serde_json::json!({ "fail": fail, "counter": counter });

    StageFactory::<dyn PreparedSource>::prepare(&definition, &args, &[], slot).expect("should prepare")
}

#[test]
fn a_source_prepared_twice_on_one_slot_opens_once() {
    let slot = OpenSlot::new();
    let capabilities = capabilities();

    counted(&slot, false, OPENED_ONCE)
        .produces(&capabilities)
        .expect("produces");
    counted(&slot, false, OPENED_ONCE)
        .produces(&capabilities)
        .expect("produces again");

    assert_eq!(COUNTED_OPENS[OPENED_ONCE].load(Ordering::SeqCst), 1);
}

#[test]
fn a_source_on_a_fresh_slot_opens_again() {
    let capabilities = capabilities();

    counted(&OpenSlot::new(), false, OPENED_AGAIN)
        .produces(&capabilities)
        .expect("produces");
    counted(&OpenSlot::new(), false, OPENED_AGAIN)
        .produces(&capabilities)
        .expect("produces");

    assert_eq!(COUNTED_OPENS[OPENED_AGAIN].load(Ordering::SeqCst), 2);
}

#[test]
fn a_failed_open_is_the_error_produces_returns() {
    let error = counted(&OpenSlot::new(), true, FAILED_OPEN)
        .produces(&capabilities())
        .expect_err("the open fails");

    assert!(error.to_string().contains("could not be opened"));
}
