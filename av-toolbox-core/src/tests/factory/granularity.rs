use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    DeviceSide,
    FormatSet,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::{ExampleFilter, reference_input};
use crate::factory::Definition;
use crate::testing::FixtureCuts;
use crate::{
    AnyDemand,
    BoundInputs,
    Filter,
    FilterDefinition,
    InputArgument,
    InputParam,
    InputSignature,
    OpenSlot,
    Parallelism,
    PreparedFilter,
    PreparedStage,
    StageDefinition,
    StageError,
    StageFactory,
    StageHandle,
    StageKind,
    StreamType,
};

fn fixture_cuts() -> AnyDemand {
    AnyDemand::new::<FixtureCuts>(())
}

#[derive(Debug, Deserialize, JsonSchema)]
struct TemporalArgs {
    radius: u32,
}

struct TemporalExample;

impl StageDefinition for TemporalExample {
    type Args = TemporalArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "temporal"
    }

    fn parallelism(args: &Self::Args) -> Parallelism {
        if args.radius == 0 {
            Parallelism::Frame
        } else {
            Parallelism::Chunk(fixture_cuts())
        }
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
        SIGNATURE
    }
}

impl FilterDefinition for TemporalExample {
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
        Ok(Box::new(ExampleFilter))
    }
}

#[test]
fn a_prepared_stage_reports_the_granularity_its_arguments_asked_for() {
    let definition = Definition::<TemporalExample>::new(StageKind::Filter).expect("should build");

    let wide = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({ "radius": 0 }),
        &reference_input(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    let narrow = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({ "radius": 4 }),
        &reference_input(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    assert_eq!(wide.parallelism(), Parallelism::Frame);
    assert_eq!(narrow.parallelism(), Parallelism::Chunk(fixture_cuts()));

    let erased = PreparedStage::Filter(narrow);
    assert_eq!(erased.parallelism(), Parallelism::Chunk(fixture_cuts()));
}

#[derive(Debug, Deserialize, JsonSchema)]
struct ChunkAuthorityArgs {}

struct ChunkAuthorityExample;

impl StageDefinition for ChunkAuthorityExample {
    type Args = ChunkAuthorityArgs;

    fn package() -> &'static str {
        "av-denoise"
    }

    fn identifier() -> &'static str {
        "chunk-authority"
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature = &InputSignature::new(&[
            InputParam::required("primary"),
            InputParam::required("secondary").chunk_authority(),
        ]);
        SIGNATURE
    }
}

impl FilterDefinition for ChunkAuthorityExample {
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
        Ok(Box::new(ExampleFilter))
    }
}

fn two_inputs() -> Vec<InputArgument> {
    vec![
        InputArgument::positional(StageHandle::new(1), StreamType::Frames),
        InputArgument::positional(StageHandle::new(2), StreamType::Frames),
    ]
}

#[test]
fn chunk_authority_survives_type_erasure_when_a_later_input_is_marked() {
    let definition = Definition::<ChunkAuthorityExample>::new(StageKind::Filter).expect("should build");

    let prepared: Box<dyn PreparedFilter> = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({}),
        &two_inputs(),
        &OpenSlot::new(),
    )
    .expect("should prepare");

    assert_eq!(prepared.chunk_authority(), Some(1));
}
