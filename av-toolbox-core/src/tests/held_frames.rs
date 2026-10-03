use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ColourInfo,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
    Subsampling,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::factory::Definition;
use crate::{
    BoundInputs,
    Filter,
    FilterDefinition,
    InputArgument,
    InputParam,
    InputSignature,
    Inputs,
    OpenSlot,
    PreparedFilter,
    PreparedStage,
    StageDefinition,
    StageError,
    StageFactory,
    StageHandle,
    StageInstance,
    StageKind,
    StreamType,
};

#[derive(Debug, Deserialize, JsonSchema)]
struct NoArgs {}

struct PassThrough;

impl StageInstance for PassThrough {}

impl Filter for PassThrough {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let frame = inputs
            .take_frame(0)
            .ok_or_else(|| StageError::msg("expected a frame in slot 0"))?;

        Ok(vec![frame])
    }
}

fn signature() -> &'static InputSignature {
    const SIGNATURE: &InputSignature = &InputSignature::new(&[InputParam::required("reference")]);
    SIGNATURE
}

fn accept_offered(offered: StreamFormat) -> Result<FormatSet, StageError> {
    Ok(FormatSet::of([offered.sample()]))
}

fn same_as_input(chosen: &[Option<StreamFormat>]) -> Result<OutputFormats, StageError> {
    let format = chosen.iter().flatten().next().copied().expect("an input");

    Ok(OutputFormats {
        template: StreamTemplate::from_format(&format),
        samples: FormatSet::of([format.sample()]),
    })
}

struct HoldingNothing;

impl StageDefinition for HoldingNothing {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "holding-nothing"
    }

    fn inputs() -> &'static InputSignature {
        signature()
    }
}

impl FilterDefinition for HoldingNothing {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accept_offered(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        same_as_input(chosen)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassThrough))
    }
}

/// Holds one frame for every 100 samples of output width.
struct WidthHolding;

impl StageDefinition for WidthHolding {
    type Args = NoArgs;

    fn package() -> &'static str {
        "test"
    }

    fn identifier() -> &'static str {
        "width-holding"
    }

    fn inputs() -> &'static InputSignature {
        signature()
    }

    fn held_frames(_args: &Self::Args, formats: &ResolvedFormats) -> usize {
        let Some(output) = formats.output() else {
            return 0;
        };
        let (width, _) = output.dimensions();

        width as usize / 100
    }
}

impl FilterDefinition for WidthHolding {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accept_offered(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        same_as_input(chosen)
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        _args: Self::Args,
        _inputs: BoundInputs,
        _formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        Ok(Box::new(PassThrough))
    }
}

fn prepare<D>() -> PreparedStage
where
    D: FilterDefinition<Args = NoArgs> + 'static,
{
    let definition = Definition::<D>::new(StageKind::Filter).expect("should build the definition");
    let inputs = vec![InputArgument::positional(StageHandle::new(1), StreamType::Frames)];
    let prepared: Box<dyn PreparedFilter> = StageFactory::<dyn PreparedFilter>::prepare(
        &definition,
        &serde_json::json!({}),
        &inputs,
        &OpenSlot::new(),
    )
    .expect("should prepare");

    PreparedStage::Filter(prepared)
}

fn wide_formats() -> ResolvedFormats {
    let format = StreamFormat::Frames(FrameFormat {
        width: 640,
        height: 360,
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    });

    ResolvedFormats::new(vec![Some(format)], Some(format))
}

#[test]
fn a_stage_holds_no_frames_unless_it_says_so() {
    let prepared = prepare::<HoldingNothing>();

    assert_eq!(prepared.held_frames(&wide_formats()), 0);
}

#[test]
fn a_stage_reads_its_held_frames_from_its_formats() {
    let prepared = prepare::<WidthHolding>();

    assert_eq!(prepared.held_frames(&wide_formats()), 6);
}
