mod args;

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
    Tag,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
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
};

pub(crate) use self::args::QtgmcArgs;
use self::args::checked_field_size;
use crate::aa::direction::{Direction, checked_dimensions};
use crate::checks::accepts_frames;
use crate::deinterlace::bwdif::{output_format, resolved_order};
use crate::deinterlace::spatial::{checked_predictor, checked_predictor_planes};
use crate::masked::{INPUT, checked_input, missing_input};

pub(crate) const STAGE: &str = "deinterlace/qtgmc";

fn not_frames() -> StageError {
    let message = format!("{STAGE} deinterlaces frames, not masks");

    StageError::msg(message)
}

fn unsupported_format(input: &FrameFormat) -> StageError {
    let message = format!(
        "{STAGE} needs integer YUV or Gray frames, got {:?} samples in a {:?} layout",
        input.sample, input.layout
    );

    StageError::msg(message)
}

fn not_processing() -> StageError {
    let message = format!("{STAGE} cannot process frames yet, its processing step is not written");

    StageError::msg(message)
}

/// Refuses a format that is not integer YUV or Gray.
fn checked_format(input: &FrameFormat) -> Result<(), StageError> {
    let is_integer = matches!(input.sample, SampleFormat::U8 | SampleFormat::U16 { .. });
    let is_layout = matches!(input.layout, FrameLayout::Yuv { .. } | FrameLayout::Grey);
    if !is_integer || !is_layout {
        return Err(unsupported_format(input));
    }

    Ok(())
}

/// Refuses `args` that `input` cannot run with.
fn checked_stage(args: &QtgmcArgs, input: &FrameFormat) -> Result<(), StageError> {
    args.checked_values()?;
    resolved_order(STAGE, args.order, input)?;
    checked_format(input)?;

    let selected = vec![true; input.channels().len()];
    checked_dimensions(STAGE, input, &selected, Direction::Vertical)?;
    checked_field_size(input)?;

    let predictor = checked_predictor(STAGE, &args.spatial)?;
    checked_predictor_planes(STAGE, predictor, input, &selected)
}

/// Rebuilds progressive frames from interlaced fields, the way QTGMC does.
pub(crate) struct Qtgmc;

impl StageDefinition for Qtgmc {
    type Args = QtgmcArgs;

    fn package() -> &'static str {
        "deinterlace"
    }

    fn identifier() -> &'static str {
        "qtgmc"
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Deinterlace]
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk
    }

    // The fields the output trails the input by, and the frames they and the current frame come from.
    fn held_frames(args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        args.held_frames()
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Qtgmc {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_frames(offered).ok_or_else(not_frames)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        checked_stage(args, &input)?;

        let output = output_format(&input, args.rate)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([input.sample]),
        })
    }

    fn build<R: Runtime>(
        _context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(STAGE, inputs)?;
        checked_stage(&args, &input)?;

        Ok(Box::new(Deinterlacer))
    }
}

/// Runs `deinterlace/qtgmc` over a chunk.
struct Deinterlacer;

impl StageInstance for Deinterlacer {}

impl Filter for Deinterlacer {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(_next) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        Err(not_processing())
    }
}
