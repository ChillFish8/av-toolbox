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
    Channel,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::PlaneName;

const STAGE: &str = "planes/extract";

/// Arguments for `planes/extract`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ExtractArgs {
    /// The plane to pull out, one channel or the chroma pair.
    pub(crate) plane: PlaneName,
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} takes planes from frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn not_yuv(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} takes chroma from YUV frames, not {layout}");

    StageError::msg(message)
}

fn missing_channel(channel: Channel, layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} cannot take {channel} from a {layout} frame");

    StageError::msg(message)
}

fn missing_plane(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The channels `plane` names, in plane order.
fn plane_channels(plane: PlaneName) -> &'static [Channel] {
    match plane {
        PlaneName::Y => &[Channel::Y],
        PlaneName::U => &[Channel::U],
        PlaneName::V => &[Channel::V],
        PlaneName::R => &[Channel::R],
        PlaneName::G => &[Channel::G],
        PlaneName::B => &[Channel::B],
        PlaneName::Chroma => &[Channel::U, Channel::V],
    }
}

/// The input format, once it is known to be frames.
fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

/// The format of the plane `plane` names, once it is checked against `input`.
pub(super) fn extracted(input: &FrameFormat, plane: PlaneName) -> Result<FrameFormat, StageError> {
    if let PlaneName::Chroma = plane {
        let FrameLayout::Yuv { subsampling } = input.layout else {
            return Err(not_yuv(input.layout));
        };

        let (width, height) = subsampling.chroma_dimensions(input.width, input.height);

        return Ok(FrameFormat {
            width,
            height,
            layout: FrameLayout::Chroma { subsampling },
            ..*input
        });
    }

    let channel = plane_channels(plane)[0];
    let Some((width, height)) = input.channel_dimensions(channel) else {
        return Err(missing_channel(channel, input.layout));
    };

    Ok(FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        ..*input
    })
}

/// Pulls one plane, or the chroma pair, out of a clip.
pub(crate) struct Extract;

impl StageDefinition for Extract {
    type Args = ExtractArgs;

    fn package() -> &'static str {
        "planes"
    }

    fn identifier() -> &'static str {
        "extract"
    }

    fn description() -> &'static str {
        "Pulls one plane, or the U and V pair, out of a clip as a clip of its own."
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

impl FilterDefinition for Extract {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        let Some(format) = offered.as_frames() else {
            return Err(not_frames());
        };

        Ok(FormatSet::of([format.sample]))
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        let output = extracted(&input, args.plane)?;
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build(
        _context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let output = extracted(&input, args.plane)?;
        let extractor = Extractor::new(output, args.plane);

        Ok(Box::new(extractor))
    }
}

/// Pulls the planes one `PlaneName` names out of a frame, sharing them rather than copying.
pub(super) struct Extractor {
    output: FrameFormat,
    channels: &'static [Channel],
}

impl Extractor {
    pub(super) fn new(output: FrameFormat, plane: PlaneName) -> Self {
        let channels = plane_channels(plane);

        Self { output, channels }
    }
}

impl StageInstance for Extractor {}

impl Filter for Extractor {
    fn process(&mut self, _context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input());
        };

        let meta = input.meta().clone();
        let mut planes = Vec::with_capacity(self.channels.len());

        for channel in self.channels {
            let Some(plane) = input.plane(*channel) else {
                return Err(missing_plane(*channel));
            };

            planes.push(plane.clone());
        }

        let output = Frame::from_planes(self.output, planes, meta).map_err(bad_frame)?;

        Ok(vec![output])
    }
}
