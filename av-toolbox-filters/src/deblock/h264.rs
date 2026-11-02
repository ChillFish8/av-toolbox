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
use av_toolbox_cubek::deblock::h264::{
    DeblockLayout,
    DeblockPlanes,
    LayoutPlane,
    Thresholds,
    deblock,
    padded_frame,
    thresholds,
    upload_layout,
};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    Client,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameMeta,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::planes::plane_like;
use crate::bits::sample_kind;
use crate::checks::{accepts_integer_or_f32_frames, between_i32};
use crate::planes::ChannelName;
use crate::scratch::device_plane;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

pub(crate) const STAGE: &str = "deblock/h264";

const DEFAULT_QUANT: i32 = 25;

const MAX_QUANT: i32 = 60;

fn default_quant() -> i32 {
    DEFAULT_QUANT
}

/// Arguments for `deblock/h264`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct H264Args {
    /// How strongly to deblock, between 0 and 60. With no offsets, nothing changes below 16.
    #[serde(default = "default_quant")]
    pub(crate) quant: i32,
    /// Moves how large a step across an edge still counts as blocking, and how far samples may move.
    #[serde(default)]
    pub(crate) aoffset: i32,
    /// Moves how flat each side of an edge must be.
    #[serde(default)]
    pub(crate) boffset: i32,
    /// The planes to deblock, every plane by default.
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// What every frame is deblocked with, once the arguments are checked against the input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Deblocking {
    /// Whether each plane, in channel order, is deblocked.
    pub(crate) selected: Vec<bool>,
    pub(crate) samples: MatrixSamples,
    pub(crate) thresholds: Thresholds,
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} deblocks frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn missing_frame() -> StageError {
    let message = format!("{STAGE} expected a frame in slot 0");

    StageError::msg(message)
}

fn missing_channel(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The input format, once it is known to be frames.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    Ok(*format)
}

/// U8, U16 and F32 frames, refusing masks.
pub(crate) fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(samples) = accepts_integer_or_f32_frames(offered) else {
        return Err(not_frames());
    };

    Ok(samples)
}

/// Checks the arguments against the input, giving what every frame is deblocked with.
pub(crate) fn checked_deblocking(args: &H264Args, input: &FrameFormat) -> Result<Deblocking, StageError> {
    between_i32(STAGE, "quant", args.quant, 0, MAX_QUANT)?;

    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::Every)?;

    let samples = sample_kind(input.sample);

    let frame_thresholds = thresholds(args.quant, args.aoffset, args.boffset, samples);

    Ok(Deblocking {
        selected,
        samples,
        thresholds: frame_thresholds,
    })
}

/// The size each selected plane of `format` is padded to, its share of the padded frame, in channel order.
pub(crate) fn padded_sizes(format: &FrameFormat, selected: &[bool]) -> Vec<(u32, u32)> {
    let (padded_width, padded_height) = padded_frame(format.width, format.height);
    let channels = format.channels();

    channels
        .iter()
        .zip(selected)
        .filter(|(_, chosen)| **chosen)
        .map(|(channel, _)| {
            let dimensions = format
                .layout
                .channel_dimensions(*channel, padded_width, padded_height);

            dimensions.expect("every channel of a format's own layout is carried")
        })
        .collect()
}

/// One layout plane for each of the `padded` sizes, all filtered against `plane_thresholds`.
fn uniform_layout(padded: &[(u32, u32)], plane_thresholds: Thresholds) -> Vec<LayoutPlane> {
    padded
        .iter()
        .map(|(width, height)| LayoutPlane {
            width: *width,
            height: *height,
            thresholds: plane_thresholds,
        })
        .collect()
}

/// Removes compression blocking with H.264's in-loop deblocking filter, the way VapourSynth's
/// `deblock.Deblock` does.
pub(crate) struct H264;

impl StageDefinition for H264 {
    type Args = H264Args;

    fn package() -> &'static str {
        "deblock"
    }

    fn identifier() -> &'static str {
        "h264"
    }

    fn description() -> &'static str {
        "Removes compression blocking with H.264's in-loop deblocking filter."
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

impl FilterDefinition for H264 {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepted_samples(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        checked_deblocking(args, &input)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([input.sample]),
        })
    }

    fn build(
        context: &BuildContext,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let deblocking = checked_deblocking(&args, &input)?;

        let client = context.client().clone();
        let padded = padded_sizes(&input, &deblocking.selected);
        let layout_planes = uniform_layout(&padded, deblocking.thresholds);
        let layout = upload_layout(&client, &layout_planes);
        let deblocker = Deblocker {
            client,
            format: input,
            deblocking,
            layout,
        };

        Ok(Box::new(deblocker))
    }
}

/// Deblocks the selected planes of every frame, with one wavefront launch covering all of them and the
/// layout uploaded at build.
struct Deblocker {
    client: Client,
    format: FrameFormat,
    deblocking: Deblocking,
    layout: DeblockLayout,
}

impl Deblocker {
    /// A fresh plane of `source`'s size in the clip's sample format.
    fn fresh_plane(
        &self,
        context: &FrameContext,
        source: &Plane,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let Some(plane) = plane_like(context, &self.format, source, meta) else {
            return Err(missing_channel(Channel::Y));
        };

        Ok(plane)
    }

    /// Deblocks `sources`, the selected planes in channel order, into the same-sized `targets`.
    fn deblock_selected(&self, context: &FrameContext, sources: &[&Plane], targets: &[&Plane]) {
        let mut inputs = Vec::with_capacity(sources.len());
        let mut outputs = Vec::with_capacity(targets.len());

        for (source, target) in sources.iter().zip(targets) {
            // Safety: the source is a plane of the clip's format, whose sample type is the element `deblock`
            // reads for `self.deblocking.samples`.
            let input = unsafe { device_plane(source) };

            // Safety: as above, for a plane of a frame allocated in the clip's format.
            let output = unsafe { device_plane(target) };

            inputs.push(input);
            outputs.push(output);
        }

        let planes = DeblockPlanes {
            inputs: &inputs,
            outputs: &outputs,
        };
        let samples = self.deblocking.samples;
        let bytes = self.layout.bytes();
        let scratch = context.scratch(bytes);
        let sizing = Sizing(context);

        deblock(
            &self.client,
            &sizing,
            planes,
            scratch.handle(),
            &self.layout,
            samples,
        );
    }
}

impl StageInstance for Deblocker {}

impl Filter for Deblocker {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let channels = self.format.channels();
        let mut planes = Vec::with_capacity(channels.len());
        let mut sources = Vec::new();
        let mut targets = Vec::new();

        for (channel, chosen) in channels.iter().zip(&self.deblocking.selected) {
            let Some(source) = input.plane(*channel) else {
                return Err(missing_channel(*channel));
            };

            if !*chosen {
                planes.push(source.clone());
                continue;
            }

            let target = self.fresh_plane(context, source, &meta)?;

            sources.push(source);
            planes.push(target.clone());
            targets.push(target);
        }

        let target_refs: Vec<&Plane> = targets.iter().collect();
        self.deblock_selected(context, &sources, &target_refs);

        match Frame::from_planes(self.format, planes, meta) {
            Ok(frame) => Ok(vec![frame]),
            Err(error) => Err(bad_frame(error)),
        }
    }
}
