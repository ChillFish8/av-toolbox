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
use av_toolbox_cubek::deblock::pp7::{
    MAX_QP,
    MIN_QP,
    Pp7Mode,
    Pp7Planes,
    ThresholdTable,
    aligned_frame,
    pp7_plane,
    upload_thresholds,
};
use av_toolbox_cubek::matrix::MatrixSamples;
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use schemars::JsonSchema;
use serde::Deserialize;

use super::planes::{channel_shares, plane_like};
use crate::bits::sample_kind;
use crate::checks::{accepts_integer_or_f32_frames, between_f64};
use crate::planes::ChannelName;
use crate::scratch::device_plane;
use crate::selection::{PlaneDefault, selected_planes};

pub(crate) const STAGE: &str = "deblock/pp7";

const DEFAULT_QP: f64 = 2.0;

fn default_qp() -> f64 {
    DEFAULT_QP
}

/// How a frequency strong enough to pass its threshold is kept.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Mode {
    /// Kept as it is.
    #[default]
    Hard,
    /// Moved toward zero by its threshold, for a softer result.
    Soft,
    /// Twice its distance past the threshold, and kept as it is past twice the threshold.
    Medium,
}

impl Mode {
    fn kernel_mode(self) -> Pp7Mode {
        match self {
            Self::Hard => Pp7Mode::Hard,
            Self::Soft => Pp7Mode::Soft,
            Self::Medium => Pp7Mode::Medium,
        }
    }
}

/// Arguments for `deblock/pp7`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pp7Args {
    /// How strongly to deblock, between 1 and 63. Fractions are used as they are.
    #[serde(default = "default_qp")]
    pub(crate) qp: f64,
    /// How each frequency past its threshold is kept, `"hard"`, `"soft"` or `"medium"`.
    #[serde(default)]
    pub(crate) mode: Mode,
    /// The planes to deblock, every plane by default.
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// What every frame is deblocked with, once the arguments are checked against the input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Pp7Deblocking {
    /// Whether each plane, in channel order, is deblocked.
    pub(crate) selected: Vec<bool>,
    pub(crate) samples: MatrixSamples,
    pub(crate) qp: f64,
    pub(crate) mode: Pp7Mode,
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} deblocks frames, not masks");

    StageError::msg(message)
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn unsupported_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} deblocks u8, u16 or f32 samples, not {sample:?}");

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
pub(crate) fn checked_pp7(args: &Pp7Args, input: &FrameFormat) -> Result<Pp7Deblocking, StageError> {
    between_f64(STAGE, "qp", args.qp, MIN_QP, MAX_QP)?;

    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::Every)?;

    let Some(samples) = sample_kind(input.sample) else {
        return Err(unsupported_sample(input.sample));
    };

    let mode = args.mode.kernel_mode();

    Ok(Pp7Deblocking {
        selected,
        samples,
        qp: args.qp,
        mode,
    })
}

/// The size each plane of `format` is aligned to before its border folds back a second time, its share of a
/// frame whose sides round up to multiples of 16, in channel order.
pub(crate) fn aligned_sizes(format: &FrameFormat) -> Vec<(u32, u32)> {
    let (aligned_width, aligned_height) = aligned_frame(format.width, format.height);

    channel_shares(format, aligned_width, aligned_height)
}

/// Removes compression blocking with pp7, the way VapourSynth's `pp7.DeblockPP7` does.
pub(crate) struct Pp7;

impl StageDefinition for Pp7 {
    type Args = Pp7Args;

    fn package() -> &'static str {
        "deblock"
    }

    fn identifier() -> &'static str {
        "pp7"
    }

    fn description() -> &'static str {
        "Removes compression blocking by dropping weak frequency detail, as pp7 does."
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

impl FilterDefinition for Pp7 {
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
        checked_pp7(args, &input)?;

        let format = StreamFormat::Frames(input);
        let template = StreamTemplate::from_format(&format);
        let samples = FormatSet::of([input.sample]);

        Ok(OutputFormats { template, samples })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let deblocking = checked_pp7(&args, &input)?;

        let client = context.client().clone();
        let table = upload_thresholds(&client, deblocking.qp, deblocking.samples);
        let aligned = aligned_sizes(&input);
        let deblocker = Pp7Deblocker {
            client,
            format: input,
            deblocking,
            table,
            aligned,
        };

        Ok(Box::new(deblocker))
    }
}

/// Deblocks the selected planes of every frame, one launch per plane, with the thresholds uploaded at build.
struct Pp7Deblocker<R: Runtime> {
    client: ComputeClient<R>,
    format: FrameFormat,
    deblocking: Pp7Deblocking,
    table: ThresholdTable,
    /// The size each plane, in channel order, is aligned to before its border folds back a second time.
    aligned: Vec<(u32, u32)>,
}

impl<R: Runtime> Pp7Deblocker<R> {
    /// Deblocks `source`, the plane at `index` in channel order, into the same-sized `target`.
    fn deblock_plane(&self, index: usize, source: &Plane, target: &Plane) {
        // Safety: the source is a plane of the clip's format, whose sample type is the element `pp7_plane`
        // reads for `self.deblocking.samples`.
        let input = unsafe { device_plane(source) };

        // Safety: as above, for a plane of a frame allocated in the clip's format.
        let output = unsafe { device_plane(target) };

        let planes = Pp7Planes { input, output };
        let aligned_size = self.aligned[index];

        pp7_plane(
            &self.client,
            planes,
            aligned_size,
            &self.table,
            self.deblocking.mode,
            self.deblocking.samples,
        );
    }
}

impl<R: Runtime> StageInstance for Pp7Deblocker<R> {}

impl<R: Runtime> Filter for Pp7Deblocker<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let channels = self.format.channels();
        let mut planes = Vec::with_capacity(channels.len());

        for (index, channel) in channels.iter().enumerate() {
            let Some(source) = input.plane(*channel) else {
                return Err(missing_channel(*channel));
            };

            if !self.deblocking.selected[index] {
                planes.push(source.clone());
                continue;
            }

            let Some(target) = plane_like(context, &self.format, source, &meta) else {
                return Err(missing_channel(*channel));
            };

            self.deblock_plane(index, source, &target);
            planes.push(target);
        }

        match Frame::from_planes(self.format, planes, meta) {
            Ok(frame) => Ok(vec![frame]),
            Err(error) => Err(bad_frame(error)),
        }
    }
}
