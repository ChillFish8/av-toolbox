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
    thresholds,
    upload_layout,
};
use av_toolbox_cubek::deblock::qed::{
    ChangeTable,
    QedPlanes,
    lowpass_frame,
    qed_interiors,
    upload_change_table,
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
    FrameLayout,
    FrameMeta,
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

use super::h264::padded_sizes;
use crate::bits::sample_kind;
use crate::checks::{accepts_integer_or_f32_frames, between_i32};
use crate::planes::ChannelName;
use crate::scratch::{Scratch, device_plane};
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

pub(crate) const STAGE: &str = "deblock/qed";

const MAX_QUANT: i32 = 60;

fn default_quant() -> Vec<i32> {
    vec![24, 26]
}

fn default_aoffset() -> Vec<i32> {
    vec![1, 1]
}

fn default_boffset() -> Vec<i32> {
    vec![2, 2]
}

/// What the chroma planes of a YUV frame take.
#[derive(Debug, Default, Deserialize, JsonSchema, Copy, Clone, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum QedChroma {
    /// The QED output, on each plane's own 8 by 8 grid.
    #[default]
    Qed,
    /// The normal pass.
    Normal,
    /// The strong pass.
    Strong,
}

/// Arguments for `deblock/qed`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct QedArgs {
    /// How strongly each pass deblocks, `[border, interior]`, each between 0 and 60. The border pass keeps
    /// block borders, and the interior pass is spread into each block's interior.
    #[serde(default = "default_quant")]
    pub(crate) quant: Vec<i32>,
    /// Each pass's `aoffset`, as `deblock/h264` takes it, `[border, interior]`.
    #[serde(default = "default_aoffset")]
    pub(crate) aoffset: Vec<i32>,
    /// Each pass's `boffset`, as `deblock/h264` takes it, `[border, interior]`.
    #[serde(default = "default_boffset")]
    pub(crate) boffset: Vec<i32>,
    /// What chroma takes, `"qed"`, `"normal"` or `"strong"`. RGB takes only `"qed"`.
    #[serde(default)]
    pub(crate) chroma: QedChroma,
    /// The planes to deblock, every plane by default.
    pub(crate) planes: Option<Vec<ChannelName>>,
}

/// How one plane of a frame is deblocked.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Treatment {
    /// Passed through unchanged.
    Untouched,
    /// Both passes, then the low-pass of the strong pass's change into each block's interior.
    Qed,
    /// The normal pass alone.
    NormalPass,
    /// The strong pass alone.
    StrongPass,
}

/// What every frame is deblocked with, once the arguments are checked against the input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct QedDeblocking {
    /// How each plane, in channel order, is deblocked.
    pub(crate) treatments: Vec<Treatment>,
    pub(crate) samples: MatrixSamples,
    /// What the normal pass, which border samples keep, filters against.
    pub(crate) normal: Thresholds,
    /// What the strong pass, whose change spreads into each block's interior, filters against.
    pub(crate) strong: Thresholds,
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

fn wrong_length(name: &str, count: usize) -> StageError {
    let message = format!("{STAGE} takes {name} as two values, [border, interior], not {count}");

    StageError::msg(message)
}

fn chroma_on_rgb(chroma: QedChroma) -> StageError {
    let name = chroma_name(chroma);
    let message = format!("{STAGE} takes only chroma \"qed\" on RGB, not \"{name}\"");

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

fn chroma_name(chroma: QedChroma) -> &'static str {
    match chroma {
        QedChroma::Qed => "qed",
        QedChroma::Normal => "normal",
        QedChroma::Strong => "strong",
    }
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

/// The two values of the pair `name`, refusing any other count.
fn pair(name: &str, values: &[i32]) -> Result<[i32; 2], StageError> {
    let [border, interior] = values else {
        return Err(wrong_length(name, values.len()));
    };

    Ok([*border, *interior])
}

/// How a selected or unselected plane carrying `channel` is deblocked, given what chroma takes.
///
/// U and V follow `chroma`, and every other channel takes the QED output.
fn treatment(channel: Channel, chosen: bool, chroma: QedChroma) -> Treatment {
    if !chosen {
        return Treatment::Untouched;
    }

    let is_chroma = matches!(channel, Channel::U | Channel::V);
    if !is_chroma {
        return Treatment::Qed;
    }

    match chroma {
        QedChroma::Qed => Treatment::Qed,
        QedChroma::Normal => Treatment::NormalPass,
        QedChroma::Strong => Treatment::StrongPass,
    }
}

/// Checks the arguments against the input, giving what every frame is deblocked with.
pub(crate) fn checked_qed(args: &QedArgs, input: &FrameFormat) -> Result<QedDeblocking, StageError> {
    let quant = pair("quant", &args.quant)?;
    let aoffset = pair("aoffset", &args.aoffset)?;
    let boffset = pair("boffset", &args.boffset)?;
    between_i32(STAGE, "quant[0] (border)", quant[0], 0, MAX_QUANT)?;
    between_i32(STAGE, "quant[1] (interior)", quant[1], 0, MAX_QUANT)?;

    let rgb_needs_qed = matches!(input.layout, FrameLayout::Rgb) && args.chroma != QedChroma::Qed;
    if rgb_needs_qed {
        return Err(chroma_on_rgb(args.chroma));
    }

    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::Every)?;

    let Some(samples) = sample_kind(input.sample) else {
        return Err(unsupported_sample(input.sample));
    };

    let channels = input.channels();
    let treatments = channels
        .iter()
        .zip(&selected)
        .map(|(channel, chosen)| treatment(*channel, *chosen, args.chroma))
        .collect();
    let normal = thresholds(quant[0], aoffset[0], boffset[0], samples);
    let strong = thresholds(quant[1], aoffset[1], boffset[1], samples);

    Ok(QedDeblocking {
        treatments,
        samples,
        normal,
        strong,
    })
}

/// The regions one frame's deblock lays out, plane by plane in channel order.
///
/// A QED plane has its normal pass's region then its strong pass's, and a plane taking one pass only that
/// pass's. Each region is the plane's share of the padded frame, as `deblock/h264` pads it.
pub(crate) fn layout_planes(format: &FrameFormat, deblocking: &QedDeblocking) -> Vec<LayoutPlane> {
    let every = vec![true; deblocking.treatments.len()];
    let padded = padded_sizes(format, &every);
    let mut planes = Vec::new();

    for (treatment, (width, height)) in deblocking.treatments.iter().zip(padded) {
        let normal = LayoutPlane {
            width,
            height,
            thresholds: deblocking.normal,
        };
        let strong = LayoutPlane {
            width,
            height,
            thresholds: deblocking.strong,
        };

        match treatment {
            Treatment::Untouched => {},
            Treatment::Qed => planes.extend([normal, strong]),
            Treatment::NormalPass => planes.push(normal),
            Treatment::StrongPass => planes.push(strong),
        }
    }

    planes
}

/// The size each plane of `format` is extended to before the low-pass, its share of a frame whose sides round
/// up to multiples of 16, in channel order.
pub(crate) fn lowpass_sizes(format: &FrameFormat) -> Vec<(u32, u32)> {
    let (lowpass_width, lowpass_height) = lowpass_frame(format.width, format.height);
    let channels = format.channels();

    channels
        .iter()
        .map(|channel| {
            let dimensions = format
                .layout
                .channel_dimensions(*channel, lowpass_width, lowpass_height);

            dimensions.expect("every channel of a format's own layout is carried")
        })
        .collect()
}

/// Removes compression blocking the way Didée's Deblock_QED does, as vsdenoise's `deblock_qed` runs it.
pub(crate) struct Qed;

impl StageDefinition for Qed {
    type Args = QedArgs;

    fn package() -> &'static str {
        "deblock"
    }

    fn identifier() -> &'static str {
        "qed"
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

impl FilterDefinition for Qed {
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
        checked_qed(args, &input)?;

        let format = StreamFormat::Frames(input);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([input.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let deblocking = checked_qed(&args, &input)?;

        let client = context.client().clone();
        let planes = layout_planes(&input, &deblocking);
        let layout = upload_layout(&client, &planes);
        let table = upload_change_table(&client, deblocking.samples);
        let lowpass = lowpass_sizes(&input);
        let deblocker = QedDeblocker {
            client,
            format: input,
            deblocking,
            layout,
            table,
            lowpass,
        };

        Ok(Box::new(deblocker))
    }
}

/// One selected plane of a frame, with its place in channel order, its source and the fresh plane it is
/// deblocked into.
struct PlaneWork<'a> {
    index: usize,
    source: &'a Plane,
    target: Plane,
}

/// Runs both passes of every selected plane in one deblock, then the low-pass into each QED plane's interior,
/// with the layout and the change table uploaded at build.
struct QedDeblocker<R: Runtime> {
    client: ComputeClient<R>,
    format: FrameFormat,
    deblocking: QedDeblocking,
    layout: DeblockLayout,
    table: ChangeTable,
    /// The size each plane, in channel order, is extended to before the low-pass.
    lowpass: Vec<(u32, u32)>,
}

impl<R: Runtime> QedDeblocker<R> {
    /// A fresh plane of `source`'s size in the clip's sample format.
    fn fresh_plane(
        &self,
        context: &FrameContext,
        source: &Plane,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let (width, height) = source.dimensions();
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.format
        };
        let frame = context.alloc_frame(&plane_format, meta.clone());

        let Some(plane) = frame.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        Ok(plane.clone())
    }

    /// A scratch plane for the strong pass of `item`, when it takes the QED output.
    fn strong_plane(&self, context: &FrameContext, item: &PlaneWork<'_>) -> Option<Scratch> {
        let treatment = self.deblocking.treatments[item.index];
        if treatment != Treatment::Qed {
            return None;
        }

        let (width, height) = item.source.dimensions();
        let scratch = match self.deblocking.samples {
            MatrixSamples::U8 => Scratch::new::<u8>(context, width, height),
            MatrixSamples::U16 { .. } => Scratch::new::<u16>(context, width, height),
            MatrixSamples::F32 => Scratch::new::<f32>(context, width, height),
        };

        Some(scratch)
    }

    /// Deblocks every plane of `work` into its target.
    ///
    /// One deblock runs every pass the planes need, the normal pass of a QED plane cropped into its target
    /// and its strong pass into a scratch plane. Then one QED launch per QED plane overwrites the target's
    /// interior.
    fn deblock_planes(&self, context: &FrameContext, work: &[PlaneWork<'_>]) {
        let samples = self.deblocking.samples;
        let strong_planes: Vec<Option<Scratch>> =
            work.iter().map(|item| self.strong_plane(context, item)).collect();
        let mut inputs = Vec::with_capacity(self.layout_regions());
        let mut outputs = Vec::with_capacity(self.layout_regions());

        for (item, strong) in work.iter().zip(&strong_planes) {
            // Safety: the source is a plane of the clip's format, whose sample type is the element `deblock`
            // reads for `samples`.
            let source = unsafe { device_plane(item.source) };

            // Safety: as above, for a plane of a frame allocated in the clip's format.
            let target = unsafe { device_plane(&item.target) };

            inputs.push(source);
            outputs.push(target);

            if let Some(scratch) = strong {
                inputs.push(source);
                outputs.push(scratch.plane());
            }
        }

        let planes = DeblockPlanes {
            inputs: &inputs,
            outputs: &outputs,
        };
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

        for (item, strong) in work.iter().zip(&strong_planes) {
            let Some(strong_scratch) = strong else {
                continue;
            };

            // Safety: as above.
            let source = unsafe { device_plane(item.source) };

            // Safety: as above.
            let output = unsafe { device_plane(&item.target) };

            let qed_planes = QedPlanes {
                source,
                strong: strong_scratch.plane(),
                output,
            };
            let lowpass_size = self.lowpass[item.index];

            qed_interiors(
                &self.client,
                &sizing,
                qed_planes,
                lowpass_size,
                &self.table,
                samples,
            );
        }
    }

    /// How many regions the layout holds, two for a QED plane and one for a plane taking one pass.
    fn layout_regions(&self) -> usize {
        self.deblocking
            .treatments
            .iter()
            .map(|treatment| match treatment {
                Treatment::Untouched => 0,
                Treatment::Qed => 2,
                Treatment::NormalPass | Treatment::StrongPass => 1,
            })
            .sum()
    }
}

impl<R: Runtime> StageInstance for QedDeblocker<R> {}

impl<R: Runtime> Filter for QedDeblocker<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let channels = self.format.channels();
        let mut planes = Vec::with_capacity(channels.len());
        let mut work = Vec::new();

        for (index, channel) in channels.iter().enumerate() {
            let Some(source) = input.plane(*channel) else {
                return Err(missing_channel(*channel));
            };

            if self.deblocking.treatments[index] == Treatment::Untouched {
                planes.push(source.clone());
                continue;
            }

            let target = self.fresh_plane(context, source, &meta)?;

            planes.push(target.clone());
            work.push(PlaneWork {
                index,
                source,
                target,
            });
        }

        self.deblock_planes(context, &work);

        match Frame::from_planes(self.format, planes, meta) {
            Ok(frame) => Ok(vec![frame]),
            Err(error) => Err(bad_frame(error)),
        }
    }
}
