use std::time::Duration;

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
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::aa::{Eedi2Settings, Eedi3DeviceClips, Eedi3Settings, sangnom_aaf, sangnom_max};
use av_toolbox_cubek::deinterlace::{BWDIF_MIN_ROWS, BwdifField, BwdifPlanes, bwdif_field};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::values::{narrow, widen};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
    DeviceSide,
    FieldOrder,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
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

use crate::aa::direction::{Direction, FieldPass, checked_dimensions, missing_plane};
use crate::aa::eedi2::{Eedi2Pass, subsampling_shifts};
use crate::aa::eedi3::Eedi3FieldPass;
use crate::aa::sangnom::SangnomPass;
use crate::bits::bits_of;
use crate::deinterlace::spatial::{Predictor, Spatial, checked_predictor, checked_predictor_planes};
use crate::masked::{INPUT, checked_input, missing_input};
use crate::planes::ChannelName;
use crate::scratch::Scratch;
use crate::selection::{PlaneDefault, selected_planes};
use crate::sizing::Sizing;

pub(crate) const STAGE: &str = "deinterlace/bwdif";

/// The field a stream shows first.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Order {
    Top,
    Bottom,
}

/// How many frames come out for each frame in.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Rate {
    /// One frame for each frame in, from its first field.
    Single,
    /// One frame for each field, at twice the input's frame rate.
    #[default]
    Double,
}

/// Arguments for `deinterlace/bwdif`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct BwdifArgs {
    /// The field shown first, the stream's own field order when not given.
    #[serde(default)]
    pub(crate) order: Option<Order>,
    /// One frame for each frame in, or one for each field, one for each field when not given.
    #[serde(default)]
    pub(crate) rate: Rate,
    /// The planes to deinterlace, every plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Where each rebuilt row's spatial prediction comes from, bwdif's own cubic when not given.
    #[serde(default)]
    pub(crate) spatial: Spatial,
}

fn describe_order(found: Option<FieldOrder>) -> String {
    match found {
        Some(order) => order.to_string(),
        None => "unknown".to_owned(),
    }
}

fn unusable_order(found: Option<FieldOrder>) -> StageError {
    let described = describe_order(found);
    let message =
        format!("{STAGE} needs an order of top or bottom, since the stream's field order is {described}");

    StageError::msg(message)
}

fn too_short(channel: Channel, height: u32) -> StageError {
    let message =
        format!("{STAGE} needs plane {channel} to have at least {BWDIF_MIN_ROWS} rows, got {height}");

    StageError::msg(message)
}

fn rate_too_large(rate: FrameRate) -> StageError {
    let message = format!("{STAGE} cannot double a frame rate of {rate}, its numerator would not fit");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} deinterlaces frames, not masks");

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

/// The field shown first, `order` when given and the stream's own interlaced order otherwise.
fn checked_order(args: &BwdifArgs, input: &FrameFormat) -> Result<Order, StageError> {
    if let Some(order) = args.order {
        return Ok(order);
    }

    match input.field_order {
        Some(FieldOrder::TopFirst) => Ok(Order::Top),
        Some(FieldOrder::BottomFirst) => Ok(Order::Bottom),
        found => Err(unusable_order(found)),
    }
}

/// Whether each plane of `input` is deinterlaced, once `args` are checked against it.
pub(crate) fn checked_planes(args: &BwdifArgs, input: &FrameFormat) -> Result<Vec<bool>, StageError> {
    let requested = args.planes.as_deref();
    let selected = selected_planes(STAGE, requested, input.layout, PlaneDefault::Every)?;
    checked_dimensions(STAGE, input, &selected, Direction::Vertical)?;

    let channels = input.channels();
    let chosen = channels
        .iter()
        .zip(&selected)
        .filter(|(_, is_selected)| **is_selected);
    for (channel, _) in chosen {
        let Some((_, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(STAGE, *channel));
        };

        if height < BWDIF_MIN_ROWS {
            return Err(too_short(*channel, height));
        }
    }

    Ok(selected)
}

/// Twice `rate`, halving an even denominator before doubling the numerator.
///
/// A reduced rate with an even denominator has an odd numerator, so halving the denominator keeps it reduced.
fn doubled(rate: FrameRate) -> Result<FrameRate, StageError> {
    let numerator = rate.numerator();
    let denominator = rate.denominator();

    let doubled = if denominator.is_multiple_of(2) {
        FrameRate::new(numerator, denominator / 2)
    } else {
        let Some(twice) = numerator.checked_mul(2) else {
            return Err(rate_too_large(rate));
        };

        FrameRate::new(twice, denominator)
    };

    Ok(doubled.expect("twice a valid rate is a valid rate"))
}

/// What comes out for `input` at `rate`.
fn output_format(input: &FrameFormat, rate: Rate) -> Result<FrameFormat, StageError> {
    let frame_rate = match (rate, input.frame_rate) {
        (Rate::Double, Some(input_rate)) => Some(doubled(input_rate)?),
        (_, unchanged) => unchanged,
    };

    Ok(FrameFormat {
        frame_rate,
        field_order: Some(FieldOrder::Progressive),
        ..*input
    })
}

/// Half of one frame at `rate`, the time from a frame's first field to its second.
fn half_frame(rate: Option<FrameRate>) -> Option<Duration> {
    let rate = rate?;
    let denominator = u128::from(rate.denominator());
    let numerator = u128::from(rate.numerator());
    let nanos = denominator * 1_000_000_000 / (2 * numerator);
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
}

fn accepts_frames(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !matches!(format.sample, SampleFormat::U8 | SampleFormat::U16 { .. }) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

/// Rebuilds progressive frames from interlaced fields, the way bwdif does.
pub(crate) struct Bwdif;

impl StageDefinition for Bwdif {
    type Args = BwdifArgs;

    fn package() -> &'static str {
        "deinterlace"
    }

    fn identifier() -> &'static str {
        "bwdif"
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Deinterlace]
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Chunk
    }

    // The current frame, whose outputs wait on the next, and the previous frame they read.
    fn held_frames(_args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        2
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Bwdif {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepts_frames(offered)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        checked_order(args, &input)?;
        let selected = checked_planes(args, &input)?;
        let predictor = checked_predictor(&args.spatial)?;
        checked_predictor_planes(predictor, &input, &selected)?;

        let output = output_format(&input, args.rate)?;
        let format = StreamFormat::Frames(output);

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
        let input = checked_input(STAGE, inputs)?;
        let order = checked_order(&args, &input)?;
        let selected = checked_planes(&args, &input)?;
        let predictor = checked_predictor(&args.spatial)?;
        checked_predictor_planes(predictor, &input, &selected)?;
        let output = output_format(&input, args.rate)?;

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let fields = fields_of(order, args.rate);
        let half_frame = half_frame(input.frame_rate);

        let deinterlacer = Deinterlacer {
            client: context.client().clone(),
            input,
            output,
            selected,
            predictor,
            bits,
            element,
            peak,
            fields,
            half_frame,
            previous: None,
            current: None,
        };

        Ok(Box::new(deinterlacer))
    }
}

/// The fields each input frame's outputs keep, in the order they are shown.
fn fields_of(order: Order, rate: Rate) -> Vec<BwdifField> {
    let first_rebuilt = match order {
        Order::Top => 1,
        Order::Bottom => 0,
    };
    let first = BwdifField {
        rebuilt: first_rebuilt,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 1 - first_rebuilt,
        first: false,
    };

    match rate {
        Rate::Single => vec![first],
        Rate::Double => vec![first, second],
    }
}

/// Runs `pass` on `current` into `outputs`, keeping for each of `fields` the rows it keeps.
fn run_fields(
    pass: &impl FieldPass,
    context: &FrameContext,
    current: DevicePlane<'_>,
    fields: &[BwdifField],
    outputs: &[Scratch],
) {
    if let [first, _] = fields {
        // The pass keeping the even rows serves the field rebuilding the odd rows.
        let (even, odd) = if first.rebuilt == 1 {
            (&outputs[0], &outputs[1])
        } else {
            (&outputs[1], &outputs[0])
        };

        pass.run_both(context, current, even.plane(), odd.plane());
        return;
    }

    for (field, output) in fields.iter().zip(outputs) {
        let kept = 1 - field.rebuilt;
        pass.run(context, current, output.plane(), kept);
    }
}

/// The three frames one output reads.
struct Window<'a> {
    previous: &'a Frame,
    current: &'a Frame,
    next: &'a Frame,
}

/// A frame's plane for `channel`.
fn channel_plane(frame: &Frame, channel: Channel) -> Result<&Plane, StageError> {
    match frame.plane(channel) {
        Some(plane) => Ok(plane),
        None => Err(missing_channel(channel)),
    }
}

/// Runs `deinterlace/bwdif` over a chunk, emitting each frame's outputs once the frame after it arrives.
struct Deinterlacer<R: Runtime> {
    client: ComputeClient<R>,
    input: FrameFormat,
    output: FrameFormat,
    /// Whether each plane, in channel order, is deinterlaced.
    selected: Vec<bool>,
    predictor: Predictor,
    bits: u32,
    element: Element,
    peak: f32,
    fields: Vec<BwdifField>,
    /// The time from a first field to its second, `None` when the frame rate is unknown.
    half_frame: Option<Duration>,
    previous: Option<Frame>,
    current: Option<Frame>,
}

impl<R: Runtime> Deinterlacer<R> {
    /// The metadata of each output of a frame carrying `meta`, in the order of `fields`.
    fn output_metas(&self, meta: &FrameMeta) -> Vec<FrameMeta> {
        if self.fields.len() == 1 {
            return vec![meta.clone()];
        }

        let first = FrameMeta {
            index: meta.index * 2,
            ..meta.clone()
        };
        let second_timestamp = match (meta.timestamp, self.half_frame) {
            (Some(timestamp), Some(half_frame)) => Some(timestamp + half_frame),
            _ => None,
        };
        let second = FrameMeta {
            index: meta.index * 2 + 1,
            timestamp: second_timestamp,
            ..meta.clone()
        };

        vec![first, second]
    }

    /// Widens `source` into a new f32 scratch plane.
    fn widened(&self, context: &FrameContext, source: &Plane) -> Scratch {
        let (width, height) = source.dimensions();
        let values = Scratch::new::<f32>(context, width, height);
        let sizing = Sizing(context);

        let source_handle = source.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };
        let values_plane = values.plane();

        widen(&self.client, &sizing, source_plane, values_plane, self.element);

        values
    }

    /// Narrows `values` into a new plane of the input's sample format.
    fn narrowed(
        &self,
        context: &FrameContext,
        values: &Scratch,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let device_values = values.plane();
        let width = device_values.width();
        let height = device_values.height();
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.input
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let output = channel_plane(&target, Channel::Y)?;
        let sizing = Sizing(context);

        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, width, height) };

        narrow(
            &self.client,
            &sizing,
            device_values,
            target_plane,
            self.element,
            self.peak,
        );

        Ok(output.clone())
    }

    /// Runs the predictor's field pass on `current` for each of `fields`, one plane per field in the same order.
    ///
    /// Each pass keeps the rows its field keeps. Two fields run through `run_both`, which EEDI3's persistent
    /// check runs in one launch. The cubic predictor has no field pass and gives no planes.
    fn predictions(
        &self,
        context: &FrameContext,
        plane_index: usize,
        current: DevicePlane<'_>,
        fields: &[BwdifField],
    ) -> Vec<Scratch> {
        if matches!(self.predictor, Predictor::Cubic) {
            return Vec::new();
        }

        let width = current.width();
        let height = current.height();
        let outputs: Vec<Scratch> = fields
            .iter()
            .map(|_| Scratch::new::<f32>(context, width, height))
            .collect();

        match self.predictor {
            Predictor::Cubic => {},
            Predictor::Eedi2(arguments) => {
                let layout = self.input.layout;
                let (across, _) = subsampling_shifts(layout, plane_index);
                let settings = Eedi2Settings::new(&arguments, self.bits, across);
                let pass = Eedi2Pass {
                    client: &self.client,
                    settings,
                };

                run_fields(&pass, context, current, fields, &outputs);
            },
            Predictor::Eedi3(arguments) => {
                let settings = Eedi3Settings::new(&arguments, self.bits);
                let pass = Eedi3FieldPass {
                    client: &self.client,
                    settings,
                    clips: Eedi3DeviceClips::default(),
                };

                run_fields(&pass, context, current, fields, &outputs);
            },
            Predictor::Sangnom { aa } => {
                let aaf = sangnom_aaf(aa, self.bits);
                let max = sangnom_max(self.bits);
                let pass = SangnomPass {
                    client: &self.client,
                    aaf,
                    max,
                };

                run_fields(&pass, context, current, fields, &outputs);
            },
        }

        outputs
    }

    /// Rebuilds plane `plane_index` of the window's current frame once for each of `fields`.
    fn rebuild_plane(
        &self,
        context: &FrameContext,
        window: &Window<'_>,
        plane_index: usize,
        fields: &[BwdifField],
    ) -> Result<Vec<Plane>, StageError> {
        let channels = self.input.channels();
        let channel = channels[plane_index];
        let previous_source = channel_plane(window.previous, channel)?;
        let current_source = channel_plane(window.current, channel)?;
        let next_source = channel_plane(window.next, channel)?;

        let previous = self.widened(context, previous_source);
        let current = self.widened(context, current_source);
        let next = self.widened(context, next_source);
        let (width, height) = current_source.dimensions();
        let sizing = Sizing(context);
        let meta = window.current.meta();
        let predictions = self.predictions(context, plane_index, current.plane(), fields);

        let mut rebuilt = Vec::with_capacity(fields.len());
        for (index, field) in fields.iter().enumerate() {
            let output = Scratch::new::<f32>(context, width, height);
            let spatial = predictions.get(index).map(Scratch::plane);
            let planes = BwdifPlanes {
                previous: previous.plane(),
                current: current.plane(),
                next: next.plane(),
                spatial,
                output: output.plane(),
            };
            bwdif_field(&self.client, &sizing, planes, *field);

            let plane = self.narrowed(context, &output, meta)?;
            rebuilt.push(plane);
        }

        Ok(rebuilt)
    }

    /// Every output of the window's current frame.
    fn deinterlace(&self, context: &FrameContext, window: &Window<'_>) -> Result<Vec<Frame>, StageError> {
        let mut planes: Vec<Vec<Plane>> = vec![Vec::new(); self.fields.len()];

        for (plane_index, channel) in self.input.channels().iter().enumerate() {
            if !self.selected[plane_index] {
                let shared = channel_plane(window.current, *channel)?;
                for output_planes in &mut planes {
                    output_planes.push(shared.clone());
                }

                continue;
            }

            let rebuilt = self.rebuild_plane(context, window, plane_index, &self.fields)?;
            for (output_planes, plane) in planes.iter_mut().zip(rebuilt) {
                output_planes.push(plane);
            }
        }

        let metas = self.output_metas(window.current.meta());
        let mut frames = Vec::with_capacity(metas.len());
        for (output_planes, meta) in planes.into_iter().zip(metas) {
            let built = Frame::from_planes(self.output, output_planes, meta);
            let frame = built.map_err(bad_frame)?;
            frames.push(frame);
        }

        Ok(frames)
    }
}

impl<R: Runtime> StageInstance for Deinterlacer<R> {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.previous = None;
        self.current = None;

        Ok(())
    }
}

impl<R: Runtime> Filter for Deinterlacer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(next) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let Some(current) = self.current.take() else {
            self.current = Some(next);
            return Ok(Vec::new());
        };

        let previous = self.previous.take();
        let window = Window {
            previous: previous.as_ref().unwrap_or(&current),
            current: &current,
            next: &next,
        };
        let outputs = self.deinterlace(context, &window)?;

        self.previous = Some(current);
        self.current = Some(next);

        Ok(outputs)
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let previous = self.previous.take();
        let Some(current) = self.current.take() else {
            return Ok(Vec::new());
        };

        let window = Window {
            previous: previous.as_ref().unwrap_or(&current),
            current: &current,
            next: &current,
        };

        self.deinterlace(context, &window)
    }
}
