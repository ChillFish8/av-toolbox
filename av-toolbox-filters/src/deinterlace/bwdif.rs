use std::time::Duration;

use av_toolbox_core::{
    AnyDemand,
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
use av_toolbox_cubek::deinterlace::{BWDIF_MIN_ROWS, BwdifField};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    DeviceSide,
    FieldOrder,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameMeta,
    FrameRate,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_transforms::{SceneRequest, Scenes};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::aa::direction::{Direction, checked_dimensions, missing_plane};
use crate::checks::accepts_frames;
use crate::deinterlace::rebuild::{FieldRebuilder, Window, channel_plane};
use crate::deinterlace::spatial::{Spatial, SpatialStage, checked_predictor, checked_predictor_planes};
use crate::masked::{INPUT, checked_input, missing_input};
use crate::planes::ChannelName;
use crate::selection::{PlaneDefault, selected_planes};

pub(crate) const STAGE: &str = "deinterlace/bwdif";

/// Names `deinterlace/bwdif` in a refusal of its `spatial`.
#[derive(Debug)]
pub(crate) struct BwdifSpatial;

impl SpatialStage for BwdifSpatial {
    const STAGE: &'static str = STAGE;
    const DEFAULT_MDIS: u32 = 20;
}

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
    /// One frame for each frame in, or one for each field. One for each field when not given.
    #[serde(default)]
    pub(crate) rate: Rate,
    /// The planes to deinterlace, every plane when not given.
    #[serde(default)]
    pub(crate) planes: Option<Vec<ChannelName>>,
    /// Where each rebuilt row's spatial prediction comes from, bwdif's own cubic when not given.
    #[serde(default)]
    pub(crate) spatial: Spatial<BwdifSpatial>,
}

fn describe_order(found: Option<FieldOrder>) -> String {
    match found {
        Some(order) => order.to_string(),
        None => "unknown".to_owned(),
    }
}

fn unusable_order(stage: &str, found: Option<FieldOrder>) -> StageError {
    let described = describe_order(found);
    let message =
        format!("{stage} needs `order` set to top or bottom, since the stream's field order is {described}");

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

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// The field shown first, `order` when given and the stream's own interlaced order otherwise.
pub(crate) fn resolved_order(
    stage: &str,
    order: Option<Order>,
    input: &FrameFormat,
) -> Result<Order, StageError> {
    if let Some(order) = order {
        return Ok(order);
    }

    match input.field_order {
        Some(FieldOrder::TopFirst) => Ok(Order::Top),
        Some(FieldOrder::BottomFirst) => Ok(Order::Bottom),
        found => Err(unusable_order(stage, found)),
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
pub(crate) fn output_format(input: &FrameFormat, rate: Rate) -> Result<FrameFormat, StageError> {
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
pub(crate) fn half_frame(rate: Option<FrameRate>) -> Option<Duration> {
    let rate = rate?;
    let denominator = u128::from(rate.denominator());
    let numerator = u128::from(rate.numerator());
    let nanos = denominator * 1_000_000_000 / (2 * numerator);
    let nanos = u64::try_from(nanos).ok()?;

    Some(Duration::from_nanos(nanos))
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

    // Its window spans a few frames, so any break costs little and chunks of 10 s spread across workers.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        let demand = AnyDemand::new::<Scenes>(SceneRequest::limited_to_seconds(10.0));

        Parallelism::Chunk(demand)
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
        accepts_frames(offered).ok_or_else(not_frames)
    }

    fn produces(
        _capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(STAGE, chosen)?;
        resolved_order(STAGE, args.order, &input)?;
        let selected = checked_planes(args, &input)?;
        let predictor = checked_predictor(STAGE, &args.spatial)?;
        checked_predictor_planes(STAGE, predictor, &input, &selected)?;

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
        let order = resolved_order(STAGE, args.order, &input)?;
        let selected = checked_planes(&args, &input)?;
        let predictor = checked_predictor(STAGE, &args.spatial)?;
        checked_predictor_planes(STAGE, predictor, &input, &selected)?;
        let output = output_format(&input, args.rate)?;

        let client = context.client().clone();
        let rebuilder = FieldRebuilder::new(STAGE, client, input, predictor);
        let fields = fields_of(order, args.rate);
        let half_frame = half_frame(input.frame_rate);

        let deinterlacer = Deinterlacer {
            rebuilder,
            input,
            output,
            selected,
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

/// The metadata of each output of a frame carrying `meta`, one when `doubled` is false and otherwise one for
/// each field.
pub(crate) fn output_metas(meta: &FrameMeta, half_frame: Option<Duration>, doubled: bool) -> Vec<FrameMeta> {
    if !doubled {
        return vec![meta.clone()];
    }

    let first = FrameMeta {
        index: meta.index * 2,
        ..meta.clone()
    };
    let second_timestamp = match (meta.timestamp, half_frame) {
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

/// Runs `deinterlace/bwdif` over a chunk, emitting each frame's outputs once the frame after it arrives.
struct Deinterlacer<R: Runtime> {
    rebuilder: FieldRebuilder<R>,
    input: FrameFormat,
    output: FrameFormat,
    /// Whether each plane, in channel order, is deinterlaced.
    selected: Vec<bool>,
    fields: Vec<BwdifField>,
    /// The time from a first field to its second, `None` when the frame rate is unknown.
    half_frame: Option<Duration>,
    previous: Option<Frame>,
    current: Option<Frame>,
}

impl<R: Runtime> Deinterlacer<R> {
    /// Every output of the window's current frame.
    fn deinterlace(&self, context: &FrameContext, window: &Window<'_>) -> Result<Vec<Frame>, StageError> {
        let mut planes: Vec<Vec<Plane>> = vec![Vec::new(); self.fields.len()];

        for (plane_index, channel) in self.input.channels().iter().enumerate() {
            if !self.selected[plane_index] {
                let shared = channel_plane(STAGE, window.current, *channel)?;
                for output_planes in &mut planes {
                    output_planes.push(shared.clone());
                }

                continue;
            }

            let rebuilt = self
                .rebuilder
                .rebuild_plane(context, window, plane_index, &self.fields)?;
            for (output_planes, plane) in planes.iter_mut().zip(rebuilt) {
                output_planes.push(plane);
            }
        }

        let metas = output_metas(window.current.meta(), self.half_frame, self.fields.len() == 2);
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
