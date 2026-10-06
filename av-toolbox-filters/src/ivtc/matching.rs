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
use av_toolbox_cubek::deinterlace::BwdifField;
use av_toolbox_cubek::ivtc::MicBlocks;
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
    FrameLayout,
    MetaValue,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_providers::{SceneRequest, Scenes};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::aa::direction::missing_plane;
use crate::bits::bits_of;
use crate::checks::{accepts_frames, between_i32, between_u32};
use crate::deinterlace::bwdif::{Order, resolved_order};
use crate::deinterlace::rebuild::{FieldRebuilder, Window, channel_plane};
use crate::deinterlace::spatial::{
    Predictor,
    Spatial,
    SpatialStage,
    checked_predictor,
    checked_predictor_planes,
};
use crate::ivtc::checks::checked_block;
use crate::ivtc::decision::{C, Decision};
use crate::ivtc::metrics::{Gathered, MetricGatherer, MetricSettings, WindowPlanes};
use crate::masked::{INPUT, checked_input, missing_input};
use crate::scratch::Scratch;

pub(crate) const STAGE: &str = "ivtc/match";

/// Names `ivtc/match` in a refusal of its fallback's `spatial`.
#[derive(Debug)]
pub(crate) struct MatchSpatial;

impl SpatialStage for MatchSpatial {
    const STAGE: &'static str = STAGE;
    const DEFAULT_MDIS: u32 = 20;
}

/// The fewest rows a plane may have.
const MIN_ROWS: u32 = 4;

/// The narrowest luma VFM's field compare can leave its 8 edge columns out of.
const MIN_COLUMNS: u32 = 16;

/// VFM's names for its raw match numbers.
const MATCH_NAMES: [&str; 5] = ["p", "c", "n", "b", "u"];

/// Which field of each frame is kept and matched against its neighbours.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub(crate) enum MatchingField {
    /// The field shown first.
    #[default]
    Same,
    /// The field shown second.
    Opposite,
}

/// What a frame still combed after matching becomes.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum Fallback {
    /// Rebuilt from its matching field by bwdif.
    Bwdif {
        /// Where each rebuilt row's spatial prediction comes from, bwdif's own cubic when not given.
        #[serde(default)]
        spatial: Spatial<MatchSpatial>,
    },
    /// Emitted as it is.
    None {},
}

impl Default for Fallback {
    fn default() -> Self {
        Self::Bwdif {
            spatial: Spatial::default(),
        }
    }
}

fn default_mode() -> u32 {
    1
}

fn enabled() -> bool {
    true
}

fn default_cthresh() -> i32 {
    9
}

fn default_mi() -> u32 {
    80
}

fn default_block() -> u32 {
    16
}

fn default_exclusion() -> u32 {
    16
}

fn default_scthresh() -> f64 {
    12.0
}

fn default_micmatch() -> u32 {
    1
}

/// Arguments for `ivtc/match`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatchArgs {
    /// The field shown first, the stream's own field order when not given.
    #[serde(default)]
    pub(crate) order: Option<Order>,
    /// The field kept from each frame, the field shown first when not given.
    #[serde(default)]
    pub(crate) field: MatchingField,
    /// VFM's mode, which matches are tried and when, between 0 and 5.
    #[serde(default = "default_mode")]
    pub(crate) mode: u32,
    /// Whether the chroma planes join the field compare.
    #[serde(default = "enabled")]
    pub(crate) mchroma: bool,
    /// How far a sample may differ from its neighbours before it is combed, between -1 and 255 on the
    /// 8-bit scale.
    #[serde(default = "default_cthresh")]
    pub(crate) cthresh: i32,
    /// How many combed samples in one block make a frame combed, between 0 and `blockx × blocky`.
    #[serde(default = "default_mi")]
    pub(crate) mi: u32,
    /// Whether chroma combing counts towards the mic.
    #[serde(default = "enabled")]
    pub(crate) chroma: bool,
    /// The width of a mic's block, a power of 2 between 4 and 512.
    #[serde(default = "default_block")]
    pub(crate) blockx: u32,
    /// The height of a mic's block, a power of 2 between 4 and 512.
    #[serde(default = "default_block")]
    pub(crate) blocky: u32,
    /// The first luma row of the band the field compare leaves out, at most `y1`.
    #[serde(default = "default_exclusion")]
    pub(crate) y0: u32,
    /// The last luma row of the band the field compare leaves out. Nothing is left out when it equals `y0`.
    #[serde(default = "default_exclusion")]
    pub(crate) y1: u32,
    /// The scene change threshold, a percentage between 0 and 100.
    #[serde(default = "default_scthresh")]
    pub(crate) scthresh: f64,
    /// When the mics decide between matches, between 0 and 2.
    #[serde(default = "default_micmatch")]
    pub(crate) micmatch: u32,
    /// What a frame still combed after matching becomes, bwdif with its cubic when not given.
    #[serde(default)]
    pub(crate) fallback: Fallback,
}

fn exclusion_reversed(y0: u32, y1: u32) -> StageError {
    let message = format!("{STAGE} needs y0 to be at most y1, got {y0} and {y1}");

    StageError::msg(message)
}

fn not_a_percentage(scthresh: f64) -> StageError {
    let message = format!("{STAGE} needs a finite scthresh between 0 and 100, got {scthresh}");

    StageError::msg(message)
}

fn unsupported_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} matches grey or YUV frames, not {layout}");

    StageError::msg(message)
}

fn unusable_height(channel: Channel, height: u32) -> StageError {
    let message = format!(
        "{STAGE} needs plane {channel} to have an even height of at least {MIN_ROWS} rows, got {height}"
    );

    StageError::msg(message)
}

fn shorter_than_a_block(height: u32, blocky: u32) -> StageError {
    let message = format!("{STAGE} needs the luma to be at least blocky {blocky} rows tall, got {height}");

    StageError::msg(message)
}

fn too_narrow(width: u32) -> StageError {
    let message = format!("{STAGE} needs the luma to be at least {MIN_COLUMNS} columns wide, got {width}");

    StageError::msg(message)
}

fn unsubsampled_width(width: u32, factor: u32) -> StageError {
    let message =
        format!("{STAGE} needs a luma width divisible by the chroma subsampling {factor}, got {width}");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} matches frames, not masks");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

/// Refuses any argument outside its range.
fn checked_arguments(args: &MatchArgs) -> Result<(), StageError> {
    between_u32(STAGE, "mode", args.mode, 0, 5)?;
    between_u32(STAGE, "micmatch", args.micmatch, 0, 2)?;
    between_i32(STAGE, "cthresh", args.cthresh, -1, 255)?;
    checked_block(STAGE, "blockx", args.blockx)?;
    checked_block(STAGE, "blocky", args.blocky)?;

    let block_samples = args.blockx * args.blocky;
    between_u32(STAGE, "mi", args.mi, 0, block_samples)?;

    if args.y0 > args.y1 {
        return Err(exclusion_reversed(args.y0, args.y1));
    }

    let percentage = (0.0..=100.0).contains(&args.scthresh);
    if !percentage {
        return Err(not_a_percentage(args.scthresh));
    }

    Ok(())
}

/// Refuses a layout or plane size VFM cannot run on.
fn checked_planes(args: &MatchArgs, input: &FrameFormat) -> Result<(), StageError> {
    match input.layout {
        FrameLayout::Grey => {},
        FrameLayout::Yuv { subsampling } => {
            let (across, _) = subsampling.factors();
            if !input.width.is_multiple_of(across) {
                return Err(unsubsampled_width(input.width, across));
            }
        },
        layout => return Err(unsupported_layout(layout)),
    }

    for channel in input.channels() {
        let Some((_, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(STAGE, *channel));
        };

        if height < MIN_ROWS || !height.is_multiple_of(2) {
            return Err(unusable_height(*channel, height));
        }
    }

    if input.height < args.blocky {
        return Err(shorter_than_a_block(input.height, args.blocky));
    }

    if input.width < MIN_COLUMNS {
        return Err(too_narrow(input.width));
    }

    Ok(())
}

/// The fallback's prediction once its settings are checked against `input`, `None` without a fallback.
fn checked_fallback(fallback: &Fallback, input: &FrameFormat) -> Result<Option<Predictor>, StageError> {
    let Fallback::Bwdif { spatial } = fallback else {
        return Ok(None);
    };

    let predictor = checked_predictor(STAGE, spatial)?;
    let every_plane = vec![true; input.channels().len()];
    checked_predictor_planes(STAGE, predictor, input, &every_plane)?;

    Ok(Some(predictor))
}

/// `args` once checked against `input`.
struct Checked {
    order: Order,
    fallback: Option<Predictor>,
}

fn checked(args: &MatchArgs, input: &FrameFormat) -> Result<Checked, StageError> {
    checked_arguments(args)?;
    let order = resolved_order(STAGE, args.order, input)?;
    checked_planes(args, input)?;
    let fallback = checked_fallback(&args.fallback, input)?;

    Ok(Checked { order, fallback })
}

fn output_format(input: &FrameFormat) -> FrameFormat {
    FrameFormat {
        field_order: Some(FieldOrder::Progressive),
        ..*input
    }
}

/// VFM's field, 1 when the matching field is the top field.
fn vfm_field(order: Order, field: MatchingField) -> u32 {
    let first = match order {
        Order::Top => 1,
        Order::Bottom => 0,
    };

    match field {
        MatchingField::Same => first,
        MatchingField::Opposite => 1 - first,
    }
}

/// Rebuilds telecined film frames by field matching, the way VIVTC's VFM does.
pub(crate) struct Match;

impl StageDefinition for Match {
    type Args = MatchArgs;

    fn package() -> &'static str {
        "ivtc"
    }

    fn identifier() -> &'static str {
        "match"
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Deinterlace]
    }

    // Its window spans a few frames, so any break costs little and chunks of 10 s or 240 frames, whichever is
    // shorter, spread across workers.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        let demand = AnyDemand::new::<Scenes>(SceneRequest::limited_to(240, 10.0));

        Parallelism::Chunk(demand)
    }

    // The current frame, whose output waits on the next, and the previous frame it reads.
    fn held_frames(_args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        2
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Match {
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
        checked(args, &input)?;

        let output = output_format(&input);
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
        let checked = checked(&args, &input)?;
        let output = output_format(&input);

        let bits = bits_of(input.sample);
        let shift = bits - 8;
        let field = vfm_field(checked.order, args.field);
        let opposite = args.field == MatchingField::Opposite;
        let decision = Decision {
            mode: args.mode,
            micmatch: args.micmatch,
            mi: args.mi,
            shift,
            opposite,
        };

        let cthresh = if args.cthresh >= 0 {
            args.cthresh << shift
        } else {
            args.cthresh
        };
        let settings = MetricSettings {
            field,
            pairs: decision.pairs(),
            mode: args.mode,
            micmatch: args.micmatch,
            cthresh,
            chroma: args.chroma,
            mchroma: args.mchroma,
            blocks: MicBlocks {
                x: args.blockx,
                y: args.blocky,
            },
            y0: args.y0,
            y1: args.y1,
            scthresh: args.scthresh / 100.0,
        };

        let client = context.client().clone();
        let predictor = checked.fallback.unwrap_or(Predictor::Cubic);
        let rebuilder = FieldRebuilder::new(STAGE, client, input, predictor);
        let gatherer_client = rebuilder.client().clone();
        let gatherer = MetricGatherer::new(gatherer_client, input, settings, bits);
        let kept = BwdifField {
            rebuilt: field,
            first: !opposite,
        };

        let matcher = Matcher {
            rebuilder,
            gatherer,
            decision,
            falls_back: checked.fallback.is_some(),
            kept,
            input,
            output,
            previous: None,
            current: None,
        };

        Ok(Box::new(matcher))
    }
}

/// Runs `ivtc/match` over a chunk, emitting each frame once the frame after it arrives.
struct Matcher<R: Runtime> {
    rebuilder: FieldRebuilder<R>,
    gatherer: MetricGatherer<R>,
    decision: Decision,
    /// Whether a combed frame is rebuilt by bwdif rather than emitted as it is.
    falls_back: bool,
    /// The bwdif field pass that rebuilds a combed frame from its matching field.
    kept: BwdifField,
    input: FrameFormat,
    output: FrameFormat,
    previous: Option<Frame>,
    current: Option<Frame>,
}

impl<R: Runtime> Matcher<R> {
    /// Every plane of `frame`, widened to f32.
    fn widened_planes(&self, context: &FrameContext, frame: &Frame) -> Result<Vec<Scratch>, StageError> {
        let mut planes = Vec::new();
        for channel in self.input.channels() {
            let source = channel_plane(STAGE, frame, *channel)?;
            let widened = self.rebuilder.widened(context, source);
            planes.push(widened);
        }

        Ok(planes)
    }

    fn window_planes(&self, context: &FrameContext, window: &Window<'_>) -> Result<WindowPlanes, StageError> {
        let previous = self.widened_planes(context, window.previous)?;
        let current = self.widened_planes(context, window.current)?;
        let next = self.widened_planes(context, window.next)?;
        let next_is_current = std::ptr::eq(window.next, window.current);

        Ok(WindowPlanes {
            previous,
            current,
            next,
            next_is_current,
        })
    }

    /// The planes of the chosen weave, the current frame's own for a `c` match.
    fn woven_planes(
        &self,
        context: &FrameContext,
        window: &Window<'_>,
        gathered: &Gathered,
        matched: usize,
    ) -> Result<Vec<Plane>, StageError> {
        let channels = self.input.channels();
        if matched == C {
            let mut planes = Vec::with_capacity(channels.len());
            for channel in channels {
                let plane = channel_plane(STAGE, window.current, *channel)?;
                planes.push(plane.clone());
            }

            return Ok(planes);
        }

        let meta = window.current.meta();
        let mut planes = Vec::with_capacity(channels.len());
        for woven in &gathered.woven[matched] {
            let plane = self.rebuilder.narrowed(context, woven, meta)?;
            planes.push(plane);
        }

        Ok(planes)
    }

    /// Every plane of the window's current frame, rebuilt by bwdif from its matching field.
    fn rebuilt_planes(&self, context: &FrameContext, window: &Window<'_>) -> Result<Vec<Plane>, StageError> {
        let fields = [self.kept];

        let mut planes = Vec::new();
        for plane_index in 0..self.input.channels().len() {
            let rebuilt = self
                .rebuilder
                .rebuild_plane(context, window, plane_index, &fields)?;
            planes.extend(rebuilt);
        }

        Ok(planes)
    }

    /// The output of the window's current frame.
    fn matched(&self, context: &FrameContext, window: &Window<'_>) -> Result<Frame, StageError> {
        let window_planes = self.window_planes(context, window)?;
        let gathered = self.gatherer.gather(context, &window_planes)?;
        let metrics = gathered.metrics;
        let matched = self.decision.choose(&metrics);
        let mic = metrics.mics[matched];
        let combed = mic >= self.decision.mi;

        let planes = if combed && self.falls_back {
            self.rebuilt_planes(context, window)?
        } else {
            self.woven_planes(context, window, &gathered, matched)?
        };

        let mut meta = window.current.meta().clone();
        let name = MATCH_NAMES[matched].to_owned();
        meta.attach("match", MetaValue::String(name));
        meta.attach("combed", MetaValue::Bool(combed));
        let mic_value = i64::from(mic);
        meta.attach("mic", MetaValue::Int(mic_value));
        meta.attach("scene_change", MetaValue::Bool(metrics.scene_change));

        let built = Frame::from_planes(self.output, planes, meta);

        built.map_err(bad_frame)
    }
}

impl<R: Runtime> StageInstance for Matcher<R> {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.previous = None;
        self.current = None;

        Ok(())
    }
}

impl<R: Runtime> Filter for Matcher<R> {
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
        let output = self.matched(context, &window)?;

        self.previous = Some(current);
        self.current = Some(next);

        Ok(vec![output])
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
        let output = self.matched(context, &window)?;

        Ok(vec![output])
    }
}
