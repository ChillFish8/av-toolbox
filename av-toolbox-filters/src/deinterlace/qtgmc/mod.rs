mod args;
mod cache;
mod chain;
mod front;
#[cfg(feature = "test-support")]
mod testing;

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
use av_toolbox_cubek::deinterlace::qtgmc::{Parity, QtgmcTables, sharpness_strength};
use av_toolbox_cubek::motion::{BlockGrid, PROGRESSIVE_SAD_MASK, PlaneScale, SadMaskTables};
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
use av_toolbox_providers::{SceneRequest, Scenes};

pub(crate) use self::args::{Input, QtgmcArgs};
use self::args::{checked_field_size, degrain_settings, mask_scene_change, motion_settings, scene_change};
use self::cache::FieldCache;
use self::chain::{Chain, ChainSettings, PlaneShape};
use self::front::kept_parity;
#[cfg(feature = "test-support")]
pub use self::testing::{QtgmcMotion, QtgmcMotionLog, QtgmcRepairFlag, qtgmc_with_motion_log};
use crate::aa::direction::{Direction, checked_dimensions, missing_plane};
use crate::bits::bits_of;
use crate::checks::accepts_frames;
use crate::deinterlace::bwdif::{Order, Rate, half_frame, output_format, output_metas, resolved_order};
use crate::deinterlace::rebuild::FieldRebuilder;
use crate::deinterlace::spatial::{Predictor, checked_predictor, checked_predictor_planes};
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

/// Refuses a format that is not integer YUV or Gray.
fn checked_format(input: &FrameFormat) -> Result<(), StageError> {
    let is_integer = matches!(input.sample, SampleFormat::U8 | SampleFormat::U16 { .. });
    let is_layout = matches!(input.layout, FrameLayout::Yuv { .. } | FrameLayout::Grey);
    if !is_integer || !is_layout {
        return Err(unsupported_format(input));
    }

    Ok(())
}

/// The field order `input` is processed in, which progressive input does not need from the stream.
fn stage_order(args: &QtgmcArgs, input: &FrameFormat) -> Result<Order, StageError> {
    if args.input == Input::Progressive {
        return Ok(args.order.unwrap_or(Order::Top));
    }

    resolved_order(STAGE, args.order, input)
}

/// Refuses `args` that `input` cannot run with.
fn checked_stage(args: &QtgmcArgs, input: &FrameFormat) -> Result<Predictor, StageError> {
    args.checked_values()?;
    stage_order(args, input)?;
    checked_format(input)?;

    let selected = vec![true; input.channels().len()];
    if args.input != Input::Progressive {
        checked_dimensions(STAGE, input, &selected, Direction::Vertical)?;
    }

    checked_field_size(input, args.input)?;

    let predictor = checked_predictor(STAGE, &args.spatial)?;
    match args.input {
        Input::Progressive => {},
        Input::Interlaced => checked_predictor_planes(STAGE, predictor, input, &selected)?,
        Input::Repair | Input::RepairReversed => {
            let luma_only: Vec<bool> = (0..selected.len()).map(|plane| plane == 0).collect();
            checked_predictor_planes(STAGE, predictor, input, &luma_only)?;
        },
    }

    Ok(predictor)
}

/// The shape of every plane of `input`, in channel order.
fn plane_shapes(input: &FrameFormat) -> Result<Vec<PlaneShape>, StageError> {
    let luma_width = input.width;
    let luma_height = input.height;

    let mut shapes = Vec::with_capacity(input.channels().len());
    for channel in input.channels() {
        let Some((width, height)) = input.channel_dimensions(*channel) else {
            return Err(missing_plane(STAGE, *channel));
        };

        let scale = PlaneScale {
            shift_x: u32::from(width < luma_width),
            shift_y: u32::from(height < luma_height),
        };
        shapes.push(PlaneShape { width, height, scale });
    }

    Ok(shapes)
}

/// The settings of every step, from `args`.
fn chain_settings(args: &QtgmcArgs) -> ChainSettings {
    let strength = sharpness_strength(args.sharpness, args.tr1, args.tr2);

    ChainSettings {
        tr0: args.tr0,
        tr1: args.tr1,
        tr2: args.tr2,
        max_tr: args.max_tr(),
        strength: (args.sharpness > 0.0).then_some(strength),
        motion: motion_settings(),
        scene_change: scene_change(),
        mask_scene_change: mask_scene_change(),
        tr1_degrain: degrain_settings(1, args.thsad1),
        tr2_degrain: degrain_settings(args.tr2.max(1), args.thsad2),
    }
}

/// The deinterlacer `args` describe for `input`, with its tables built.
fn deinterlacer<R: Runtime>(
    context: &BuildContext<R>,
    args: &QtgmcArgs,
    input: FrameFormat,
) -> Result<Deinterlacer<R>, StageError> {
    let predictor = checked_stage(args, &input)?;
    let order = stage_order(args, &input)?;
    let output = output_format(&input, args.resolved_rate())?;
    let shapes = plane_shapes(&input)?;

    let client = context.client();
    let sizes: Vec<(u32, u32)> = shapes.iter().map(|shape| (shape.width, shape.height)).collect();
    let luma = (input.width, input.height);
    let tables = match args.input {
        Input::Progressive => QtgmcTables::without_bobs(client, luma),
        _ => QtgmcTables::new(client, &sizes),
    };
    let settings = chain_settings(args);
    let grid = BlockGrid::new(&settings.motion, input.width, input.height);
    let mask_tables = args
        .input
        .is_repair()
        .then(|| SadMaskTables::new(client, &grid, PROGRESSIVE_SAD_MASK, luma));
    let rebuilder = FieldRebuilder::new(STAGE, client.clone(), input, predictor);

    let chain = Chain {
        rebuilder,
        predictor,
        tables,
        mask_tables,
        mode: args.input,
        input,
        output,
        shapes,
        grid,
        settings,
        bits: bits_of(input.sample),
    };
    let first = match order {
        Order::Top => Parity::Top,
        Order::Bottom => Parity::Bottom,
    };

    Ok(Deinterlacer {
        chain,
        cache: FieldCache::new(),
        parities: [first, first.other()],
        doubled: args.resolved_rate() == Rate::Double,
        half_frame: half_frame(input.frame_rate),
        #[cfg(feature = "test-support")]
        log: None,
    })
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

    fn description() -> &'static str {
        "Deinterlaces with motion-compensated smoothing, as QTGMC does.\n\n\
         It is slower than `deinterlace/bwdif` but steadier, with less shimmer. It can also clean up \
         progressive frames or repair the output of another deinterlacer."
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Deinterlace]
    }

    // It restarts only at real cuts, and whole scenes still spread across workers.
    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::chunk::<Scenes>(SceneRequest::natural())
    }

    // The entries the output trails the input by, and the frames they and the current frame come from.
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

        let output = output_format(&input, args.resolved_rate())?;
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
        let deinterlacer = deinterlacer(context, &args, input)?;

        Ok(Box::new(deinterlacer))
    }
}

/// Runs `deinterlace/qtgmc` over a chunk, emitting each field once every step its output reads has run.
struct Deinterlacer<R: Runtime> {
    chain: Chain<R>,
    cache: FieldCache,
    /// The parity of each field of a frame, in the order they are shown.
    parities: [Parity; 2],
    /// Whether both fields of each frame are shown, rather than only the first.
    doubled: bool,
    /// The time from a first field to its second, `None` when the frame rate is unknown.
    half_frame: Option<Duration>,
    /// Where each motion search is recorded, for tests.
    #[cfg(feature = "test-support")]
    log: Option<QtgmcMotionLog>,
}

impl<R: Runtime> Deinterlacer<R> {
    /// Runs every step that has become ready, in the chain's order, and takes the outputs now due.
    fn advance(&mut self, context: &FrameContext) -> Result<Vec<Frame>, StageError> {
        for field in self.cache.held() {
            self.chain.search_clip(context, &mut self.cache, field)?;
        }

        for field in self.cache.held() {
            let repaired = self.chain.repair(context, &mut self.cache, field)?;
            if repaired {
                self.record_repair(field)?;
            }
        }

        for field in self.cache.held() {
            let found = self.chain.find_motion(context, &mut self.cache, field)?;
            if found {
                self.record_motion(field)?;
            }
        }

        for field in self.cache.held() {
            self.chain.limit(context, &mut self.cache, field)?;
        }

        for field in self.cache.held() {
            self.chain.finish(context, &mut self.cache, field)?;
        }

        let outputs = self.cache.take_outputs();
        let reach = self.chain.settings.reach();
        self.cache.release(reach);

        Ok(outputs)
    }

    /// Records the motion of `field` in the test log, when there is one.
    #[cfg(feature = "test-support")]
    fn record_motion(&self, field: usize) -> Result<(), StageError> {
        let Some(log) = &self.log else {
            return Ok(());
        };

        log.record(&self.chain, &self.cache, field)
    }

    #[cfg(not(feature = "test-support"))]
    fn record_motion(&self, _field: usize) -> Result<(), StageError> {
        Ok(())
    }

    /// Records the repair mask's flag of `field` in the test log, when there is one.
    #[cfg(feature = "test-support")]
    fn record_repair(&self, field: usize) -> Result<(), StageError> {
        let Some(log) = &self.log else {
            return Ok(());
        };

        log.record_repair(&self.chain, &self.cache, field)
    }

    #[cfg(not(feature = "test-support"))]
    fn record_repair(&self, _field: usize) -> Result<(), StageError> {
        Ok(())
    }

    /// Adds one progressive frame to the cache as its own entry.
    fn push_frame(&mut self, context: &FrameContext, frame: &Frame) -> Result<(), StageError> {
        let meta = frame.meta().clone();
        let reversed = self.chain.mode == Input::RepairReversed;
        let kept = kept_parity(self.parities[0], reversed, meta.index);

        let state = self.chain.entry_of(context, frame, kept, meta)?;
        self.cache.push(state);

        Ok(())
    }

    /// Adds both fields of one interlaced frame to the cache.
    fn push_fields(&mut self, context: &FrameContext, frame: &Frame) -> Result<(), StageError> {
        let metas = output_metas(frame.meta(), self.half_frame, self.doubled);
        let mut shown = metas.into_iter();
        let field_metas = [shown.next(), shown.next()];

        let fields = self.chain.fields_of(context, frame, self.parities, field_metas)?;
        for state in fields {
            self.cache.push(state);
        }

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Deinterlacer<R> {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.cache.restart();

        Ok(())
    }
}

impl<R: Runtime> Filter for Deinterlacer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(next) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        match self.chain.mode.is_progressive() {
            true => self.push_frame(context, &next)?,
            false => self.push_fields(context, &next)?,
        }

        self.advance(context)
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        self.cache.end();
        let outputs = self.advance(context)?;
        self.cache.restart();

        Ok(outputs)
    }
}
