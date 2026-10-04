use std::mem;
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
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::ivtc::{
    BlockPlanes,
    DecimateMetric,
    HalfBlocks,
    half_block_differences,
    half_block_grid,
};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameRate,
    Offered,
    OutputFormats,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use cubecl::prelude::CubeElement;
use cubecl::server::ServerError;
use schemars::JsonSchema;
use serde::Deserialize;

use crate::aa::eedi2::subsampling_shifts;
use crate::bits::bits_of;
use crate::checks::between_u32;
use crate::deinterlace::rebuild::{channel_plane, widened_plane};
use crate::ivtc::checks::checked_block;
use crate::ivtc::decimation::{Thresholds, find_drop, first_cycle_metrics};
use crate::masked::{INPUT, checked_input, missing_input};
use crate::scratch::Scratch;
use crate::sizing::Sizing;

pub(crate) const STAGE: &str = "ivtc/decimate";

const NANOS_PER_SECOND: u128 = 1_000_000_000;

fn default_cycle() -> u32 {
    5
}

fn default_dupthresh() -> f64 {
    1.1
}

fn default_scthresh() -> f64 {
    15.0
}

fn default_block() -> u32 {
    32
}

/// Arguments for `ivtc/decimate`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct DecimateArgs {
    /// How many frames make up each cycle that one frame is dropped from, between 2 and 25.
    #[serde(default = "default_cycle")]
    pub(crate) cycle: u32,
    /// Whether the chroma planes count towards the differences, set when not given if the stream has chroma.
    #[serde(default)]
    pub(crate) chroma: Option<bool>,
    /// The duplicate threshold, a percentage of one block between 0 and 100.
    #[serde(default = "default_dupthresh")]
    pub(crate) dupthresh: f64,
    /// The scene change threshold, a percentage of the luma plane between 0 and 100.
    #[serde(default = "default_scthresh")]
    pub(crate) scthresh: f64,
    /// The width of a block, a power of 2 between 4 and 512.
    #[serde(default = "default_block")]
    pub(crate) blockx: u32,
    /// The height of a block, a power of 2 between 4 and 512.
    #[serde(default = "default_block")]
    pub(crate) blocky: u32,
}

fn not_a_percentage(name: &str, value: f64) -> StageError {
    let message = format!("{STAGE} needs a finite {name} between 0 and 100, got {value}");

    StageError::msg(message)
}

fn chroma_on_grey() -> StageError {
    let message = format!("{STAGE} cannot count chroma on a grey stream, got chroma true");

    StageError::msg(message)
}

fn unsupported_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} decimates grey or YUV frames, not {layout}");

    StageError::msg(message)
}

fn within_half_a_block(name: &str, size: u32, axis: &str, block: u32) -> StageError {
    let message = format!("{STAGE} needs the luma to be more than half of {name} {block} {axis}, got {size}");

    StageError::msg(message)
}

fn unrepresentable_rate(rate: FrameRate, cycle: u32) -> StageError {
    let message = format!("{STAGE} cannot represent {rate} with one frame in {cycle} dropped");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} decimates frames, not masks");

    StageError::msg(message)
}

fn bad_frame(error: FrameError) -> StageError {
    let message = format!("{STAGE} built a bad frame: {error}");

    StageError::msg(message)
}

fn unreadable_cells(error: ServerError) -> StageError {
    let message = format!("{STAGE} could not read back its block differences");

    StageError::with_source(message, error)
}

fn checked_percentage(name: &str, value: f64) -> Result<(), StageError> {
    if (0.0..=100.0).contains(&value) {
        return Ok(());
    }

    Err(not_a_percentage(name, value))
}

/// Refuses any argument outside its range.
fn checked_arguments(args: &DecimateArgs) -> Result<(), StageError> {
    between_u32(STAGE, "cycle", args.cycle, 2, 25)?;
    checked_percentage("dupthresh", args.dupthresh)?;
    checked_percentage("scthresh", args.scthresh)?;
    checked_block(STAGE, "blockx", args.blockx)?;
    checked_block(STAGE, "blocky", args.blocky)?;

    Ok(())
}

/// Whether the chroma planes are counted, refusing chroma on a grey stream.
fn resolved_chroma(args: &DecimateArgs, input: &FrameFormat) -> Result<bool, StageError> {
    let has_chroma = match input.layout {
        FrameLayout::Grey => false,
        FrameLayout::Yuv { .. } => true,
        layout => return Err(unsupported_layout(layout)),
    };

    match args.chroma {
        Some(true) if !has_chroma => Err(chroma_on_grey()),
        Some(chroma) => Ok(chroma),
        None => Ok(has_chroma),
    }
}

/// Refuses a luma at most half a block across or down, which has no window of four half blocks.
fn checked_luma(args: &DecimateArgs, input: &FrameFormat) -> Result<(), StageError> {
    if input.width <= args.blockx / 2 {
        return Err(within_half_a_block("blockx", input.width, "across", args.blockx));
    }

    if input.height <= args.blocky / 2 {
        return Err(within_half_a_block("blocky", input.height, "down", args.blocky));
    }

    Ok(())
}

fn greatest_common_divisor(mut first: u64, mut second: u64) -> u64 {
    while second != 0 {
        let remainder = first % second;
        first = second;
        second = remainder;
    }

    first
}

/// `rate` with one frame in every `cycle` dropped, unknown when `rate` is.
fn output_rate(rate: Option<FrameRate>, cycle: u32) -> Result<Option<FrameRate>, StageError> {
    let Some(rate) = rate else {
        return Ok(None);
    };

    let numerator = u64::from(rate.numerator()) * u64::from(cycle - 1);
    let denominator = u64::from(rate.denominator()) * u64::from(cycle);
    let divisor = greatest_common_divisor(numerator, denominator);
    let reduced_numerator = u32::try_from(numerator / divisor);
    let reduced_denominator = u32::try_from(denominator / divisor);

    match (reduced_numerator, reduced_denominator) {
        (Ok(numerator), Ok(denominator)) => Ok(FrameRate::new(numerator, denominator)),
        _ => Err(unrepresentable_rate(rate, cycle)),
    }
}

/// `args` once checked against `input`.
struct Checked {
    chroma: bool,
    output: FrameFormat,
}

fn checked(args: &DecimateArgs, input: &FrameFormat) -> Result<Checked, StageError> {
    checked_arguments(args)?;
    let chroma = resolved_chroma(args, input)?;
    checked_luma(args, input)?;
    let frame_rate = output_rate(input.frame_rate, args.cycle)?;

    let output = FrameFormat { frame_rate, ..*input };

    Ok(Checked { chroma, output })
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

/// The half block of each counted plane, VDecimate's half block scaled down by a chroma plane's subsampling.
fn half_blocks(args: &DecimateArgs, input: &FrameFormat, planes: usize) -> Vec<HalfBlocks> {
    let luma = HalfBlocks {
        x: args.blockx / 2,
        y: args.blocky / 2,
    };

    (0..planes)
        .map(|plane_index| {
            let (shift_x, shift_y) = subsampling_shifts(input.layout, plane_index);
            HalfBlocks {
                x: luma.x >> shift_x,
                y: luma.y >> shift_y,
            }
        })
        .collect()
}

/// The time from the first output frame to the `emitted`th at `rate`.
fn output_offset(rate: FrameRate, emitted: u64) -> Duration {
    let numerator = u128::from(rate.numerator());
    let denominator = u128::from(rate.denominator());
    let nanos = u128::from(emitted) * denominator * NANOS_PER_SECOND / numerator;
    let seconds = (nanos / NANOS_PER_SECOND) as u64;
    let subsecond = (nanos % NANOS_PER_SECOND) as u32;

    Duration::new(seconds, subsecond)
}

/// Drops one frame in every cycle, the one most likely to be a duplicate, the way VIVTC's VDecimate does.
pub(crate) struct Decimate;

impl StageDefinition for Decimate {
    type Args = DecimateArgs;

    fn package() -> &'static str {
        "ivtc"
    }

    fn identifier() -> &'static str {
        "decimate"
    }

    fn tags() -> &'static [Tag] {
        &[Tag::Deinterlace]
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Stream
    }

    // The frames of one cycle and the frame before it, which the cycle's first frame is measured against.
    fn held_frames(args: &Self::Args, _formats: &ResolvedFormats) -> usize {
        args.cycle as usize + 1
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Decimate {
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
        let checked = checked(args, &input)?;
        let format = StreamFormat::Frames(checked.output);

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

        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let planes = if checked.chroma { input.channels().len() } else { 1 };
        let halves = half_blocks(&args, &input, planes);
        let (across, down) = half_block_grid(input.width, input.height, halves[0]);
        let thresholds = Thresholds::new(
            args.dupthresh,
            args.scthresh,
            bits,
            input.width,
            input.height,
            args.blockx,
            args.blocky,
        );

        let decimator = Decimator {
            client: context.client().clone(),
            input,
            output: checked.output,
            element,
            halves,
            across,
            down,
            thresholds,
            cycle: args.cycle as usize,
            previous: None,
            frames: Vec::new(),
            metrics: Vec::new(),
            cycle_index: 0,
            emitted: 0,
            start: None,
        };

        Ok(Box::new(decimator))
    }
}

/// The index and timestamp of a stream's first frame, which every output's are counted from.
#[derive(Debug, Clone, Copy)]
struct StreamStart {
    index: u64,
    timestamp: Option<Duration>,
}

/// Runs `ivtc/decimate` over a stream, emitting each cycle once all of its frames are in.
struct Decimator<R: Runtime> {
    client: ComputeClient<R>,
    input: FrameFormat,
    output: FrameFormat,
    element: Element,
    /// The half block of each counted plane, luma first.
    halves: Vec<HalfBlocks>,
    across: u32,
    down: u32,
    thresholds: Thresholds,
    cycle: usize,
    /// The previous frame's counted planes, widened to f32.
    previous: Option<Vec<Scratch>>,
    frames: Vec<Frame>,
    metrics: Vec<DecimateMetric>,
    cycle_index: u64,
    emitted: u64,
    start: Option<StreamStart>,
}

impl<R: Runtime> Decimator<R> {
    /// Every counted plane of `frame`, widened to f32.
    fn widened_planes(&self, context: &FrameContext, frame: &Frame) -> Result<Vec<Scratch>, StageError> {
        let counted = &self.input.channels()[..self.halves.len()];

        let mut planes = Vec::with_capacity(counted.len());
        for channel in counted {
            let source = channel_plane(STAGE, frame, *channel)?;
            let widened = widened_plane(&self.client, context, source, self.element);
            planes.push(widened);
        }

        Ok(planes)
    }

    /// VDecimate's measures of `current` against `previous`, from every counted plane's half blocks.
    fn metric(
        &self,
        context: &FrameContext,
        previous: &[Scratch],
        current: &[Scratch],
    ) -> Result<DecimateMetric, StageError> {
        let sizing = Sizing(context);

        let mut launched = Vec::with_capacity(current.len());
        for (plane_index, half) in self.halves.iter().enumerate() {
            let cells = Scratch::new::<u32>(context, self.across, self.down);
            let planes = BlockPlanes {
                previous: previous[plane_index].plane(),
                current: current[plane_index].plane(),
                cells: cells.plane(),
            };
            half_block_differences(&self.client, &sizing, planes, *half, self.across);

            launched.push(cells);
        }

        let cell_count = self.across as usize * self.down as usize;
        let mut totals = vec![0u64; cell_count];
        for cells in &launched {
            let read = self.read_u32(cells.plane());
            let values = read.map_err(unreadable_cells)?;
            for (total, value) in totals.iter_mut().zip(values) {
                *total += u64::from(value);
            }
        }

        let metric = DecimateMetric::from_cells(&totals, self.across as usize, self.down as usize);

        Ok(metric)
    }

    fn read_u32(&self, plane: DevicePlane<'_>) -> Result<Vec<u32>, ServerError> {
        let handle = plane.handle().clone();
        let bytes = self.client.read_one(handle)?;

        Ok(u32::from_bytes(&bytes).to_vec())
    }

    /// `frame` as the next output, at the output rate and counted on from the stream's first frame.
    fn emitted_frame(&mut self, frame: &Frame, start: StreamStart) -> Result<Frame, StageError> {
        let mut meta = frame.meta().clone();
        meta.index = start.index + self.emitted;
        if let (Some(rate), Some(first)) = (self.output.frame_rate, start.timestamp) {
            let offset = output_offset(rate, self.emitted);
            meta.timestamp = Some(first + offset);
        }

        let mut planes = Vec::with_capacity(self.input.channels().len());
        for channel in self.input.channels() {
            let plane = channel_plane(STAGE, frame, *channel)?;
            planes.push(plane.clone());
        }

        let built = Frame::from_planes(self.output, planes, meta);
        self.emitted += 1;

        built.map_err(bad_frame)
    }

    /// Every frame of the gathered cycle but the one it drops, in order.
    ///
    /// A cycle of one frame drops that frame, so it emits nothing.
    fn decimated(&mut self) -> Result<Vec<Frame>, StageError> {
        let frames = mem::take(&mut self.frames);
        let mut metrics = mem::take(&mut self.metrics);
        let cycle_index = self.cycle_index;
        self.cycle_index += 1;

        let Some(start) = self.start else {
            return Ok(Vec::new());
        };

        if frames.len() < 2 {
            return Ok(Vec::new());
        }

        if cycle_index == 0 {
            first_cycle_metrics(&mut metrics, &self.thresholds);
        }

        let dropped = find_drop(&metrics, &self.thresholds);

        let mut outputs = Vec::with_capacity(frames.len() - 1);
        for (index, frame) in frames.iter().enumerate() {
            if index == dropped {
                continue;
            }

            let output = self.emitted_frame(frame, start)?;
            outputs.push(output);
        }

        Ok(outputs)
    }
}

impl<R: Runtime> StageInstance for Decimator<R> {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.previous = None;
        self.frames.clear();
        self.metrics.clear();
        self.cycle_index = 0;
        self.emitted = 0;
        self.start = None;

        Ok(())
    }
}

impl<R: Runtime> Filter for Decimator<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(frame) = inputs.take_frame(INPUT) else {
            return Err(missing_input(STAGE));
        };

        let widened = self.widened_planes(context, &frame)?;
        let previous = self.previous.take();
        let measured_against = previous.as_deref().unwrap_or(&widened);
        let metric = self.metric(context, measured_against, &widened)?;
        self.previous = Some(widened);

        if self.start.is_none() {
            let meta = frame.meta();
            let start = StreamStart {
                index: meta.index,
                timestamp: meta.timestamp,
            };
            self.start = Some(start);
        }

        self.frames.push(frame);
        self.metrics.push(metric);
        if self.frames.len() < self.cycle {
            return Ok(Vec::new());
        }

        self.decimated()
    }

    fn end_of_stream(&mut self, _context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        if self.frames.is_empty() {
            return Ok(Vec::new());
        }

        self.decimated()
    }
}
