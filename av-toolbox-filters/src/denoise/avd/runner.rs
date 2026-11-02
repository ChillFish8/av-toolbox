use std::collections::VecDeque;

use av_denoise_core::{DevicePlane, Engine, EngineSpec, Nl4dOptions, TuningConfig, build_engine};
use av_toolbox_core::{Artefact, Filter, Inputs, StageError, StageInstance, WantedArtefacts};
use av_toolbox_cubek::depth::{Element, Source};
use av_toolbox_cubek::values::{FloatCodes, codes_to_float, offset};
use av_toolbox_cubek::words::copy_plane;
use av_toolbox_gpu::{
    BuildContext,
    Channel,
    Client,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Plane,
    PlaneBuffer,
    SampleFormat,
};
use cubecl::server::Handle;

use super::args::{AvdArgs, engine_spec};
use super::grain::{GRAIN_TABLE, GrainLog};
use super::planes::{PlaneGroups, needs_padding, normalises, plane_groups, stage_geometry};
use super::tuning::tuning_config;
use crate::bits::bits_of;
use crate::codes::sample_range;
use crate::masked::INPUT;
use crate::sizing::Sizing;

fn refused(stage: &str, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} failed in av-denoise: {error}");

    StageError::msg(message)
}

fn unbuildable(stage: &str, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} could not build its engine: {error}");

    StageError::msg(message)
}

fn no_engine(stage: &str) -> StageError {
    let message = format!("{stage} has no engine after failing to rebuild it");

    StageError::msg(message)
}

fn missing_frame(stage: &str) -> StageError {
    let message = format!("{stage} expected a frame on its input");

    StageError::msg(message)
}

fn missing_channel(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn nothing_held(stage: &str) -> StageError {
    let message = format!("{stage} had a frame ready with no input frame held for it");

    StageError::msg(message)
}

fn left_over(stage: &str, frames: usize) -> StageError {
    let message = format!("{stage} ended its stream still holding {frames} frames");

    StageError::msg(message)
}

fn no_luma(stage: &str) -> StageError {
    let message = format!("{stage} measures grain on luma, and its planes leave luma out");

    StageError::msg(message)
}

fn no_frame_rate(stage: &str) -> StageError {
    let message = format!("{stage} needs the stream's frame rate to time a grain table");

    StageError::msg(message)
}

fn no_grain_export(stage: &str) -> StageError {
    let message = format!("{stage} has no engine that measures grain");

    StageError::msg(message)
}

fn bad_frame(stage: &str, error: FrameError) -> StageError {
    let message = format!("{stage} built a bad frame: {error}");

    StageError::msg(message)
}

/// The element a padded copy moves, which only integer planes ever need.
fn element_of(sample: SampleFormat) -> Element {
    match sample {
        SampleFormat::U8 => Element::U8,
        _ => Element::U16,
    }
}

/// How integer samples of `sample` are read by a conversion kernel.
fn source_of(sample: SampleFormat) -> Source {
    match sample {
        SampleFormat::U8 => Source::U8,
        SampleFormat::U16 { .. } => Source::U16,
        SampleFormat::F32 => Source::F32,
    }
}

/// Whether `spec` denoises luma.
fn denoises_luma(spec: &EngineSpec) -> bool {
    match spec {
        EngineSpec::Nlmeans { luma, .. } => luma.is_some(),
        EngineSpec::Nl4d { luma, .. } => luma.is_some(),
    }
}

/// `spec` with its luma group measuring grain, which only nl4d can.
fn grain_exporting(stage: &str, spec: EngineSpec) -> Result<EngineSpec, StageError> {
    let EngineSpec::Nl4d {
        luma: Some(options),
        chroma,
        geometry,
    } = spec
    else {
        return Err(no_grain_export(stage));
    };

    let luma = Nl4dOptions {
        grain_export: true,
        ..options
    };

    Ok(EngineSpec::Nl4d {
        luma: Some(luma),
        chroma,
        geometry,
    })
}

/// Where av-denoise reads neutral float chroma, which av-toolbox stores at 0.
const CHROMA_CENTRE: f32 = 0.5;

/// How one plane reaches the engine.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Ingress {
    /// The frame's own plane is bound.
    Direct,
    /// The plane falls short of whole words, so it goes through scratch rounded up to them.
    Padded,
    /// Float chroma goes through scratch shifted by [CHROMA_CENTRE], since av-denoise reads it from 0 to 1.
    Centred,
    /// Integer codes go through F32 scratch as `code / max`, where av-denoise reads luma and chroma alike.
    ///
    /// nl4d then measures colour on an 8-bit scale, so U16 neutral chroma reads about 127.6 rather than 128.
    Normalised,
}

/// How one plane comes back from the engine as F32.
#[derive(Debug, Copy, Clone, PartialEq)]
enum Egress {
    /// The engine writes the output plane itself.
    Direct,
    /// Float chroma comes back through scratch, shifted back from [CHROMA_CENTRE] to 0.
    Centred,
    /// The engine writes integer input as `code / max`, which these codes map onto full range samples.
    Rescaled(FloatCodes),
    /// A float plane the engine passes through goes to throwaway scratch, and the input plane is kept.
    Held,
}

/// One plane of the frame on its way through the engine.
#[derive(Debug, Copy, Clone, PartialEq)]
struct PlanePath {
    width: u32,
    height: u32,
    ingress: Ingress,
    egress: Egress,
}

impl PlanePath {
    /// The path of the plane at `plane_index` of `format`.
    ///
    /// Float chroma is always centred, since nl4d reads chroma it does not denoise to measure colour.
    fn new(
        stage: &str,
        format: &FrameFormat,
        groups: PlaneGroups,
        plane_index: usize,
    ) -> Result<Self, StageError> {
        let channel = format.channels()[plane_index];
        let Some((width, height)) = format.channel_dimensions(channel) else {
            return Err(missing_channel(stage, channel));
        };

        let chroma = matches!(channel, Channel::U | Channel::V);
        let denoised = if chroma { groups.chroma } else { groups.luma };
        let float = format.sample == SampleFormat::F32;
        let centred = float && chroma;

        let ingress = if normalises(format, groups) {
            Ingress::Normalised
        } else if centred {
            Ingress::Centred
        } else if needs_padding(format.sample, width, height) {
            Ingress::Padded
        } else {
            Ingress::Direct
        };

        let egress = if !float {
            let bits = bits_of(format.sample);
            let range = sample_range(format.colour.range);
            let codes = FloatCodes::new(bits, range, chroma);
            Egress::Rescaled(codes)
        } else if !denoised {
            Egress::Held
        } else if centred {
            Egress::Centred
        } else {
            Egress::Direct
        };

        Ok(Self {
            width,
            height,
            ingress,
            egress,
        })
    }

    fn samples(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Bytes of one padded plane, rounded up to whole words.
    fn padded_bytes(&self, sample: SampleFormat) -> usize {
        let bytes = self.samples() * sample.bytes();

        bytes.next_multiple_of(4)
    }
}

/// The scratch handle when a plane was staged, otherwise the plane's own.
fn bound_handle<'a>(plane: &'a Plane, staged: &'a Option<PlaneBuffer>) -> &'a Handle {
    match staged {
        Some(scratch) => scratch.handle(),
        None => plane.buffer().handle(),
    }
}

/// Every plane of a frame as av-denoise takes them, luma first.
fn device_planes<'a>(handles: &[&'a Handle], paths: &[PlanePath]) -> Vec<DevicePlane<'a>> {
    handles
        .iter()
        .zip(paths)
        .map(|(handle, path)| DevicePlane::new(handle, path.width, path.height))
        .collect()
}

/// Runs one av-denoise engine over a stream, taking and writing every plane of each frame.
pub(crate) struct AvdFilter {
    stage: &'static str,
    client: Client,
    device: cubecl::Device,
    tuning: TuningConfig,
    format: FrameFormat,
    /// The input's format with F32 samples, which every output frame has.
    output: FrameFormat,
    /// The engine, missing only after a rebuild failed.
    engine: Option<Box<dyn Engine>>,
    /// What the engine was built from, kept to rebuild it.
    spec: EngineSpec,
    /// How each plane of the frame travels through the engine, luma first.
    paths: Vec<PlanePath>,
    /// Input frames waiting for their output, oldest first.
    held: VecDeque<Frame>,
    /// What the engine measured on luma, when a grain table is wanted.
    grain: Option<GrainLog>,
}

impl AvdFilter {
    pub(crate) fn new<A: AvdArgs>(
        context: &BuildContext,
        args: &A,
        input: FrameFormat,
    ) -> Result<Self, StageError> {
        let output = FrameFormat {
            sample: SampleFormat::F32,
            ..input
        };

        let groups = plane_groups(A::STAGE, args.planes(), &input)?;
        let geometry = stage_geometry(&input, groups);
        let spec = engine_spec(args, groups, geometry)?;

        let plane_count = input.channels().len();
        let mut paths = Vec::with_capacity(plane_count);
        for plane_index in 0..plane_count {
            let path = PlanePath::new(A::STAGE, &input, groups, plane_index)?;
            paths.push(path);
        }

        let client = context.client();
        let device = context.cubecl_device();
        let tuning = tuning_config(context.tuning(), None);
        let built = build_engine(device, client, spec, &tuning);
        let engine = built.map_err(|error| unbuildable(A::STAGE, error))?;

        Ok(Self {
            stage: A::STAGE,
            client: client.clone(),
            device: device.clone(),
            tuning,
            format: input,
            output,
            engine: Some(engine),
            spec,
            paths,
            held: VecDeque::new(),
            grain: None,
        })
    }

    fn engine(&mut self) -> Result<&mut Box<dyn Engine>, StageError> {
        let stage = self.stage;

        self.engine.as_mut().ok_or_else(|| no_engine(stage))
    }

    /// Rebuilds the engine to measure grain on luma, and starts keeping what it measures.
    fn export_grain(&mut self) -> Result<(), StageError> {
        if !denoises_luma(&self.spec) {
            return Err(no_luma(self.stage));
        }

        let Some(rate) = self.format.frame_rate else {
            return Err(no_frame_rate(self.stage));
        };

        let spec = grain_exporting(self.stage, self.spec)?;

        // The original engine is dropped before the exporting one is built, so only one holds the GPU.
        drop(self.engine.take());

        let built = build_engine(&self.device, &self.client, spec, &self.tuning);
        let engine = built.map_err(|error| unbuildable(self.stage, error))?;
        self.engine = Some(engine);
        self.spec = spec;

        let numerator = u64::from(rate.numerator());
        let denominator = u64::from(rate.denominator());
        self.grain = Some(GrainLog::new((numerator, denominator)));

        Ok(())
    }

    /// Keeps the grain the engine measured over the stream that just ended.
    fn keep_grain(&mut self) -> Result<(), StageError> {
        if self.grain.is_none() {
            return Ok(());
        }

        let stage = self.stage;
        let drained = self.engine()?.drain_grain_chunks();
        let chunks = drained.map_err(|error| refused(stage, error))?;

        if let Some(grain) = self.grain.as_mut() {
            grain.keep(chunks);
        }

        Ok(())
    }

    fn plane_of<'a>(&self, frame: &'a Frame, index: usize) -> Result<&'a Plane, StageError> {
        let channel = self.format.channels()[index];

        frame
            .plane(channel)
            .ok_or_else(|| missing_channel(self.stage, channel))
    }

    /// Copies `plane` into word-rounded scratch, zeroing the tail.
    fn padded_copy(&self, context: &FrameContext, path: &PlanePath, plane: &Plane) -> PlaneBuffer {
        let bytes = path.padded_bytes(self.format.sample);
        let scratch = context.scratch(bytes);
        let padded_samples = (bytes / self.format.sample.bytes()) as u32;
        let element = element_of(self.format.sample);
        let sizing = Sizing(context);
        let (width, height) = plane.dimensions();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source =
            unsafe { av_toolbox_cubek::DevicePlane::new(plane.buffer().handle(), width * height, 1) };

        // Safety: the scratch holds `padded_samples` samples of the frame's sample format.
        let target = unsafe { av_toolbox_cubek::DevicePlane::new(scratch.handle(), padded_samples, 1) };

        copy_plane(&self.client, &sizing, source, target, element);

        scratch
    }

    /// Writes `source` plus `amount` into `target`, two f32 planes of the same size.
    fn shifted(&self, context: &FrameContext, source: &Handle, target: &Handle, samples: u32, amount: f32) {
        let sizing = Sizing(context);

        // Safety: both buffers hold `samples` f32 values.
        let source_plane = unsafe { av_toolbox_cubek::DevicePlane::new(source, samples, 1) };

        // Safety: both buffers hold `samples` f32 values.
        let target_plane = unsafe { av_toolbox_cubek::DevicePlane::new(target, samples, 1) };

        offset(&self.client, &sizing, source_plane, target_plane, amount);
    }

    /// Copies float chroma into scratch, moved from centred on 0 to centred on [CHROMA_CENTRE].
    fn centred_copy(&self, context: &FrameContext, plane: &Plane) -> PlaneBuffer {
        let scratch = context.scratch(plane.buffer().bytes());
        let samples = plane.samples() as u32;

        self.shifted(
            context,
            plane.buffer().handle(),
            scratch.handle(),
            samples,
            CHROMA_CENTRE,
        );

        scratch
    }

    /// Copies integer `plane` into F32 scratch as `code / max`.
    fn normalised_copy(&self, context: &FrameContext, path: &PlanePath, plane: &Plane) -> PlaneBuffer {
        let samples = path.samples();
        let scratch = context.scratch(samples * SampleFormat::F32.bytes());
        let bits = bits_of(self.format.sample);
        let peak = ((1u32 << bits) - 1) as f32;
        let codes = FloatCodes {
            offset: 0.0,
            scale: peak,
            peak,
        };
        let source = source_of(self.format.sample);
        let sizing = Sizing(context);

        // Safety: a frame's plane holds exactly `samples` samples of its sample format.
        let source_plane =
            unsafe { av_toolbox_cubek::DevicePlane::new(plane.buffer().handle(), samples as u32, 1) };

        // Safety: the scratch holds `samples` f32 values.
        let target_plane = unsafe { av_toolbox_cubek::DevicePlane::new(scratch.handle(), samples as u32, 1) };

        codes_to_float(&self.client, &sizing, source_plane, target_plane, source, codes);

        scratch
    }

    /// The scratch `plane` is staged through on its way into the engine, if it needs any.
    fn staged_input(&self, context: &FrameContext, path: &PlanePath, plane: &Plane) -> Option<PlaneBuffer> {
        match path.ingress {
            Ingress::Direct => None,
            Ingress::Padded => Some(self.padded_copy(context, path, plane)),
            Ingress::Centred => Some(self.centred_copy(context, plane)),
            Ingress::Normalised => Some(self.normalised_copy(context, path, plane)),
        }
    }

    /// Empty F32 scratch for the engine to write a plane into before it is restored, if it needs any.
    fn staged_output(context: &FrameContext, path: &PlanePath) -> Option<PlaneBuffer> {
        if path.egress == Egress::Direct {
            return None;
        }

        let bytes = path.samples() * SampleFormat::F32.bytes();

        Some(context.scratch(bytes))
    }

    /// Pushes every plane of `frame` into the engine and returns how many frames are ready.
    fn push(&mut self, context: &FrameContext, frame: &Frame) -> Result<usize, StageError> {
        let mut sources = Vec::with_capacity(self.paths.len());
        for plane_index in 0..self.paths.len() {
            let plane = self.plane_of(frame, plane_index)?;
            sources.push(plane);
        }

        let mut staged = Vec::with_capacity(sources.len());
        for (plane, path) in sources.iter().zip(&self.paths) {
            let scratch = self.staged_input(context, path, plane);
            staged.push(scratch);
        }

        let handles: Vec<&Handle> = sources
            .iter()
            .zip(&staged)
            .map(|(plane, scratch)| bound_handle(plane, scratch))
            .collect();
        let planes = device_planes(&handles, &self.paths);

        let stage = self.stage;
        let pushed = self.engine()?.push(&planes);

        pushed.map_err(|error| refused(stage, error))
    }

    /// A fresh F32 plane of `width` by `height`.
    fn fresh_plane(
        &self,
        context: &FrameContext,
        (width, height): (u32, u32),
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.output
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let plane = target
            .plane(Channel::Y)
            .ok_or_else(|| missing_channel(self.stage, Channel::Y))?;

        Ok(plane.clone())
    }

    /// The plane the engine's output of `plane_index` ends up in, which a held plane takes from `held` as it is.
    fn output_plane(
        &self,
        context: &FrameContext,
        held: &Frame,
        plane_index: usize,
        path: &PlanePath,
    ) -> Result<Plane, StageError> {
        if path.egress == Egress::Held {
            let plane = self.plane_of(held, plane_index)?;
            return Ok(plane.clone());
        }

        let meta = held.meta();

        self.fresh_plane(context, (path.width, path.height), meta)
    }

    /// Writes the engine's oldest ready frame into F32 planes, luma first.
    fn emit_planes(&mut self, context: &FrameContext, held: &Frame) -> Result<Vec<Plane>, StageError> {
        let mut outputs = Vec::with_capacity(self.paths.len());
        for (plane_index, path) in self.paths.iter().enumerate() {
            let plane = self.output_plane(context, held, plane_index, path)?;
            outputs.push(plane);
        }

        let mut staged = Vec::with_capacity(outputs.len());
        for path in &self.paths {
            let scratch = Self::staged_output(context, path);
            staged.push(scratch);
        }

        let handles: Vec<&Handle> = outputs
            .iter()
            .zip(&staged)
            .map(|(plane, scratch)| bound_handle(plane, scratch))
            .collect();
        let planes = device_planes(&handles, &self.paths);

        let stage = self.stage;
        let emitted = self.engine()?.emit_into(&planes);
        emitted.map_err(|error| refused(stage, error))?;

        for ((output, scratch), path) in outputs.iter().zip(&staged).zip(&self.paths) {
            let Some(scratch) = scratch else {
                continue;
            };

            self.restore(context, path.egress, scratch, output);
        }

        Ok(outputs)
    }

    /// Writes an F32 scratch plane into its output plane, mapped onto full range samples.
    fn restore(&self, context: &FrameContext, egress: Egress, scratch: &PlaneBuffer, output: &Plane) {
        let samples = output.samples() as u32;

        match egress {
            Egress::Direct | Egress::Held => {},
            Egress::Centred => {
                let amount = -CHROMA_CENTRE;
                self.shifted(
                    context,
                    scratch.handle(),
                    output.buffer().handle(),
                    samples,
                    amount,
                );
            },
            Egress::Rescaled(codes) => self.rescaled(context, scratch, output, codes),
        }
    }

    /// Maps the engine's `code / max` floats in `scratch` onto full range samples in `output`.
    fn rescaled(&self, context: &FrameContext, scratch: &PlaneBuffer, output: &Plane, codes: FloatCodes) {
        let sizing = Sizing(context);
        let samples = output.samples() as u32;

        // Safety: the scratch holds `samples` f32 values.
        let source = unsafe { av_toolbox_cubek::DevicePlane::new(scratch.handle(), samples, 1) };

        // Safety: the output is an F32 plane of `samples` values.
        let target = unsafe { av_toolbox_cubek::DevicePlane::new(output.buffer().handle(), samples, 1) };

        let unit_codes = codes.per_unit();
        codes_to_float(&self.client, &sizing, source, target, Source::F32, unit_codes);
    }

    /// Builds the F32 output of the oldest held frame from the engine's planes.
    fn emit_one(&mut self, context: &FrameContext, held: Frame) -> Result<Frame, StageError> {
        let planes = self.emit_planes(context, &held)?;
        let meta = held.meta().clone();

        Frame::from_planes(self.output, planes, meta).map_err(|error| bad_frame(self.stage, error))
    }

    fn emit(&mut self, context: &FrameContext, ready: usize) -> Result<Vec<Frame>, StageError> {
        let mut frames = Vec::with_capacity(ready);

        for _ in 0..ready {
            let Some(held) = self.held.pop_front() else {
                return Err(nothing_held(self.stage));
            };

            let frame = self.emit_one(context, held)?;
            frames.push(frame);
        }

        Ok(frames)
    }
}

impl StageInstance for AvdFilter {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        self.engine()?.reset();
        self.held.clear();

        if let Some(grain) = self.grain.as_mut() {
            grain.restart();
        }

        Ok(())
    }

    fn wants_artefacts(&mut self, wanted: &WantedArtefacts) -> Result<(), StageError> {
        if !wanted.contains(GRAIN_TABLE) {
            return Ok(());
        }

        self.export_grain()
    }

    fn take_artefacts(&mut self, _context: &mut FrameContext) -> Result<Vec<Artefact>, StageError> {
        let Some(grain) = self.grain.as_mut() else {
            return Ok(Vec::new());
        };

        let piece = grain.take_piece()?;

        Ok(vec![piece])
    }
}

impl Filter for AvdFilter {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(frame) = inputs.take_frame(INPUT) else {
            return Err(missing_frame(self.stage));
        };

        if let Some(grain) = self.grain.as_mut() {
            grain.saw(frame.meta().index);
        }

        let ready = self.push(context, &frame)?;
        self.held.push_back(frame);

        self.emit(context, ready)
    }

    fn end_of_stream(&mut self, context: &mut FrameContext) -> Result<Vec<Frame>, StageError> {
        let stage = self.stage;
        let finished = self.engine()?.finish();
        let ready = finished.map_err(|error| refused(stage, error))?;

        let frames = self.emit(context, ready)?;

        if !self.held.is_empty() {
            return Err(left_over(self.stage, self.held.len()));
        }

        self.keep_grain()?;

        Ok(frames)
    }
}
