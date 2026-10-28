use std::collections::VecDeque;

use av_denoise_core::{ChannelMode, Engine, Geometry, Nl4d, Nlmeans};
use av_toolbox_core::{Artefact, Filter, Inputs, StageError, StageInstance, WantedArtefacts};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::values::{FloatCodes, offset, unit_to_float};
use av_toolbox_cubek::words::copy_plane;
use av_toolbox_gpu::{
    BuildContext,
    Channel,
    ComputeClient,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Plane,
    PlaneBuffer,
    Runtime,
    SampleFormat,
};
use cubecl::server::Handle;

use super::args::{AvdArgs, EngineOptions};
use super::grain::{GRAIN_TABLE, GrainLog};
use super::planes::{EngineSpec, engine_sample, engine_specs, needs_padding};
use crate::bits::bits_of;
use crate::codes::{CodeValues, sample_range};
use crate::masked::INPUT;
use crate::scratch::Scratch;
use crate::sizing::Sizing;

fn refused(stage: &str, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} failed in av-denoise: {error}");

    StageError::msg(message)
}

fn unbuildable(stage: &str, spec: &EngineSpec, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} could not build its {:?} engine: {error}", spec.channels);

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

fn out_of_step(stage: &str, expected: usize, found: usize) -> StageError {
    let message = format!("{stage} had engines ready with {expected} and {found} frames at once");

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

fn built_engine<R: Runtime>(
    client: &ComputeClient<R>,
    options: EngineOptions,
    geometry: Geometry,
) -> Result<Box<dyn Engine>, av_denoise_core::Error> {
    match options {
        EngineOptions::Nlmeans(algorithm) => {
            let engine = Nlmeans::new(client, algorithm, geometry)?;
            Ok(Box::new(engine))
        },
        EngineOptions::Nl4d(options) => {
            let engine = Nl4d::new(client, options, geometry)?;
            Ok(Box::new(engine))
        },
    }
}

/// Where av-denoise reads neutral float chroma, which av-toolbox stores at 0.
const CHROMA_CENTRE: f32 = 0.5;

/// How one plane reaches its engine.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Ingress {
    /// The frame's own plane is bound.
    Direct,
    /// The plane falls short of whole words, so it goes through scratch rounded up to them.
    Padded,
    /// Float chroma goes through scratch shifted by [CHROMA_CENTRE], since av-denoise reads it from 0 to 1.
    Centred,
}

/// How one plane comes back from its engine as F32.
#[derive(Debug, Copy, Clone, PartialEq)]
enum Egress {
    /// The engine writes the output plane itself.
    Direct,
    /// Float chroma comes back through scratch, shifted back from [CHROMA_CENTRE] to 0.
    Centred,
    /// The engine writes integer input as `code / max`, which these codes map onto full range samples.
    Rescaled(FloatCodes),
}

fn is_chroma(format: &FrameFormat, plane_index: usize) -> bool {
    let channel = format.channels()[plane_index];

    matches!(channel, Channel::U | Channel::V)
}

/// How the plane at `plane_index` of `format` reaches an engine of `spec`'s size.
fn ingress_of(format: &FrameFormat, spec: &EngineSpec, plane_index: usize) -> Ingress {
    let chroma = is_chroma(format, plane_index);
    if format.sample == SampleFormat::F32 && chroma {
        return Ingress::Centred;
    }

    if needs_padding(format.sample, spec.width, spec.height) {
        return Ingress::Padded;
    }

    Ingress::Direct
}

/// How the plane at `plane_index` of `format` comes back from its engine.
fn egress_of(format: &FrameFormat, plane_index: usize) -> Egress {
    let chroma = is_chroma(format, plane_index);

    if format.sample != SampleFormat::F32 {
        let bits = bits_of(format.sample);
        let range = sample_range(format.colour.range);
        let codes = FloatCodes::new(bits, range, chroma);
        return Egress::Rescaled(codes);
    }

    if chroma {
        return Egress::Centred;
    }

    Egress::Direct
}

/// The scratch handle when a plane was staged, otherwise the plane's own.
fn bound_handle<'a>(plane: &'a Plane, staged: &'a Option<PlaneBuffer>) -> &'a Handle {
    match staged {
        Some(scratch) => scratch.handle(),
        None => plane.buffer().handle(),
    }
}

/// One av-denoise engine and the planes it denoises.
struct PlaneEngine {
    engine: Box<dyn Engine>,
    spec: EngineSpec,
    /// How each of its planes reaches it, in the engine's plane order.
    ingress: Vec<Ingress>,
    /// How each of its planes comes back, in the engine's plane order.
    egress: Vec<Egress>,
    /// The settings the engine was built with, kept to rebuild it.
    options: EngineOptions,
    /// The size and samples the engine was built with, kept to rebuild it.
    geometry: Geometry,
}

impl PlaneEngine {
    /// Bytes of one padded plane, rounded up to whole words.
    fn padded_bytes(&self, sample: SampleFormat) -> usize {
        let samples = self.spec.width as usize * self.spec.height as usize;
        let bytes = samples * sample.bytes();

        bytes.next_multiple_of(4)
    }
}

/// Runs av-denoise engines over a stream, one engine per group of planes, all in lockstep.
pub(crate) struct AvdFilter<R: Runtime> {
    stage: &'static str,
    client: ComputeClient<R>,
    format: FrameFormat,
    /// The input's format with F32 samples, which every output frame has.
    output: FrameFormat,
    /// How planes no engine covers become F32.
    codes: CodeValues,
    engines: Vec<PlaneEngine>,
    /// Input frames waiting for their output, oldest first.
    held: VecDeque<Frame>,
    /// What the engine covering luma measured, when a grain table is wanted.
    grain: Option<GrainLog>,
}

impl<R: Runtime> AvdFilter<R> {
    pub(crate) fn new<A: AvdArgs>(
        context: &BuildContext<R>,
        args: &A,
        input: FrameFormat,
    ) -> Result<Self, StageError> {
        let sample = engine_sample(input.sample);
        let output = FrameFormat {
            sample: SampleFormat::F32,
            ..input
        };

        let client = context.client();
        let specs = engine_specs(A::STAGE, args.planes(), args.fused(), &input)?;

        let mut engines = Vec::with_capacity(specs.len());
        for spec in specs {
            let options = args.options(spec.channels)?;
            let geometry = Geometry {
                width: spec.width,
                height: spec.height,
                channels: spec.channels,
                input: sample,
                output: av_denoise_core::SampleFormat::F32,
            };

            let built = built_engine(client, options, geometry);
            let engine = built.map_err(|error| unbuildable(A::STAGE, &spec, error))?;
            let ingress = spec
                .planes
                .iter()
                .map(|plane_index| ingress_of(&input, &spec, *plane_index))
                .collect();
            let egress = spec
                .planes
                .iter()
                .map(|plane_index| egress_of(&input, *plane_index))
                .collect();

            engines.push(PlaneEngine {
                engine,
                spec,
                ingress,
                egress,
                options,
                geometry,
            });
        }

        Ok(Self {
            stage: A::STAGE,
            client: client.clone(),
            format: input,
            output,
            codes: CodeValues::new(&input, SampleFormat::F32),
            engines,
            held: VecDeque::new(),
            grain: None,
        })
    }

    /// The engine covering luma, which is the only one av-denoise measures grain with.
    fn luma_engine(&self) -> Option<usize> {
        self.engines
            .iter()
            .position(|engine| engine.spec.channels != ChannelMode::Chroma)
    }

    /// Rebuilds the engine covering luma to measure grain, and starts keeping what it measures.
    fn export_grain(&mut self) -> Result<(), StageError> {
        let Some(index) = self.luma_engine() else {
            return Err(no_luma(self.stage));
        };

        let Some(rate) = self.format.frame_rate else {
            return Err(no_frame_rate(self.stage));
        };

        let EngineOptions::Nl4d(mut options) = self.engines[index].options else {
            return Err(no_grain_export(self.stage));
        };

        options.grain_export = true;
        let exporting = EngineOptions::Nl4d(options);

        // The original engine is dropped before the exporting one is built, so only one holds the GPU.
        let PlaneEngine {
            engine,
            spec,
            ingress,
            egress,
            geometry,
            ..
        } = self.engines.remove(index);
        drop(engine);

        let built = built_engine(&self.client, exporting, geometry);
        let engine = built.map_err(|error| unbuildable(self.stage, &spec, error))?;
        let exporting_engine = PlaneEngine {
            engine,
            spec,
            ingress,
            egress,
            options: exporting,
            geometry,
        };
        self.engines.insert(index, exporting_engine);

        let numerator = u64::from(rate.numerator());
        let denominator = u64::from(rate.denominator());
        self.grain = Some(GrainLog::new(index, (numerator, denominator)));

        Ok(())
    }

    /// Keeps the grain the exporting engine measured over the stream that just ended.
    fn keep_grain(&mut self) -> Result<(), StageError> {
        let Some(grain) = self.grain.as_mut() else {
            return Ok(());
        };

        let stage = self.stage;
        let engine = &mut self.engines[grain.engine()].engine;
        let drained = engine.drain_grain_chunks();
        let chunks = drained.map_err(|error| refused(stage, error))?;

        grain.keep(chunks);

        Ok(())
    }

    fn plane_of<'a>(&self, frame: &'a Frame, index: usize) -> Result<&'a Plane, StageError> {
        let channel = self.format.channels()[index];

        frame
            .plane(channel)
            .ok_or_else(|| missing_channel(self.stage, channel))
    }

    /// Copies `plane` into word-rounded scratch, zeroing the tail.
    fn padded_copy(&self, context: &FrameContext, engine: &PlaneEngine, plane: &Plane) -> PlaneBuffer {
        let bytes = engine.padded_bytes(self.format.sample);
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

    /// The scratch `plane` is staged through on its way into an engine, if it needs any.
    fn staged_input(
        &self,
        context: &FrameContext,
        engine: &PlaneEngine,
        plane: &Plane,
        ingress: Ingress,
    ) -> Option<PlaneBuffer> {
        match ingress {
            Ingress::Direct => None,
            Ingress::Padded => Some(self.padded_copy(context, engine, plane)),
            Ingress::Centred => Some(self.centred_copy(context, plane)),
        }
    }

    /// Empty F32 scratch for an engine to write a plane into before it is restored, if it needs any.
    fn staged_output(context: &FrameContext, engine: &PlaneEngine, egress: Egress) -> Option<PlaneBuffer> {
        if egress == Egress::Direct {
            return None;
        }

        let samples = engine.spec.width as usize * engine.spec.height as usize;
        let bytes = samples * SampleFormat::F32.bytes();

        Some(context.scratch(bytes))
    }

    /// Pushes `frame` into every engine and returns how many frames are ready.
    fn push(&mut self, context: &FrameContext, frame: &Frame) -> Result<usize, StageError> {
        let mut ready = None;

        for index in 0..self.engines.len() {
            let count = self.push_engine(context, index, frame)?;

            match ready {
                Some(expected) if expected != count => return Err(out_of_step(self.stage, expected, count)),
                _ => ready = Some(count),
            }
        }

        Ok(ready.unwrap_or(0))
    }

    fn push_engine(
        &mut self,
        context: &FrameContext,
        index: usize,
        frame: &Frame,
    ) -> Result<usize, StageError> {
        let engine = &self.engines[index];
        let width = engine.spec.width;
        let height = engine.spec.height;

        let mut sources = Vec::with_capacity(engine.spec.planes.len());
        for plane_index in &engine.spec.planes {
            let plane = self.plane_of(frame, *plane_index)?;
            sources.push(plane);
        }

        let mut staged = Vec::with_capacity(sources.len());
        for (plane, ingress) in sources.iter().zip(&engine.ingress) {
            let scratch = self.staged_input(context, engine, plane, *ingress);
            staged.push(scratch);
        }

        let handles: Vec<_> = sources
            .iter()
            .zip(&staged)
            .map(|(plane, scratch)| bound_handle(plane, scratch))
            .collect();
        let planes: Vec<av_denoise_core::DevicePlane<'_>> = handles
            .iter()
            .map(|handle| av_denoise_core::DevicePlane::new(handle, width, height))
            .collect();

        let stage = self.stage;
        let pushed = self.engines[index].engine.push(&planes);

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

    /// Writes the engine's oldest ready frame into fresh planes, in the engine's plane order.
    fn emit_engine(
        &mut self,
        context: &FrameContext,
        index: usize,
        meta: &FrameMeta,
    ) -> Result<Vec<Plane>, StageError> {
        let engine = &self.engines[index];
        let width = engine.spec.width;
        let height = engine.spec.height;

        let mut outputs = Vec::with_capacity(engine.spec.planes.len());
        for _ in &engine.spec.planes {
            let plane = self.fresh_plane(context, (width, height), meta)?;
            outputs.push(plane);
        }

        let mut staged = Vec::with_capacity(outputs.len());
        for egress in &engine.egress {
            let scratch = Self::staged_output(context, engine, *egress);
            staged.push(scratch);
        }

        let handles: Vec<_> = outputs
            .iter()
            .zip(&staged)
            .map(|(plane, scratch)| bound_handle(plane, scratch))
            .collect();
        let planes: Vec<av_denoise_core::DevicePlane<'_>> = handles
            .iter()
            .map(|handle| av_denoise_core::DevicePlane::new(handle, width, height))
            .collect();

        let stage = self.stage;
        let emitted = self.engines[index].engine.emit_into(&planes);
        emitted.map_err(|error| refused(stage, error))?;

        let egress = self.engines[index].egress.clone();
        for ((output, scratch), egress) in outputs.iter().zip(&staged).zip(egress) {
            let Some(scratch) = scratch else {
                continue;
            };

            self.restore(context, egress, scratch, output);
        }

        Ok(outputs)
    }

    /// Writes an F32 scratch plane into its output plane, mapped onto full range samples.
    fn restore(&self, context: &FrameContext, egress: Egress, scratch: &PlaneBuffer, output: &Plane) {
        let samples = output.samples() as u32;

        match egress {
            Egress::Direct => {},
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

    /// Maps an engine's `code / max` floats in `scratch` onto full range samples in `output`.
    fn rescaled(&self, context: &FrameContext, scratch: &PlaneBuffer, output: &Plane, codes: FloatCodes) {
        let sizing = Sizing(context);
        let samples = output.samples() as u32;

        // Safety: the scratch holds `samples` f32 values.
        let source = unsafe { av_toolbox_cubek::DevicePlane::new(scratch.handle(), samples, 1) };

        // Safety: the output is an F32 plane of `samples` values.
        let target = unsafe { av_toolbox_cubek::DevicePlane::new(output.buffer().handle(), samples, 1) };

        unit_to_float(&self.client, &sizing, source, target, codes);
    }

    /// The plane at `index` of `held`, which no engine covers, as F32.
    fn unselected_plane(
        &self,
        context: &FrameContext,
        held: &Frame,
        index: usize,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let source = self.plane_of(held, index)?;
        if self.format.sample == SampleFormat::F32 {
            return Ok(source.clone());
        }

        let dimensions = source.dimensions();
        let (width, height) = dimensions;
        let target = self.fresh_plane(context, dimensions, meta)?;
        let channel = self.format.channels()[index];
        let sizing = Sizing(context);
        let values = Scratch::new::<f32>(context, width, height);

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane =
            unsafe { av_toolbox_cubek::DevicePlane::new(source.buffer().handle(), width, height) };

        // Safety: the target is an F32 plane of `width * height` values.
        let target_plane =
            unsafe { av_toolbox_cubek::DevicePlane::new(target.buffer().handle(), width, height) };

        self.codes
            .widen(&self.client, &sizing, source_plane, values.plane(), channel);
        self.codes
            .store(&self.client, &sizing, values.plane(), target_plane, channel);

        Ok(target)
    }

    /// Builds the F32 output of the oldest held frame from every engine's planes and the held frame's other planes.
    fn emit_one(&mut self, context: &FrameContext, held: Frame) -> Result<Frame, StageError> {
        let meta = held.meta().clone();
        let channels = self.format.channels();
        let mut denoised: Vec<Option<Plane>> = vec![None; channels.len()];

        for index in 0..self.engines.len() {
            let planes = self.emit_engine(context, index, &meta)?;
            let indices = self.engines[index].spec.planes.clone();
            for (plane_index, plane) in indices.into_iter().zip(planes) {
                denoised[plane_index] = Some(plane);
            }
        }

        let mut complete = Vec::with_capacity(channels.len());
        for (index, slot) in denoised.into_iter().enumerate() {
            let plane = match slot {
                Some(plane) => plane,
                None => self.unselected_plane(context, &held, index, &meta)?,
            };
            complete.push(plane);
        }

        Frame::from_planes(self.output, complete, meta).map_err(|error| bad_frame(self.stage, error))
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

impl<R: Runtime> StageInstance for AvdFilter<R> {
    fn start_of_stream(&mut self, _context: &mut FrameContext) -> Result<(), StageError> {
        for engine in &mut self.engines {
            engine.engine.reset();
        }

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

impl<R: Runtime> Filter for AvdFilter<R> {
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
        let mut tail = None;

        for engine in &mut self.engines {
            let finished = engine.engine.finish();
            let count = finished.map_err(|error| refused(self.stage, error))?;

            match tail {
                Some(expected) if expected != count => return Err(out_of_step(self.stage, expected, count)),
                _ => tail = Some(count),
            }
        }

        let ready = tail.unwrap_or(0);
        let frames = self.emit(context, ready)?;

        if !self.held.is_empty() {
            return Err(left_over(self.stage, self.held.len()));
        }

        self.keep_grain()?;

        Ok(frames)
    }
}
