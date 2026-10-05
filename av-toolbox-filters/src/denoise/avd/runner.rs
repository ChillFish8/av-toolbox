use std::collections::VecDeque;

use av_denoise_core::{Engine, Geometry, Nl4d, Nlmeans};
use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_cubek::depth::Element;
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

use super::args::{AvdArgs, EngineOptions};
use super::planes::{EngineSpec, engine_sample, engine_specs, needs_padding};
use crate::masked::INPUT;
use crate::sizing::Sizing;

fn refused(stage: &str, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} failed in av-denoise: {error}");

    StageError::msg(message)
}

fn unbuildable(stage: &str, spec: &EngineSpec, error: av_denoise_core::Error) -> StageError {
    let message = format!("{stage} could not build its {:?} engine: {error}", spec.channels);

    StageError::msg(message)
}

fn unsupported_sample(stage: &str, sample: SampleFormat) -> StageError {
    let message = format!("{stage} needs U8, U16 or F32 samples, got {sample}");

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

/// One av-denoise engine and the planes it denoises.
struct PlaneEngine {
    engine: Box<dyn Engine>,
    spec: EngineSpec,
    /// Whether its planes fall short of whole words and go through padded scratch.
    padded: bool,
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
    engines: Vec<PlaneEngine>,
    /// Input frames waiting for their output, oldest first.
    held: VecDeque<Frame>,
}

impl<R: Runtime> AvdFilter<R> {
    pub(crate) fn new<A: AvdArgs>(
        context: &BuildContext<R>,
        args: &A,
        input: FrameFormat,
    ) -> Result<Self, StageError> {
        let Some(sample) = engine_sample(input.sample) else {
            return Err(unsupported_sample(A::STAGE, input.sample));
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
                output: sample,
            };

            let built = built_engine(client, options, geometry);
            let engine = built.map_err(|error| unbuildable(A::STAGE, &spec, error))?;
            let padded = needs_padding(input.sample, spec.width, spec.height);

            engines.push(PlaneEngine { engine, spec, padded });
        }

        Ok(Self {
            stage: A::STAGE,
            client: client.clone(),
            format: input,
            engines,
            held: VecDeque::new(),
        })
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

        let staged: Vec<PlaneBuffer> = match engine.padded {
            true => sources
                .iter()
                .map(|plane| self.padded_copy(context, engine, plane))
                .collect(),
            false => Vec::new(),
        };

        let handles: Vec<_> = match engine.padded {
            true => staged.iter().map(PlaneBuffer::handle).collect(),
            false => sources.iter().map(|plane| plane.buffer().handle()).collect(),
        };
        let planes: Vec<av_denoise_core::DevicePlane<'_>> = handles
            .iter()
            .map(|handle| av_denoise_core::DevicePlane::new(handle, width, height))
            .collect();

        let stage = self.stage;
        let pushed = self.engines[index].engine.push(&planes);

        pushed.map_err(|error| refused(stage, error))
    }

    /// A fresh plane of the engine's size and the frame's sample format.
    fn fresh_plane(
        &self,
        context: &FrameContext,
        engine: &PlaneEngine,
        meta: &FrameMeta,
    ) -> Result<Plane, StageError> {
        let plane_format = FrameFormat {
            width: engine.spec.width,
            height: engine.spec.height,
            layout: FrameLayout::Grey,
            ..self.format
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
            let plane = self.fresh_plane(context, engine, meta)?;
            outputs.push(plane);
        }

        let staged: Vec<PlaneBuffer> = match engine.padded {
            true => {
                let bytes = engine.padded_bytes(self.format.sample);
                outputs.iter().map(|_| context.scratch(bytes)).collect()
            },
            false => Vec::new(),
        };

        let handles: Vec<_> = match engine.padded {
            true => staged.iter().map(PlaneBuffer::handle).collect(),
            false => outputs.iter().map(|plane| plane.buffer().handle()).collect(),
        };
        let planes: Vec<av_denoise_core::DevicePlane<'_>> = handles
            .iter()
            .map(|handle| av_denoise_core::DevicePlane::new(handle, width, height))
            .collect();

        let stage = self.stage;
        let emitted = self.engines[index].engine.emit_into(&planes);
        emitted.map_err(|error| refused(stage, error))?;

        if self.engines[index].padded {
            self.unpad(context, &staged, &outputs);
        }

        Ok(outputs)
    }

    /// Copies each padded scratch plane back into its output plane.
    fn unpad(&self, context: &FrameContext, staged: &[PlaneBuffer], outputs: &[Plane]) {
        let element = element_of(self.format.sample);
        let sizing = Sizing(context);

        for (scratch, output) in staged.iter().zip(outputs) {
            let (width, height) = output.dimensions();
            let samples = width * height;
            let padded_samples = (scratch.bytes() / self.format.sample.bytes()) as u32;

            // Safety: the scratch holds `padded_samples` samples of the frame's sample format.
            let source = unsafe { av_toolbox_cubek::DevicePlane::new(scratch.handle(), padded_samples, 1) };

            // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
            let target = unsafe { av_toolbox_cubek::DevicePlane::new(output.buffer().handle(), samples, 1) };

            copy_plane(&self.client, &sizing, source, target, element);
        }
    }

    /// Builds the output of the oldest held frame from every engine's planes and the held frame's other planes.
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
                None => self.plane_of(&held, index)?.clone(),
            };
            complete.push(plane);
        }

        Frame::from_planes(self.format, complete, meta).map_err(|error| bad_frame(self.stage, error))
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

        Ok(())
    }
}

impl<R: Runtime> Filter for AvdFilter<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(frame) = inputs.take_frame(INPUT) else {
            return Err(missing_frame(self.stage));
        };

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

        Ok(frames)
    }
}
