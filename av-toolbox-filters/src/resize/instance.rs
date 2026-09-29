use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::resize::{Kernel, ResizePlanes, Samples, Weights, apply, taps, upload};
use av_toolbox_gpu::{
    Channel,
    ComputeClient,
    Frame,
    FrameContext,
    FrameFormat,
    Plane,
    Runtime,
    SampleFormat,
};

use super::geometry::{Geometry, PlaneShape};
use crate::sizing::Sizing;

/// One plane's weight tables on the device.
pub(crate) struct Tables {
    across: Weights,
    down: Weights,
}

impl Tables {
    pub(crate) fn new<R: Runtime>(client: &ComputeClient<R>, shape: &PlaneShape, kernel: &Kernel) -> Self {
        let across_taps = taps(shape.width, shape.across.scale, shape.across.offset, kernel);
        let down_taps = taps(shape.height, shape.down.scale, shape.down.offset, kernel);
        let across = upload(client, &across_taps);
        let down = upload(client, &down_taps);

        Self { across, down }
    }
}

/// How a resize stores samples of `sample`.
pub(crate) fn samples_for(sample: SampleFormat) -> Samples {
    match sample {
        SampleFormat::U16 { depth } => Samples::U16 {
            largest: depth.max_value(),
        },
        _ => Samples::U8,
    }
}

/// Resizes `input` into `output` through `tables`, whatever their sizes.
pub(crate) fn resize_plane<R: Runtime>(
    client: &ComputeClient<R>,
    context: &FrameContext,
    input: &Plane,
    output: &Plane,
    tables: &Tables,
    samples: Samples,
) {
    let (in_width, in_height) = input.dimensions();
    let (out_width, out_height) = output.dimensions();
    let scratch_bytes = out_width as usize * in_height as usize * size_of::<f32>();
    let scratch = context.scratch(scratch_bytes);

    let input_handle = input.buffer().handle();
    let scratch_handle = scratch.handle();
    let output_handle = output.buffer().handle();

    // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
    let input_plane = unsafe { DevicePlane::new(input_handle, in_width, in_height) };

    // Safety: the scratch was sized for exactly `out_width * in_height` f32 samples.
    let scratch_plane = unsafe { DevicePlane::new(scratch_handle, out_width, in_height) };

    // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
    let output_plane = unsafe { DevicePlane::new(output_handle, out_width, out_height) };

    let planes = ResizePlanes {
        input: input_plane,
        scratch: scratch_plane,
        output: output_plane,
    };
    let sizing = Sizing(context);
    apply(client, &sizing, planes, &tables.across, &tables.down, samples);
}

fn missing_input(stage: &str) -> StageError {
    let message = format!("{stage} expected a frame in slot 0");

    StageError::msg(message)
}

fn missing_channel(stage: &str) -> StageError {
    let message = format!("{stage} was handed a frame missing a channel");

    StageError::msg(message)
}

pub(super) struct Resizer<R: Runtime> {
    client: ComputeClient<R>,
    stage: String,
    output: FrameFormat,
    samples: Samples,
    luma: Tables,
    /// The subsampled chroma planes' tables, `None` when chroma is read like luma.
    chroma: Option<Tables>,
    identity: bool,
}

impl<R: Runtime> Resizer<R> {
    /// Builds and uploads every plane's weight tables once.
    pub(super) fn new(
        client: ComputeClient<R>,
        stage: String,
        output: FrameFormat,
        shapes: &Geometry,
        kernel: &Kernel,
        identity: bool,
    ) -> Self {
        let samples = samples_for(output.sample);
        let luma = Tables::new(&client, &shapes.luma, kernel);
        let chroma = shapes.chroma.map(|shape| Tables::new(&client, &shape, kernel));

        Self {
            client,
            stage,
            output,
            samples,
            luma,
            chroma,
            identity,
        }
    }

    fn tables_for(&self, channel: Channel) -> &Tables {
        let is_chroma = matches!(channel, Channel::U | Channel::V);

        match (&self.chroma, is_chroma) {
            (Some(chroma), true) => chroma,
            _ => &self.luma,
        }
    }
}

impl<R: Runtime> StageInstance for Resizer<R> {}

impl<R: Runtime> Filter for Resizer<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Frame, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input(&self.stage));
        };

        if self.identity {
            return Ok(input);
        }

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);

        for channel in self.output.channels() {
            let Some(input_plane) = input.plane(*channel) else {
                return Err(missing_channel(&self.stage));
            };

            let Some(output_plane) = output.plane(*channel) else {
                return Err(missing_channel(&self.stage));
            };

            let tables = self.tables_for(*channel);

            resize_plane(
                &self.client,
                context,
                input_plane,
                output_plane,
                tables,
                self.samples,
            );
        }

        Ok(output)
    }
}
