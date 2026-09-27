use av_toolbox_core::StageError;
use av_toolbox_cubek::DevicePlane;
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::values::{narrow, widen};
use av_toolbox_gpu::{
    Channel,
    ComputeClient,
    Frame,
    FrameContext,
    FrameError,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    Plane,
    Runtime,
};

use crate::scratch::Scratch;
use crate::sizing::Sizing;

/// Filters one plane at a time, in f32 code values.
pub(crate) trait PlaneFilter {
    /// Filters `input` into `output`, both f32 planes of one size.
    fn filter(
        &self,
        context: &FrameContext,
        plane_index: usize,
        input: DevicePlane<'_>,
        output: DevicePlane<'_>,
    ) -> Result<(), StageError>;
}

fn missing_channel(stage: &str, channel: Channel) -> StageError {
    let message = format!("{stage} was handed a frame missing {channel}");

    StageError::msg(message)
}

fn bad_frame(stage: &str, error: FrameError) -> StageError {
    let message = format!("{stage} built a bad frame: {error}");

    StageError::msg(message)
}

/// How a stage's frames are stored, and which of their planes it filters.
pub(crate) struct Planewise {
    pub(crate) stage: &'static str,
    pub(crate) format: FrameFormat,
    /// Whether each plane, in channel order, is filtered.
    pub(crate) selected: Vec<bool>,
    pub(crate) element: Element,
    pub(crate) peak: f32,
}

impl Planewise {
    /// Runs `filter` on every selected plane of `frame`, sharing the rest.
    pub(crate) fn run<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        context: &FrameContext,
        frame: Frame,
        filter: &impl PlaneFilter,
    ) -> Result<Frame, StageError> {
        let meta = frame.meta().clone();
        let channels = self.format.channels();
        let mut planes = Vec::with_capacity(channels.len());

        for (plane_index, channel) in channels.iter().enumerate() {
            let Some(source) = frame.plane(*channel) else {
                return Err(missing_channel(self.stage, *channel));
            };

            if !self.selected[plane_index] {
                planes.push(source.clone());
                continue;
            }

            let filtered = self.filter_plane(client, context, source, plane_index, &meta, filter)?;
            planes.push(filtered);
        }

        match Frame::from_planes(self.format, planes, meta) {
            Ok(output) => Ok(output),
            Err(error) => Err(bad_frame(self.stage, error)),
        }
    }

    fn filter_plane<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        context: &FrameContext,
        source: &Plane,
        plane_index: usize,
        meta: &FrameMeta,
        filter: &impl PlaneFilter,
    ) -> Result<Plane, StageError> {
        let (width, height) = source.dimensions();
        let sizing = Sizing(context);
        let plane_format = FrameFormat {
            width,
            height,
            layout: FrameLayout::Grey,
            ..self.format
        };
        let target = context.alloc_frame(&plane_format, meta.clone());
        let Some(output) = target.plane(Channel::Y) else {
            return Err(missing_channel(self.stage, Channel::Y));
        };

        let input_values = Scratch::new::<f32>(context, width, height);
        let output_values = Scratch::new::<f32>(context, width, height);
        let input_plane = input_values.plane();
        let output_plane = output_values.plane();

        let source_handle = source.buffer().handle();
        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, width, height) };

        widen(client, &sizing, source_plane, input_plane, self.element);
        filter.filter(context, plane_index, input_plane, output_plane)?;
        narrow(
            client,
            &sizing,
            output_plane,
            target_plane,
            self.element,
            self.peak,
        );

        Ok(output.clone())
    }
}
