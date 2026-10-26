use av_toolbox_core::{Filter, Inputs, StageError, StageInstance};
use av_toolbox_gpu::{
    Channel,
    ComputeClient,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Plane,
    SampleFormat,
};
// `#[cube]` expands into calls on traits the prelude carries, so it is imported whole rather than
// by name.
use cubecl::prelude::*;

use super::{Edges, cropped};

#[cube(launch_unchecked)]
fn crop_kernel<E: Int>(
    input: &Array<E>,
    output: &mut Array<E>,
    input_width: u32,
    output_width: u32,
    left: u32,
    top: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let position = ABSOLUTE_POS as u32;
    let column = position % output_width;
    let row = position / output_width;
    let source = (row + top) * input_width + column + left;

    output[ABSOLUTE_POS] = input[source as usize];
}

/// Where one plane's window starts, in that plane's own samples.
fn plane_offset(layout: FrameLayout, channel: Channel, edges: &Edges) -> (u32, u32) {
    let FrameLayout::Yuv { subsampling } = layout else {
        return (edges.left, edges.top);
    };

    if channel == Channel::Y {
        return (edges.left, edges.top);
    }

    let (factor_x, factor_y) = subsampling.factors();
    (edges.left / factor_x, edges.top / factor_y)
}

fn missing_input(stage: &str) -> StageError {
    let message = format!("{stage} expected a frame in slot 0");

    StageError::msg(message)
}

fn missing_channel(stage: &str) -> StageError {
    let message = format!("{stage} was handed a frame missing a channel");

    StageError::msg(message)
}

/// Copies one window out of every plane of a frame.
pub(crate) struct Cropper<R: Runtime> {
    client: ComputeClient<R>,
    stage: &'static str,
    input: FrameFormat,
    output: FrameFormat,
    edges: Edges,
}

impl<R: Runtime> Cropper<R> {
    pub(crate) fn new(
        client: ComputeClient<R>,
        stage: &'static str,
        input: FrameFormat,
        edges: Edges,
    ) -> Result<Self, StageError> {
        let output = cropped(&input, &edges, stage)?;

        Ok(Self {
            client,
            stage,
            input,
            output,
            edges,
        })
    }

    fn crop_plane(&self, context: &FrameContext, channel: Channel, input: &Plane, output: &Plane) {
        let (input_width, _) = input.dimensions();
        let (output_width, _) = output.dimensions();
        let (left, top) = plane_offset(self.input.layout, channel, &self.edges);
        let cubes = context.cube_count(output.samples());
        let shape = context.cube_dim();

        // Safety: both bindings name exactly as many elements as their plane holds, and every read
        // lands inside the input because the window was checked against the input's size.
        unsafe {
            match self.input.sample {
                SampleFormat::U8 => {
                    let input = input.array::<u8, R>();
                    let output = output.array::<u8, R>();
                    crop_kernel::launch_unchecked::<u8, R>(
                        &self.client,
                        cubes,
                        shape,
                        input,
                        output,
                        input_width,
                        output_width,
                        left,
                        top,
                    );
                },
                SampleFormat::U16 { .. } => {
                    let input = input.array::<u16, R>();
                    let output = output.array::<u16, R>();
                    crop_kernel::launch_unchecked::<u16, R>(
                        &self.client,
                        cubes,
                        shape,
                        input,
                        output,
                        input_width,
                        output_width,
                        left,
                        top,
                    );
                },
                SampleFormat::F32 => {
                    let input = input.array::<u32, R>();
                    let output = output.array::<u32, R>();
                    crop_kernel::launch_unchecked::<u32, R>(
                        &self.client,
                        cubes,
                        shape,
                        input,
                        output,
                        input_width,
                        output_width,
                        left,
                        top,
                    );
                },
            }
        }
    }
}

impl<R: Runtime> StageInstance for Cropper<R> {}

impl<R: Runtime> Filter for Cropper<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_input(self.stage));
        };

        if self.edges.is_zero() {
            return Ok(vec![input]);
        }

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);

        for channel in self.output.channels() {
            let Some(input_plane) = input.plane(*channel) else {
                return Err(missing_channel(self.stage));
            };

            let Some(output_plane) = output.plane(*channel) else {
                return Err(missing_channel(self.stage));
            };

            self.crop_plane(context, *channel, input_plane, output_plane);
        }

        Ok(vec![output])
    }
}
