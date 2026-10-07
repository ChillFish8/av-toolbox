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
};
use av_toolbox_cubek::DevicePlane;
#[cfg(test)]
use av_toolbox_cubek::artcnn::{BodyPath, CnnError};
use av_toolbox_cubek::depth::Element;
use av_toolbox_cubek::resize::{Kernel, Samples};
use av_toolbox_cubek::values::{narrow, widen};
use av_toolbox_gpu::{
    BuildContext,
    Capabilities,
    Channel,
    ComputeClient,
    DeviceSide,
    FormatSet,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    Offered,
    OutputFormats,
    Plane,
    ResolvedFormats,
    Runtime,
    SampleFormat,
    StreamFormat,
    StreamTemplate,
};
use av_toolbox_models::{Located, Store};
use schemars::JsonSchema;
use serde::Deserialize;

pub(crate) use crate::artcnn::ModelName;
use crate::artcnn::Supersampler;
#[cfg(test)]
pub(crate) use crate::artcnn::{KERNEL_ENV, frame_tile, kernels_from_value};
use crate::bits::bits_of;
pub(crate) use crate::cnn::PrecisionName;
use crate::cnn::checked_precision;
#[cfg(test)]
pub(crate) use crate::cnn::{allocation_cap, body_path, fallbacks, warmed_path};
use crate::resize::{Tables, Window, geometry, resize_plane, samples_for};
use crate::scratch::Scratch;
use crate::sizing::Sizing;

const STAGE: &str = "upscale/artcnn";

/// Catrom, the bicubic the chroma planes are resized with.
const CATROM: Kernel = Kernel::Bicubic { b: 0.0, c: 0.5 };

/// Arguments for `upscale/artcnn`.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ArtcnnArgs {
    /// The network: `r8f64`, or the larger and slower `r16f96`.
    #[serde(default)]
    pub(crate) model: ModelName,

    /// The feature maps' precision: `f16`, or the slower and more exact `f32`.
    #[serde(default)]
    pub(crate) precision: PrecisionName,
}

fn upscalable_layout(layout: FrameLayout) -> bool {
    matches!(layout, FrameLayout::Grey | FrameLayout::Yuv { .. })
}

fn upscalable_sample(sample: SampleFormat) -> bool {
    matches!(sample, SampleFormat::U8 | SampleFormat::U16 { .. })
}

fn missing_input() -> StageError {
    let message = format!("{STAGE} needs its input bound");

    StageError::msg(message)
}

fn not_frames() -> StageError {
    let message = format!("{STAGE} upscales frames, not masks");

    StageError::msg(message)
}

fn unusable_layout(layout: FrameLayout) -> StageError {
    let message = format!("{STAGE} upscales grey or yuv frames, not {layout} frames");

    StageError::msg(message)
}

fn unusable_sample(sample: SampleFormat) -> StageError {
    let message = format!("{STAGE} upscales U8 or U16 frames, not {sample} frames");

    StageError::msg(message)
}

fn missing_frame() -> StageError {
    let message = format!("{STAGE} expected a frame in slot 0");

    StageError::msg(message)
}

fn missing_channel(channel: Channel) -> StageError {
    let message = format!("{STAGE} was handed a frame missing {channel}");

    StageError::msg(message)
}

/// Finds `name`'s weight file in `store`, or the copy built into the binary.
pub(crate) fn locate(store: &Store, name: ModelName) -> Result<Located, StageError> {
    crate::artcnn::locate(STAGE, store, name)
}

#[cfg(test)]
pub(crate) fn failed_convolution_and_im2col(convolution: CnnError, im2col: CnnError) -> StageError {
    let failures = vec![(BodyPath::Convolution, convolution), (BodyPath::Im2col, im2col)];

    crate::cnn::failed_every_path(STAGE, failures)
}

/// The input format, once it is known to be U8 or U16 frames in a grey or YUV layout.
pub(crate) fn checked_input(chosen: &[Option<StreamFormat>]) -> Result<FrameFormat, StageError> {
    let Some(input) = chosen.first().copied().flatten() else {
        return Err(missing_input());
    };

    let Some(format) = input.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Err(unusable_sample(format.sample));
    }

    Ok(*format)
}

/// The samples accepted from `offered`, refusing masks and layouts the stage cannot upscale.
fn accepted_samples(offered: StreamFormat) -> Result<FormatSet, StageError> {
    let Some(format) = offered.as_frames() else {
        return Err(not_frames());
    };

    if !upscalable_layout(format.layout) {
        return Err(unusable_layout(format.layout));
    }

    if !upscalable_sample(format.sample) {
        return Ok(FormatSet::new());
    }

    Ok(FormatSet::of([format.sample]))
}

pub(crate) fn doubled(input: &FrameFormat) -> FrameFormat {
    FrameFormat {
        width: input.width * 2,
        height: input.height * 2,
        ..*input
    }
}

/// The tables that double a YUV input's chroma, keeping its siting. Grey has none.
///
/// 4:4:4 chroma is sited like luma, so it takes the luma plane's shape.
fn chroma_tables<R: Runtime>(client: &ComputeClient<R>, input: &FrameFormat) -> Option<Tables> {
    let FrameLayout::Yuv { .. } = input.layout else {
        return None;
    };

    let window = Window {
        width: input.width * 2,
        height: input.height * 2,
        src_left: 0.0,
        src_top: 0.0,
        src_width: None,
        src_height: None,
    };
    let shapes = geometry(input, &window);
    let shape = shapes.chroma.unwrap_or(shapes.luma);
    let tables = Tables::new(client, &shape, &CATROM);

    Some(tables)
}

/// Doubles a clip's size, running its luma through the ArtCNN network and its chroma through Catrom.
///
/// It runs as a single instance, since one frame's kernels already fill the device, and holds up to about
/// 1 GiB of device scratch.
pub(crate) struct Artcnn;

impl StageDefinition for Artcnn {
    type Args = ArtcnnArgs;

    fn package() -> &'static str {
        "upscale"
    }

    fn identifier() -> &'static str {
        "artcnn"
    }

    fn parallelism(_args: &Self::Args) -> Parallelism {
        Parallelism::Stream
    }

    fn inputs() -> &'static InputSignature {
        const SIGNATURE: &InputSignature =
            &InputSignature::new(&[InputParam::required("input").accepting(StreamType::Frames)]);
        SIGNATURE
    }
}

impl FilterDefinition for Artcnn {
    type Side = DeviceSide;

    fn accepts_input(
        _capabilities: &Capabilities,
        _args: &Self::Args,
        _index: usize,
        offered: StreamFormat,
        _inputs: &[Offered],
    ) -> Result<FormatSet, StageError> {
        accepted_samples(offered)
    }

    fn produces(
        capabilities: &Capabilities,
        args: &Self::Args,
        chosen: &[Option<StreamFormat>],
    ) -> Result<OutputFormats, StageError> {
        let input = checked_input(chosen)?;
        checked_precision(STAGE, capabilities, args.precision)?;

        let store = Store::from_environment();
        locate(&store, args.model)?;

        let output = doubled(&input);
        let format = StreamFormat::Frames(output);

        Ok(OutputFormats {
            template: StreamTemplate::from_format(&format),
            samples: FormatSet::of([output.sample]),
        })
    }

    fn build<R: Runtime>(
        context: &BuildContext<R>,
        args: Self::Args,
        _inputs: BoundInputs,
        formats: &ResolvedFormats,
    ) -> Result<Box<dyn Filter>, StageError> {
        let inputs = formats.inputs();
        let input = checked_input(inputs)?;
        let supersampler = Supersampler::load(
            STAGE,
            context,
            args.model,
            args.precision,
            input.width,
            input.height,
        )?;

        let client = context.client().clone();
        let chroma = chroma_tables(&client, &input);
        let bits = bits_of(input.sample);
        let element = if bits == 8 { Element::U8 } else { Element::U16 };
        let peak = ((1u32 << bits) - 1) as f32;
        let samples = samples_for(input.sample);

        let upscaler = Upscaler {
            client,
            supersampler,
            output: doubled(&input),
            chroma,
            element,
            peak,
            samples,
        };

        Ok(Box::new(upscaler))
    }
}

/// Runs `upscale/artcnn` on every frame, with the network and tables chosen at build.
struct Upscaler<R: Runtime> {
    client: ComputeClient<R>,
    supersampler: Supersampler,
    output: FrameFormat,
    chroma: Option<Tables>,
    element: Element,
    peak: f32,
    samples: Samples,
}

impl<R: Runtime> Upscaler<R> {
    /// Widens `input` to f32, runs the network over it, and narrows the doubled result into `output`.
    fn upscale_luma(&self, context: &FrameContext, input: &Plane, output: &Plane) -> Result<(), StageError> {
        let (width, height) = input.dimensions();
        let (doubled_width, doubled_height) = output.dimensions();
        let sizing = Sizing(context);

        let input_values = Scratch::new::<f32>(context, width, height);
        let output_values = Scratch::new::<f32>(context, doubled_width, doubled_height);
        let input_plane = input_values.plane();
        let output_plane = output_values.plane();

        let source_handle = input.buffer().handle();
        let target_handle = output.buffer().handle();

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let source_plane = unsafe { DevicePlane::new(source_handle, width, height) };

        // Safety: a frame's plane holds exactly `width * height` samples of its sample format.
        let target_plane = unsafe { DevicePlane::new(target_handle, doubled_width, doubled_height) };

        widen(&self.client, &sizing, source_plane, input_plane, self.element);

        self.supersampler
            .run(&self.client, context, input_plane, self.peak, output_plane)?;

        narrow(
            &self.client,
            &sizing,
            output_plane,
            target_plane,
            self.element,
            self.peak,
        );

        Ok(())
    }
}

impl<R: Runtime> StageInstance for Upscaler<R> {}

impl<R: Runtime> Filter for Upscaler<R> {
    fn process(&mut self, context: &mut FrameContext, mut inputs: Inputs) -> Result<Vec<Frame>, StageError> {
        let Some(input) = inputs.take_frame(0) else {
            return Err(missing_frame());
        };

        let meta = input.meta().clone();
        let output = context.alloc_frame(&self.output, meta);

        let Some(input_luma) = input.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        let Some(output_luma) = output.plane(Channel::Y) else {
            return Err(missing_channel(Channel::Y));
        };

        self.upscale_luma(context, input_luma, output_luma)?;

        let Some(tables) = &self.chroma else {
            return Ok(vec![output]);
        };

        for channel in [Channel::U, Channel::V] {
            let Some(input_plane) = input.plane(channel) else {
                return Err(missing_channel(channel));
            };

            let Some(output_plane) = output.plane(channel) else {
                return Err(missing_channel(channel));
            };

            resize_plane(
                &self.client,
                context,
                input_plane,
                output_plane,
                tables,
                self.samples,
            );
        }

        Ok(vec![output])
    }
}
