mod float;

use av_toolbox_core::{
    Filter,
    InputArgument,
    Inputs,
    PipelineRegistry,
    PreparedStage,
    StageHandle,
    StageKind,
    StreamType,
};
use av_toolbox_cubek::Dispatch;
use av_toolbox_cubek::depth::Element;
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    ColourInfo,
    CubeCount,
    CubeDim,
    Depth,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    MaskChroma,
    MaskFormat,
    PlaneBuffer,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
    StreamTemplate,
    Subsampling,
    test_device,
};
use cubecl::prelude::Runtime;

pub(crate) use self::float::{
    ROUND_TRIP,
    assert_a_flat_frame_is_unchanged,
    assert_codes_within,
    assert_planes_match,
    f32_parity,
    float_bytes,
    float_planes,
    float_samples,
    kept_sample_parity,
    run_into_codes,
    run_masked_into_codes,
};

pub(crate) fn backend() -> Backend {
    Backend::open_default(test_device()).expect("a backend should open")
}

/// Sizes launches the way a stage would, from a frame context.
pub(crate) struct Sizing(pub(crate) FrameContext);

impl Dispatch for Sizing {
    fn cube_count(&self, samples: usize) -> CubeCount {
        self.0.cube_count(samples)
    }

    fn cube_dim(&self) -> CubeDim {
        self.0.cube_dim()
    }
}

pub(crate) fn frame(
    width: u32,
    height: u32,
    layout: FrameLayout,
    sample: SampleFormat,
    location: ChromaLocation,
) -> FrameFormat {
    let colour = ColourInfo {
        chroma_location: location,
        ..ColourInfo::default()
    };

    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour,
        frame_rate: None,
        field_order: None,
    }
}

/// Builds `stage` the way a pipeline would, through the registry and a real backend.
pub(crate) fn built_filter(
    backend: &Backend,
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
) -> Box<dyn Filter> {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let handle = StageHandle::new(1);
    let arguments = [InputArgument::positional(handle, StreamType::Frames)];
    let prepared = registry
        .prepare(StageKind::Filter, stage, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let input_format = StreamFormat::Frames(input);
    let output_format = StreamFormat::Frames(output);
    let formats = ResolvedFormats::new(vec![Some(input_format)], Some(output_format));

    prepared
        .build(backend, &formats)
        .expect("should build")
        .into_device()
        .expect("a device filter")
}

/// Runs one frame through a filter on the device and reads every output plane back as bytes.
struct FilterOnce {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
    planes: Vec<Vec<u8>>,
    index: u64,
}

impl BuildTask for FilterOnce {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(self.index);
        let mut frame = host.alloc_frame(&self.input, meta);
        for (index, channel) in self.input.channels().iter().enumerate() {
            let plane = frame
                .plane_mut(*channel)
                .expect("the format carries this channel");
            plane.copy_from_slice(&self.planes[index]);
        }

        let frozen = frame.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).expect("the item should load");
        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let emitted = self.filter.process(&mut context, inputs).expect("should process");
        let processed = only_frame(emitted);

        assert_eq!(processed.format(), &self.output);

        self.output
            .channels()
            .iter()
            .map(|channel| {
                let plane = processed
                    .plane(*channel)
                    .expect("the output carries this channel");
                let handle = plane.buffer().handle().clone();
                let bytes = build
                    .client()
                    .read_one(handle)
                    .expect("the device should hand back the plane");
                bytes.to_vec()
            })
            .collect()
    }
}

/// Runs `stage` on one frame and reads every output plane back as bytes.
pub(crate) fn run_once(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
) -> Vec<Vec<u8>> {
    run_once_at(stage, input, output, args, planes, 0)
}

/// Runs `stage` on one frame at `index` in the stream, and reads every output plane back as bytes.
pub(crate) fn run_once_at(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    index: u64,
) -> Vec<Vec<u8>> {
    let backend = backend();
    let filter = built_filter(&backend, stage, input, output, args);

    backend.build(FilterOnce {
        filter,
        input,
        output,
        planes: planes.to_vec(),
        index,
    })
}

/// Runs one frame through `stage` and reports whether the output's first plane is the input's,
/// rather than a fresh allocation.
struct IdentityCheck {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
}

impl BuildTask for IdentityCheck {
    type Output = bool;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
        let frame = host.alloc_frame(&self.input, meta);
        let frozen = frame.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).expect("the item should load");

        let first_channel = self.input.channels()[0];
        let StreamItem::Frame(loaded_frame) = &loaded else {
            panic!("expected a frame");
        };
        let input_plane = loaded_frame
            .plane(first_channel)
            .expect("the input carries this channel");
        let input_buffer: *const PlaneBuffer = input_plane.buffer();

        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let processed = self.filter.process(&mut context, inputs).expect("should process");
        let output = only_frame(processed);

        assert_eq!(output.format(), &self.output);

        let output_plane = output
            .plane(first_channel)
            .expect("the output carries this channel");
        let output_buffer: *const PlaneBuffer = output_plane.buffer();

        std::ptr::eq(input_buffer, output_buffer)
    }
}

/// Whether `stage` hands back the input frame unchanged, rather than a new allocation.
pub(crate) fn passes_through(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
) -> bool {
    let backend = backend();
    let filter = built_filter(&backend, stage, input, output, args);

    backend.build(IdentityCheck {
        filter,
        input,
        output,
    })
}

/// Every plane of `format`, each sample `(index * 37 + plane * 11) % 256`.
pub(crate) fn patterned_planes(format: &FrameFormat) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples)
                .map(|index| ((index * 37 + plane * 11) % 256) as u8)
                .collect()
        })
        .collect()
}

/// Checks every sample of `found` is within 1 of `expected`, naming `label` in a failure.
pub(crate) fn assert_within_one(found: &[u8], expected: &[u8], label: &str) {
    assert_eq!(found.len(), expected.len(), "{label}");

    for (index, (got, want)) in found.iter().zip(expected).enumerate() {
        assert!(
            got.abs_diff(*want) <= 1,
            "{label}: sample {index} was {got}, expected {want}"
        );
    }
}

/// Runs one frame through `filter` and reports whether the output's Y plane is the input's, rather
/// than a fresh allocation.
struct LumaSharedCheck {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
}

impl BuildTask for LumaSharedCheck {
    type Output = bool;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
        let frame = host.alloc_frame(&self.input, meta);
        let frozen = frame.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).expect("the item should load");

        let StreamItem::Frame(loaded_frame) = &loaded else {
            panic!("expected a frame");
        };
        let input_plane = loaded_frame
            .plane(Channel::Y)
            .expect("the input carries a Y plane");
        let input_buffer: *const PlaneBuffer = input_plane.buffer();

        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let processed = self.filter.process(&mut context, inputs).expect("should process");
        let output = only_frame(processed);

        assert_eq!(output.format(), &self.output);

        let output_plane = output.plane(Channel::Y).expect("the output carries a Y plane");
        let output_buffer: *const PlaneBuffer = output_plane.buffer();

        std::ptr::eq(input_buffer, output_buffer)
    }
}

/// Whether `stage` hands the input's Y plane straight through, rather than copying it.
pub(crate) fn luma_is_shared(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
) -> bool {
    let backend = backend();
    let filter = built_filter(&backend, stage, input, output, args);

    backend.build(LumaSharedCheck {
        filter,
        input,
        output,
    })
}

/// Runs one frame through `stage` and reports whether every one of the input's planes is shared
/// straight through to the output, rather than a fresh allocation.
struct AllPlanesSharedCheck {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
}

impl BuildTask for AllPlanesSharedCheck {
    type Output = bool;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
        let frame = host.alloc_frame(&self.input, meta);
        let frozen = frame.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).expect("the item should load");

        let StreamItem::Frame(loaded_frame) = &loaded else {
            panic!("expected a frame");
        };
        let input_buffers: Vec<*const PlaneBuffer> = self
            .input
            .channels()
            .iter()
            .map(|channel| {
                let plane = loaded_frame
                    .plane(*channel)
                    .expect("the input carries this channel");
                plane.buffer() as *const PlaneBuffer
            })
            .collect();

        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let processed = self.filter.process(&mut context, inputs).expect("should process");
        let output = only_frame(processed);

        assert_eq!(output.format(), &self.output);

        self.output
            .channels()
            .iter()
            .zip(&input_buffers)
            .all(|(channel, input_buffer)| {
                let plane = output.plane(*channel).expect("the output carries this channel");
                let output_buffer: *const PlaneBuffer = plane.buffer();

                std::ptr::eq(*input_buffer, output_buffer)
            })
    }
}

/// Whether `stage` hands every one of the input's planes straight through to the output.
pub(crate) fn every_plane_is_shared(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
) -> bool {
    let backend = backend();
    let filter = built_filter(&backend, stage, input, output, args);

    backend.build(AllPlanesSharedCheck {
        filter,
        input,
        output,
    })
}

/// A clip read back from the device, as its format and every plane's bytes.
pub(crate) struct DeviceOutput {
    pub(crate) format: FrameFormat,
    pub(crate) planes: Vec<Vec<u8>>,
}

/// The one frame a filter emitted for one input.
fn only_frame(mut frames: Vec<Frame>) -> Frame {
    assert_eq!(
        frames.len(),
        1,
        "a filter emitted {} frames for one input",
        frames.len()
    );
    frames.remove(0)
}

/// Builds `stage` against one positional input per entry of `inputs`, working out its output
/// format the way a pipeline would, by asking the stage what it produces.
pub(crate) fn built_multi_filter(
    backend: &Backend,
    stage: &str,
    inputs: &[FrameFormat],
    args: serde_json::Value,
) -> (Box<dyn Filter>, FrameFormat) {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let arguments: Vec<InputArgument> = inputs
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let handle = StageHandle::new(index as u64 + 1);
            InputArgument::positional(handle, StreamType::Frames)
        })
        .collect();

    let prepared = registry
        .prepare(StageKind::Filter, stage, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let chosen: Vec<Option<StreamFormat>> = inputs
        .iter()
        .map(|format| Some(StreamFormat::Frames(*format)))
        .collect();

    let produced = prepared
        .produces(backend.capabilities(), &chosen)
        .expect("should produce a format");
    let StreamTemplate::Frames(output) = produced.template else {
        panic!("expected a frame template");
    };

    let resolved = ResolvedFormats::new(chosen, Some(StreamFormat::Frames(output)));
    let filter = prepared
        .build(backend, &resolved)
        .expect("should build")
        .into_device()
        .expect("a device filter");

    (filter, output)
}

/// Runs one frame through a filter with several inputs, and reads every output plane back.
struct MultiFilterOnce {
    filter: Box<dyn Filter>,
    inputs: Vec<FrameFormat>,
    input_planes: Vec<Vec<Vec<u8>>>,
    output: FrameFormat,
}

impl BuildTask for MultiFilterOnce {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let mut slots: Vec<Option<StreamItem>> = Vec::with_capacity(self.inputs.len());
        for (input_index, input_format) in self.inputs.iter().enumerate() {
            let meta = FrameMeta::at(0);
            let mut frame = host.alloc_frame(input_format, meta);
            for (channel_index, channel) in input_format.channels().iter().enumerate() {
                let plane = frame
                    .plane_mut(*channel)
                    .expect("the format carries this channel");
                plane.copy_from_slice(&self.input_planes[input_index][channel_index]);
            }

            let frozen = frame.freeze();
            let item = HostStreamItem::Frame(frozen);
            let loaded = context.load_host(&item).expect("the item should load");
            slots.push(Some(loaded));
        }

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let emitted = self.filter.process(&mut context, inputs).expect("should process");
        let processed = only_frame(emitted);

        assert_eq!(processed.format(), &self.output);

        self.output
            .channels()
            .iter()
            .map(|channel| {
                let plane = processed
                    .plane(*channel)
                    .expect("the output carries this channel");
                let handle = plane.buffer().handle().clone();
                let bytes = build
                    .client()
                    .read_one(handle)
                    .expect("the device should hand back the plane");
                bytes.to_vec()
            })
            .collect()
    }
}

/// Runs `stage` on one frame built from `input` and `planes`, and reads its output back.
pub(crate) fn run_filter_on_frame(
    stage: &str,
    args: serde_json::Value,
    input: &FrameFormat,
    planes: &[Vec<u8>],
) -> DeviceOutput {
    let backend = backend();
    let (filter, output) = built_multi_filter(&backend, stage, std::slice::from_ref(input), args);

    let planes = backend.build(MultiFilterOnce {
        filter,
        inputs: vec![*input],
        input_planes: vec![planes.to_vec()],
        output,
    });

    DeviceOutput {
        format: output,
        planes,
    }
}

/// Runs `stage` on the frame `input` carries, chaining one device output into the next stage.
pub(crate) fn run_filter_on_output(
    stage: &str,
    args: serde_json::Value,
    input: &DeviceOutput,
) -> DeviceOutput {
    run_filter_on_frame(stage, args, &input.format, &input.planes)
}

/// Runs `planes/merge` on one positional input per entry of `inputs`, and reads its output back.
pub(crate) fn merge_frames(args: serde_json::Value, inputs: &[DeviceOutput]) -> DeviceOutput {
    let backend = backend();
    let formats: Vec<FrameFormat> = inputs.iter().map(|output| output.format).collect();
    let (filter, output) = built_multi_filter(&backend, "planes/merge", &formats, args);

    let input_planes: Vec<Vec<Vec<u8>>> = inputs.iter().map(|output| output.planes.clone()).collect();

    let planes = backend.build(MultiFilterOnce {
        filter,
        inputs: formats,
        input_planes,
        output,
    });

    DeviceOutput {
        format: output,
        planes,
    }
}

pub(crate) fn to_bytes(values: &[u16], element: Element) -> Vec<u8> {
    match element {
        Element::U8 => values.iter().map(|value| *value as u8).collect(),
        Element::U16 => values.iter().flat_map(|value| value.to_le_bytes()).collect(),
    }
}

pub(crate) fn from_bytes(bytes: &[u8], element: Element) -> Vec<u16> {
    match element {
        Element::U8 => bytes.iter().map(|value| u16::from(*value)).collect(),
        Element::U16 => {
            let (pairs, _) = bytes.as_chunks::<2>();
            pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
        },
    }
}

fn mask_chroma_for(layout: FrameLayout) -> Option<MaskChroma> {
    let FrameLayout::Yuv { subsampling } = layout else {
        return None;
    };

    if subsampling == Subsampling::Yuv444 {
        return None;
    }

    Some(MaskChroma {
        subsampling,
        location: ChromaLocation::Left,
    })
}

pub(crate) fn mask_format(format: &FrameFormat) -> MaskFormat {
    MaskFormat {
        frame_rate: None,
        width: format.width,
        height: format.height,
        sample: SampleFormat::U8,
        chroma: mask_chroma_for(format.layout),
    }
}

/// Builds `stage` with a bound mask, the way a pipeline would.
pub(crate) fn built_masked(
    backend: &Backend,
    stage: &str,
    format: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
) -> Box<dyn Filter> {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let input_handle = StageHandle::new(1);
    let mask_handle = StageHandle::new(2);
    let arguments = [
        InputArgument::positional(input_handle, StreamType::Frames),
        InputArgument::positional(mask_handle, StreamType::Mask),
    ];
    let prepared = registry
        .prepare(StageKind::Filter, stage, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let frames = StreamFormat::Frames(format);
    let mask = StreamFormat::Mask(mask_format(&format));
    let output_format = StreamFormat::Frames(output);
    let formats = ResolvedFormats::new(vec![Some(frames), Some(mask)], Some(output_format));

    prepared
        .build(backend, &formats)
        .expect("should build")
        .into_device()
        .expect("a device filter")
}

/// Runs one frame and its mask through a filter and reads every output plane back.
struct MaskedOnce {
    filter: Box<dyn Filter>,
    format: FrameFormat,
    output: FrameFormat,
    planes: Vec<Vec<u8>>,
    mask: Vec<u8>,
    mask_chroma: Vec<u8>,
}

impl BuildTask for MaskedOnce {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
        let mut input = host.alloc_frame(&self.format, meta);
        for (index, channel) in self.format.channels().iter().enumerate() {
            let plane = input
                .plane_mut(*channel)
                .expect("the format carries this channel");
            plane.copy_from_slice(&self.planes[index]);
        }

        let mask_meta = FrameMeta::at(0);
        let layout = mask_format(&self.format);
        let mut mask = host.alloc_mask(&layout, mask_meta);
        mask.plane_mut().copy_from_slice(&self.mask);
        let chroma_plane = mask.chroma_plane_mut().expect("the mask carries a chroma plane");
        chroma_plane.copy_from_slice(&self.mask_chroma);

        let input_frame = input.freeze();
        let mask_frame = mask.freeze();
        let items = [
            HostStreamItem::Frame(input_frame),
            HostStreamItem::Mask(mask_frame),
        ];
        let slots: Vec<Option<StreamItem>> = items
            .iter()
            .map(|item| Some(context.load_host(item).expect("the item should load")))
            .collect();

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let processed = self.filter.process(&mut context, inputs).expect("should process");
        let output = only_frame(processed);

        assert_eq!(output.format(), &self.output);

        self.output
            .channels()
            .iter()
            .map(|channel| {
                let plane = output.plane(*channel).expect("the output carries this channel");
                let handle = plane.buffer().handle().clone();
                let bytes = build
                    .client()
                    .read_one(handle)
                    .expect("the device should hand back the plane");
                bytes.to_vec()
            })
            .collect()
    }
}

/// Runs `stage` on `planes` into `output` frames, with a mask of the full and chroma bytes in `mask`.
pub(crate) fn run_masked_into(
    stage: &str,
    format: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    planes: &[Vec<u8>],
    mask: (Vec<u8>, Vec<u8>),
) -> Vec<Vec<u8>> {
    let backend = backend();
    let filter = built_masked(&backend, stage, format, output, args);
    let (mask, mask_chroma) = mask;

    backend.build(MaskedOnce {
        filter,
        format,
        output,
        planes: planes.to_vec(),
        mask,
        mask_chroma,
    })
}

/// A mask of `value` over the frame and its chroma plane.
pub(crate) fn uniform_mask(format: &FrameFormat, value: u8) -> (Vec<u8>, Vec<u8>) {
    let chroma = mask_format(format)
        .chroma_dimensions()
        .expect("a subsampled mask");
    let full = vec![value; (format.width * format.height) as usize];
    let small = vec![value; (chroma.0 * chroma.1) as usize];

    (full, small)
}

/// A mask of 255 over the left half of the frame and 0 over the right, and the same for its chroma
/// plane.
pub(crate) fn split_mask(format: &FrameFormat) -> (Vec<u8>, Vec<u8>) {
    let chroma = mask_format(format)
        .chroma_dimensions()
        .expect("a subsampled mask");
    let half = |width: u32, height: u32| -> Vec<u8> {
        (0..width * height)
            .map(|index| if index % width < width / 2 { 255 } else { 0 })
            .collect()
    };

    (half(format.width, format.height), half(chroma.0, chroma.1))
}

/// Runs one frame through a filter and reports, per plane, whether the output shares the input's
/// buffer.
struct SharedPlanes {
    filter: Box<dyn Filter>,
    format: FrameFormat,
}

impl BuildTask for SharedPlanes {
    type Output = Vec<bool>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
        let input = host.alloc_frame(&self.format, meta);
        let frozen = input.freeze();
        let item = HostStreamItem::Frame(frozen);
        let loaded = context.load_host(&item).expect("the item should load");

        let StreamItem::Frame(loaded_frame) = &loaded else {
            panic!("expected a frame");
        };
        let input_buffers: Vec<*const PlaneBuffer> = self
            .format
            .channels()
            .iter()
            .map(|channel| {
                let plane = loaded_frame
                    .plane(*channel)
                    .expect("the input carries this channel");
                plane.buffer() as *const PlaneBuffer
            })
            .collect();

        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let processed = self.filter.process(&mut context, inputs).expect("should process");
        let output = only_frame(processed);

        self.format
            .channels()
            .iter()
            .zip(&input_buffers)
            .map(|(channel, input_buffer)| {
                let plane = output.plane(*channel).expect("the output carries this channel");
                let output_buffer: *const PlaneBuffer = plane.buffer();

                std::ptr::eq(*input_buffer, output_buffer)
            })
            .collect()
    }
}

/// Builds `stage` without a mask and reports which planes it shares with the input.
pub(crate) fn shared_planes(stage: &str, format: FrameFormat, args: serde_json::Value) -> Vec<bool> {
    let backend = backend();
    let filter = built_filter(&backend, stage, format, format, args);

    backend.build(SharedPlanes { filter, format })
}

pub(crate) fn ten_bit() -> SampleFormat {
    SampleFormat::U16 { depth: Depth::TEN }
}

pub(crate) fn element_of(sample: SampleFormat) -> Element {
    match sample {
        SampleFormat::U8 => Element::U8,
        _ => Element::U16,
    }
}

pub(crate) fn peak_of(sample: SampleFormat) -> u32 {
    match sample {
        SampleFormat::U8 => 255,
        _ => 1023,
    }
}

/// Code values as f32 samples.
pub(crate) fn widened(values: &[u16]) -> Vec<f32> {
    values.iter().map(|value| f32::from(*value)).collect()
}

pub(crate) fn encoded(format: &FrameFormat, values: &[Vec<u16>]) -> Vec<Vec<u8>> {
    let element = element_of(format.sample);

    values.iter().map(|plane| to_bytes(plane, element)).collect()
}
