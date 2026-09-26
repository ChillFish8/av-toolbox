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
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    ChromaLocation,
    ColourInfo,
    Device,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    PlaneBuffer,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
};
use cubecl::prelude::Runtime;

pub(crate) fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
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
}

impl BuildTask for FilterOnce {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let meta = FrameMeta::at(0);
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
        let processed = self.filter.process(&mut context, inputs).expect("should process");

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
    let backend = backend();
    let filter = built_filter(&backend, stage, input, output, args);

    backend.build(FilterOnce {
        filter,
        input,
        output,
        planes: planes.to_vec(),
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
        let output = self.filter.process(&mut context, inputs).expect("should process");

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
