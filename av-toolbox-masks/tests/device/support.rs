use av_toolbox_core::{
    InputArgument,
    Inputs,
    Mask,
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
    ColourInfo,
    Depth,
    Device,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    Mask as GpuMask,
    MaskFormat,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
    Subsampling,
};
use cubecl::prelude::Runtime;
use serde_json::Value;

pub fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
}

pub fn mask_format(width: u32, height: u32) -> MaskFormat {
    MaskFormat {
        width,
        height,
        sample: SampleFormat::U8,
        chroma: None,
    }
}

fn argument(position: usize, format: &StreamFormat) -> InputArgument {
    let stream = match format {
        StreamFormat::Frames(_) => StreamType::Frames,
        StreamFormat::Mask(_) => StreamType::Mask,
    };
    let handle = StageHandle::new(position as u64 + 1);

    InputArgument::positional(handle, stream)
}

/// Builds a registered mask the way a pipeline would, through the registry and a real backend.
pub fn built_mask(
    backend: &Backend,
    identifier: &str,
    args: Value,
    inputs: &[StreamFormat],
    output: MaskFormat,
) -> Box<dyn Mask> {
    let mut registry = PipelineRegistry::new();
    av_toolbox_masks::register(&mut registry).expect("the masks register");

    let arguments: Vec<InputArgument> = inputs
        .iter()
        .enumerate()
        .map(|(position, format)| argument(position, format))
        .collect();
    let prepared = registry
        .prepare(StageKind::Mask, identifier, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Mask(prepared) = prepared else {
        panic!("expected a mask");
    };

    let chosen: Vec<Option<StreamFormat>> = inputs.iter().copied().map(Some).collect();
    let produced = StreamFormat::Mask(output);
    let formats = ResolvedFormats::new(chosen, Some(produced));

    prepared
        .build(backend, &formats)
        .expect("should build")
        .into_device()
        .expect("a device mask")
}

pub fn upload_mask(context: &FrameContext, format: &MaskFormat, samples: &[u8]) -> GpuMask {
    let mut host = HostPool::new().alloc_mask(format, FrameMeta::at(0));
    host.plane_mut().copy_from_slice(samples);

    let item = HostStreamItem::Mask(host.freeze());
    let loaded = context.load_host(&item).expect("the mask should load");

    loaded.into_mask().expect("a mask loads as a mask")
}

pub fn upload_frame(context: &FrameContext, format: &FrameFormat, planes: &[Vec<u8>]) -> Frame {
    let mut host = HostPool::new().alloc_frame(format, FrameMeta::at(0));
    for (channel, samples) in format.channels().iter().zip(planes) {
        let target = host.plane_mut(*channel).expect("the frame carries this channel");
        target.copy_from_slice(samples);
    }

    let item = HostStreamItem::Frame(host.freeze());
    let loaded = context.load_host(&item).expect("the frame should load");

    loaded.into_frame().expect("a frame loads as a frame")
}

pub fn grey(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout: FrameLayout::Grey,
        sample: SampleFormat::U8,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

pub fn ten_bit_grey(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        sample: SampleFormat::U16 { depth: Depth::TEN },
        ..grey(width, height)
    }
}

pub fn yuv420(width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        layout: FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        },
        ..grey(width, height)
    }
}

pub fn read_mask<R: Runtime>(build: &BuildContext<R>, mask: &GpuMask) -> Vec<u8> {
    let handle = mask.plane().buffer().handle().clone();
    let bytes = build
        .client()
        .read_one(handle)
        .expect("the device should hand back the mask");

    bytes.to_vec()
}

/// What a mask selected, and what its inputs held once it was done with them.
pub struct Selected {
    pub output: Vec<u8>,
    pub output_format: MaskFormat,
    pub output_has_chroma_plane: bool,
    pub inputs_after: Vec<Vec<u8>>,
}

/// Runs a mask once over masks holding `inputs`, keeping its own handle on each input so a write
/// through one would show.
pub struct SelectMasks {
    pub mask: Box<dyn Mask>,
    pub format: MaskFormat,
    pub inputs: Vec<Vec<u8>>,
}

impl BuildTask for SelectMasks {
    type Output = Selected;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();

        let held: Vec<GpuMask> = self
            .inputs
            .iter()
            .map(|samples| upload_mask(&context, &self.format, samples))
            .collect();
        let slots = held
            .iter()
            .map(|mask| Some(StreamItem::Mask(mask.clone())))
            .collect();

        self.mask
            .start_of_stream(&mut context)
            .expect("should start the stream");
        let selected = self
            .mask
            .select(&mut context, Inputs::new(slots))
            .expect("should select");

        let output = read_mask(build, &selected);
        let output_format = *selected.format();
        let output_has_chroma_plane = selected.chroma_plane().is_some();
        let inputs_after = held.iter().map(|mask| read_mask(build, mask)).collect();

        Selected {
            output,
            output_format,
            output_has_chroma_plane,
            inputs_after,
        }
    }
}

/// Runs a mask over each frame in turn, one frame's planes per entry, handing back each mask.
pub struct SelectFrames {
    pub mask: Box<dyn Mask>,
    pub format: FrameFormat,
    pub frames: Vec<Vec<Vec<u8>>>,
}

impl BuildTask for SelectFrames {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();

        self.mask
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let mut selected = Vec::with_capacity(self.frames.len());
        for planes in &self.frames {
            let frame = upload_frame(&context, &self.format, planes);
            let inputs = Inputs::new(vec![Some(StreamItem::Frame(frame))]);
            let mask = self.mask.select(&mut context, inputs).expect("should select");

            let samples = read_mask(build, &mask);
            selected.push(samples);
        }

        selected
    }
}

/// Asserts two masks agree within `levels` everywhere but a small fraction of `outliers`.
///
/// Float blurs on the device and in a CPU reference do not match bit for bit, and a sample
/// sitting on a threshold can land either side of it.
pub fn assert_masks_close(actual: &[u8], expected: &[u8], levels: u8, outliers: f32) {
    assert_eq!(actual.len(), expected.len());

    let differing = actual
        .iter()
        .zip(expected)
        .filter(|(got, want)| got.abs_diff(**want) > levels)
        .count();
    let allowed = (expected.len() as f32 * outliers).ceil() as usize;

    assert!(
        differing <= allowed,
        "{differing} of {} samples differ by more than {levels}, allowing {allowed}",
        expected.len(),
    );
}

/// Little-endian bytes of `samples`, as a U16 plane stores them.
pub fn wide_bytes(samples: &[u16]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}
