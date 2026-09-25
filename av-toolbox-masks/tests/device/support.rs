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
    Device,
    FrameContext,
    FrameMeta,
    HostPool,
    HostStreamItem,
    Mask as GpuMask,
    MaskFormat,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
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
        let inputs_after = held.iter().map(|mask| read_mask(build, mask)).collect();

        Selected { output, inputs_after }
    }
}
