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
    Attachments,
    Backend,
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    MaskChroma,
    MaskFormat,
    MetaValue,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
    StreamTemplate,
    Subsampling,
};

use crate::support::backend;

const STAGE: &str = "mask/to_frames";
const WIDTH: u32 = 16;
const HEIGHT: u32 = 8;

/// The one frame read back from a converted mask.
struct Converted {
    format: FrameFormat,
    values: Vec<f32>,
    meta: FrameMeta,
}

/// Builds the stage against one mask input, asking it what it produces.
fn built(backend: &Backend, mask: MaskFormat, args: serde_json::Value) -> (Box<dyn Filter>, FrameFormat) {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let handle = StageHandle::new(1);
    let arguments = [InputArgument::positional(handle, StreamType::Mask)];
    let prepared = registry
        .prepare(StageKind::Filter, STAGE, &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let chosen = vec![Some(StreamFormat::Mask(mask))];
    let produced = prepared
        .produces(backend.capabilities(), &chosen)
        .expect("should produce a format");
    let StreamTemplate::Frames(output) = produced.template else {
        panic!("expected a frame template");
    };

    let output_format = StreamFormat::Frames(output);
    let resolved = ResolvedFormats::new(chosen, Some(output_format));
    let filter = prepared
        .build(backend, &resolved)
        .expect("should build")
        .into_device()
        .expect("a device filter");

    (filter, output)
}

/// Converts one mask through the built stage.
struct ConvertOnce {
    filter: Box<dyn Filter>,
    mask: MaskFormat,
    full: Vec<u8>,
    chroma: Vec<u8>,
    meta: FrameMeta,
    output: FrameFormat,
}

impl BuildTask for ConvertOnce {
    type Output = Converted;

    fn run(mut self, build: &BuildContext) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let mut host_mask = host.alloc_mask(&self.mask, self.meta);
        host_mask.plane_mut().copy_from_slice(&self.full);
        if let Some(chroma_plane) = host_mask.chroma_plane_mut() {
            chroma_plane.copy_from_slice(&self.chroma);
        }

        let frozen = host_mask.freeze();
        let item = HostStreamItem::Mask(frozen);
        let loaded = context.load_host(&item).expect("the mask should load");
        let slots: Vec<Option<StreamItem>> = vec![Some(loaded)];

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(slots);
        let mut emitted = self.filter.process(&mut context, inputs).expect("should process");

        assert_eq!(emitted.len(), 1, "one mask makes one frame");

        let frame = emitted.remove(0);

        assert_eq!(frame.format(), &self.output);

        let plane = frame.plane(Channel::Y).expect("a grey frame has a Y plane");
        let handle = plane.buffer().handle().clone();
        let bytes = build
            .client()
            .read_one(handle)
            .expect("the device should hand back the plane");
        let (words, _) = bytes.as_chunks::<4>();
        let values = words.iter().map(|word| f32::from_le_bytes(*word)).collect();

        Converted {
            format: *frame.format(),
            values,
            meta: frame.meta().clone(),
        }
    }
}

fn mask_format(chroma: Option<MaskChroma>) -> MaskFormat {
    MaskFormat {
        frame_rate: None,
        width: WIDTH,
        height: HEIGHT,
        sample: SampleFormat::U8,
        chroma,
    }
}

fn chroma_420() -> Option<MaskChroma> {
    Some(MaskChroma {
        subsampling: Subsampling::Yuv420,
        location: ChromaLocation::Left,
    })
}

fn repeating(samples: usize) -> Vec<u8> {
    [0, 51, 128, 255].into_iter().cycle().take(samples).collect()
}

/// A chroma plane with a different value at every sample.
fn chroma_values(samples: usize) -> Vec<u8> {
    (0..samples).map(|index| (index * 7 % 256) as u8).collect()
}

fn converted(mask: MaskFormat, args: serde_json::Value, meta: FrameMeta) -> Converted {
    let backend = backend();
    let (filter, output) = built(&backend, mask, args);

    let full = repeating(mask.samples());
    let chroma_samples = mask
        .chroma_dimensions()
        .map_or(0, |(width, height)| (width * height) as usize);
    let chroma = chroma_values(chroma_samples);

    backend.build(ConvertOnce {
        filter,
        mask,
        full,
        chroma,
        meta,
        output,
    })
}

fn assert_shares(found: &[f32], codes: &[u8]) {
    assert_eq!(found.len(), codes.len());

    for (index, (value, code)) in found.iter().zip(codes).enumerate() {
        let expected = f32::from(*code) / 255.0;
        assert!(
            (value - expected).abs() <= 1e-6,
            "sample {index} was {value}, expected {expected}",
        );
    }
}

fn float_named(attachments: &Attachments, name: &str) -> f64 {
    let found = attachments
        .stages()
        .flat_map(|(_, values)| values.iter())
        .find(|(key, _)| key == name);

    match found {
        Some((_, MetaValue::Float(value))) => *value,
        other => panic!("expected a float named {name}, found {other:?}"),
    }
}

#[test]
fn mask_values_become_their_share_of_255() {
    let mask = mask_format(None);
    let args = serde_json::json!({});
    let output = converted(mask, args, FrameMeta::at(0));

    let codes = repeating(mask.samples());

    assert_eq!(output.values[0], 0.0);
    assert!((output.values[1] - 0.2).abs() <= 1e-6);
    assert_eq!(output.values[3], 1.0);
    assert_shares(&output.values, &codes);
}

#[test]
fn the_mask_metadata_is_kept() {
    let mask = mask_format(None);
    let mut meta = FrameMeta::at(0);
    meta.attach("score", MetaValue::Float(1.5));

    let args = serde_json::json!({});
    let output = converted(mask, args, meta);

    let mut attachments = output.meta.attachments.clone();
    attachments.stamp("mask/to_frames");
    let score = float_named(&attachments, "score");

    assert_eq!(score, 1.5);
}

#[test]
fn a_mask_chroma_plane_is_dropped() {
    let mask = mask_format(chroma_420());
    let args = serde_json::json!({});
    let output = converted(mask, args, FrameMeta::at(0));

    let codes = repeating(mask.samples());

    assert_eq!(output.format.layout, FrameLayout::Grey);
    assert_eq!((output.format.width, output.format.height), (WIDTH, HEIGHT));
    assert_shares(&output.values, &codes);
}
