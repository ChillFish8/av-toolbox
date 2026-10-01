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
use av_toolbox_cubek::resize::{Kernel, taps};
use av_toolbox_cubek::testing::reference_resize;
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    ColourInfo,
    Depth,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    MaskChroma,
    MaskFormat,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
    Subsampling,
    test_device,
};
use cubecl::prelude::Runtime;

fn backend() -> Backend {
    Backend::open_default(test_device()).expect("a backend should open")
}

/// The chroma a subsampled layout's mask carries in these tests, sited at `ChromaLocation::Left`.
///
/// `None` for a full-size layout, matching what a real graph resolves.
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

fn mask_format(format: &FrameFormat) -> MaskFormat {
    MaskFormat {
        width: format.width,
        height: format.height,
        sample: SampleFormat::U8,
        chroma: mask_chroma_for(format.layout),
    }
}

/// Builds `blend/masked` the way a pipeline would, through the registry and a real backend.
fn built_blend(backend: &Backend, format: FrameFormat, args: serde_json::Value) -> Box<dyn Filter> {
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let base_handle = StageHandle::new(1);
    let overlay_handle = StageHandle::new(2);
    let mask_handle = StageHandle::new(3);
    let arguments = [
        InputArgument::positional(base_handle, StreamType::Frames),
        InputArgument::positional(overlay_handle, StreamType::Frames),
        InputArgument::positional(mask_handle, StreamType::Mask),
    ];
    let prepared = registry
        .prepare(StageKind::Filter, "blend/masked", &args, &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let frames = StreamFormat::Frames(format);
    let mask_layout = mask_format(&format);
    let mask = StreamFormat::Mask(mask_layout);
    let formats = ResolvedFormats::new(vec![Some(frames), Some(frames), Some(mask)], Some(frames));

    prepared
        .build(backend, &formats)
        .expect("should build")
        .into_device()
        .expect("a device filter")
}

/// Blends two clips through a mask on the device and reads every output plane back.
struct BlendOnce {
    filter: Box<dyn Filter>,
    format: FrameFormat,
    base: Vec<Vec<u8>>,
    overlay: Vec<Vec<u8>>,
    mask: Vec<u8>,
    /// The mask's chroma plane, when its layout is subsampled.
    mask_chroma: Option<Vec<u8>>,
}

impl BuildTask for BlendOnce {
    type Output = Vec<Vec<u8>>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();

        let base_meta = FrameMeta::at(0);
        let mut base = host.alloc_frame(&self.format, base_meta);
        let overlay_meta = FrameMeta::at(0);
        let mut overlay = host.alloc_frame(&self.format, overlay_meta);
        for (index, channel) in self.format.channels().iter().enumerate() {
            let base_plane = base.plane_mut(*channel).expect("the format carries this channel");
            base_plane.copy_from_slice(&self.base[index]);

            let overlay_plane = overlay
                .plane_mut(*channel)
                .expect("the format carries this channel");
            overlay_plane.copy_from_slice(&self.overlay[index]);
        }

        let format = mask_format(&self.format);
        let mask_meta = FrameMeta::at(0);
        let mut mask = host.alloc_mask(&format, mask_meta);
        mask.plane_mut().copy_from_slice(&self.mask);
        if let Some(chroma_bytes) = &self.mask_chroma {
            let chroma_plane = mask
                .chroma_plane_mut()
                .expect("the format carries a chroma plane");
            chroma_plane.copy_from_slice(chroma_bytes);
        }

        let base_frame = base.freeze();
        let overlay_frame = overlay.freeze();
        let mask_frame = mask.freeze();
        let items = [
            HostStreamItem::Frame(base_frame),
            HostStreamItem::Frame(overlay_frame),
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
        let output = self.filter.process(&mut context, inputs).expect("should blend");

        self.format
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

/// Blends one plane on the CPU, per sample, as `base + (overlay - base) * m / 255`, rounded.
fn reference(base: &[u16], overlay: &[u16], mask: &[u8]) -> Vec<u16> {
    let mut blended = Vec::with_capacity(base.len());

    for index in 0..base.len() {
        let weight = f32::from(mask[index]) / 255.0;
        let low = f32::from(base[index]);
        let high = f32::from(overlay[index]);
        let value = (low + (high - low) * weight).round();
        blended.push(value as u16);
    }

    blended
}

fn narrow(samples: &[u8]) -> Vec<u16> {
    samples.iter().map(|sample| u16::from(*sample)).collect()
}

fn widen(samples: &[u16]) -> Vec<u8> {
    samples.iter().flat_map(|sample| sample.to_le_bytes()).collect()
}

fn unwiden(bytes: &[u8]) -> Vec<u16> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect()
}

fn frame(width: u32, height: u32, layout: FrameLayout, sample: SampleFormat) -> FrameFormat {
    FrameFormat {
        width,
        height,
        layout,
        sample,
        colour: ColourInfo::default(),
        frame_rate: None,
    }
}

fn ramp(samples: usize, step: usize, largest: usize) -> Vec<u16> {
    (0..samples)
        .map(|index| (index * step % (largest + 1)) as u16)
        .collect()
}

/// A ramp sized to each channel of `format`, offset by its index for variety between channels.
fn ramps_of(format: &FrameFormat, step: usize) -> Vec<Vec<u16>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            ramp(samples, step + index, 1023)
        })
        .collect()
}

/// The reference blend of one U8 plane, rounded back down to U8.
fn reference_u8(base: &[u8], overlay: &[u8], mask: &[u8]) -> Vec<u8> {
    let base_samples = narrow(base);
    let overlay_samples = narrow(overlay);
    let blended = reference(&base_samples, &overlay_samples, mask);

    blended.iter().map(|sample| *sample as u8).collect()
}

/// Blends U8 clips on the device and checks every plane against the reference.
fn check_u8(
    format: FrameFormat,
    base: Vec<Vec<u8>>,
    overlay: Vec<Vec<u8>>,
    mask: Vec<u8>,
    args: serde_json::Value,
) {
    let expected: Vec<Vec<u8>> = base
        .iter()
        .zip(overlay.iter())
        .map(|(base_plane, overlay_plane)| {
            let base_samples = narrow(base_plane);
            let overlay_samples = narrow(overlay_plane);
            let blended = reference(&base_samples, &overlay_samples, &mask);
            blended.iter().map(|sample| *sample as u8).collect()
        })
        .collect();

    let backend = backend();
    let filter = built_blend(&backend, format, args);
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base,
        overlay,
        mask,
        mask_chroma: None,
    });

    assert_eq!(planes, expected);
}

fn planes_of(format: &FrameFormat, fill: impl Fn(usize, usize) -> u8) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            (0..samples).map(|sample| fill(index, sample)).collect()
        })
        .collect()
}

/// Downscales a mask's full plane into its chroma plane on the CPU, with bilinear weights placed
/// by the chroma location and reflect-101 edges.
fn downscaled_chroma(full: &[u8], width: u32, height: u32, chroma: MaskChroma) -> Vec<u8> {
    let (factor_x, factor_y) = chroma.factors();
    let (offset_x, offset_y) = chroma.location.offsets(factor_x, factor_y);
    let (chroma_width, chroma_height) = chroma.subsampling.chroma_dimensions(width, height);

    let across = taps(chroma_width, factor_x as f32, offset_x, &Kernel::Bilinear);
    let down = taps(chroma_height, factor_y as f32, offset_y, &Kernel::Bilinear);

    reference_resize(full, width, height, &across, &down)
}

/// The mask plane `channel` blends with, matching [plane_for](av_toolbox_gpu::Mask::plane_for).
fn reference_mask_plane<'a>(
    format: &FrameFormat,
    channel: Channel,
    full: &'a [u8],
    chroma: &'a [u8],
) -> &'a [u8] {
    let full_size = (format.width, format.height);
    let dimensions = format
        .channel_dimensions(channel)
        .expect("a listed channel carries dimensions");

    if dimensions == full_size { full } else { chroma }
}

/// Blends U8 clips through a mask that carries a chroma plane, and checks every plane against the
/// reference, each read from the mask plane matching its channel.
fn check_u8_subsampled(
    format: FrameFormat,
    base: Vec<Vec<u8>>,
    overlay: Vec<Vec<u8>>,
    mask: Vec<u8>,
    mask_chroma: Vec<u8>,
    args: serde_json::Value,
) {
    let expected: Vec<Vec<u8>> = format
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let mask_plane = reference_mask_plane(&format, *channel, &mask, &mask_chroma);
            reference_u8(&base[index], &overlay[index], mask_plane)
        })
        .collect();

    let backend = backend();
    let filter = built_blend(&backend, format, args);
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base,
        overlay,
        mask,
        mask_chroma: Some(mask_chroma),
    });

    assert_eq!(planes, expected);
}

#[test]
fn a_444_frame_of_odd_size_blends_every_plane() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let format = frame(7, 5, layout, SampleFormat::U8);
    let base = planes_of(
        &format,
        |plane, sample| if plane == 0 { (sample * 7) as u8 } else { 100 },
    );
    let overlay = planes_of(
        &format,
        |plane, sample| if plane == 0 { 255 - (sample * 7) as u8 } else { 200 },
    );
    let mask: Vec<u8> = (0..35).map(|index| (index * 7 % 256) as u8).collect();
    let args = serde_json::json!({});

    check_u8(format, base, overlay, mask, args);
}

#[test]
fn a_444_ten_bit_frame_blends_at_its_depth() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let sample = SampleFormat::U16 { depth: Depth::TEN };
    let format = frame(8, 4, layout, sample);
    let base: Vec<Vec<u16>> = (0..3).map(|plane| ramp(32, 31 + plane, 1023)).collect();
    let overlay: Vec<Vec<u16>> = (0..3).map(|plane| ramp(32, 97 + plane, 1023)).collect();
    let mask: Vec<u8> = (0..32).map(|index| (index * 9 % 256) as u8).collect();

    let expected: Vec<Vec<u16>> = (0..3)
        .map(|plane| reference(&base[plane], &overlay[plane], &mask))
        .collect();

    let backend = backend();
    let args = serde_json::json!({});
    let filter = built_blend(&backend, format, args);
    let base_bytes: Vec<Vec<u8>> = base.iter().map(|plane| widen(plane)).collect();
    let overlay_bytes: Vec<Vec<u8>> = overlay.iter().map(|plane| widen(plane)).collect();
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base: base_bytes,
        overlay: overlay_bytes,
        mask,
        mask_chroma: None,
    });

    let blended: Vec<Vec<u16>> = planes.iter().map(|plane| unwiden(plane)).collect();
    assert_eq!(blended, expected);
}

#[test]
fn an_rgb_frame_blends_every_channel() {
    let format = frame(8, 4, FrameLayout::Rgb, SampleFormat::U8);
    let base = planes_of(&format, |plane, sample| ((plane * 20 + sample) % 256) as u8);
    let overlay = planes_of(&format, |plane, sample| (255 - (plane * 20 + sample) % 256) as u8);
    let mask: Vec<u8> = (0..32).map(|index| (index * 9 % 256) as u8).collect();
    let args = serde_json::json!({});

    check_u8(format, base, overlay, mask, args);
}

#[test]
fn a_full_mask_takes_the_overlay_and_an_empty_one_the_base() {
    let format = frame(8, 8, FrameLayout::Grey, SampleFormat::U8);
    let base = planes_of(&format, |_, sample| sample as u8);
    let overlay = planes_of(&format, |_, sample| 255 - sample as u8);

    let backend = backend();
    let full_args = serde_json::json!({});
    let full_filter = built_blend(&backend, format, full_args);
    let full_base = base.clone();
    let full_overlay = overlay.clone();
    let full_mask = vec![255; 64];
    let full = backend.build(BlendOnce {
        filter: full_filter,
        format,
        base: full_base,
        overlay: full_overlay,
        mask: full_mask,
        mask_chroma: None,
    });

    let empty_args = serde_json::json!({});
    let empty_filter = built_blend(&backend, format, empty_args);
    let empty_base = base.clone();
    let empty_overlay = overlay.clone();
    let empty_mask = vec![0; 64];
    let empty = backend.build(BlendOnce {
        filter: empty_filter,
        format,
        base: empty_base,
        overlay: empty_overlay,
        mask: empty_mask,
        mask_chroma: None,
    });

    assert_eq!(full, overlay);
    assert_eq!(empty, base);
}

#[test]
fn planes_leaves_the_other_channels_as_base() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let format = frame(8, 4, layout, SampleFormat::U8);
    let base = planes_of(&format, |plane, sample| ((plane * 10 + sample) % 256) as u8);
    let overlay = planes_of(&format, |plane, sample| (255 - (plane * 10 + sample) % 256) as u8);
    let mask: Vec<u8> = (0..32).map(|index| (index * 9 % 256) as u8).collect();
    let args = serde_json::json!({ "planes": [0] });

    let base_samples = narrow(&base[0]);
    let overlay_samples = narrow(&overlay[0]);
    let blended = reference(&base_samples, &overlay_samples, &mask);
    let expected_plane0: Vec<u8> = blended.iter().map(|sample| *sample as u8).collect();

    let backend = backend();
    let base_for_blend = base.clone();
    let filter = built_blend(&backend, format, args);
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base: base_for_blend,
        overlay,
        mask,
        mask_chroma: None,
    });

    assert_eq!(planes[0], expected_plane0);
    assert_eq!(planes[1], base[1]);
    assert_eq!(planes[2], base[2]);
}

#[test]
fn a_420_frame_of_odd_size_blends_every_plane() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let format = frame(7, 5, layout, SampleFormat::U8);
    let base = planes_of(&format, |plane, sample| ((plane * 13 + sample * 5) % 256) as u8);
    let overlay = planes_of(&format, |plane, sample| {
        (255 - (plane * 13 + sample * 5) % 256) as u8
    });

    let mask: Vec<u8> = (0..35).map(|index| (index * 7 % 256) as u8).collect();
    let chroma = mask_chroma_for(layout).expect("4:2:0 carries chroma");
    let mask_chroma = downscaled_chroma(&mask, format.width, format.height, chroma);
    assert_eq!(mask_chroma.len(), 12);

    let args = serde_json::json!({});

    check_u8_subsampled(format, base, overlay, mask, mask_chroma, args);
}

#[test]
fn a_422_frame_blends_every_plane() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv422,
    };
    let format = frame(8, 4, layout, SampleFormat::U8);
    let base = planes_of(&format, |plane, sample| ((plane * 17 + sample * 3) % 256) as u8);
    let overlay = planes_of(&format, |plane, sample| {
        (255 - (plane * 17 + sample * 3) % 256) as u8
    });

    let mask: Vec<u8> = (0..32).map(|index| (index * 11 % 256) as u8).collect();
    let chroma = mask_chroma_for(layout).expect("4:2:2 carries chroma");
    let mask_chroma = downscaled_chroma(&mask, format.width, format.height, chroma);

    let args = serde_json::json!({});

    check_u8_subsampled(format, base, overlay, mask, mask_chroma, args);
}

#[test]
fn a_420_ten_bit_frame_blends_at_its_depth() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let sample = SampleFormat::U16 { depth: Depth::TEN };
    let format = frame(8, 4, layout, sample);
    let base = ramps_of(&format, 31);
    let overlay = ramps_of(&format, 97);

    let mask: Vec<u8> = (0..32).map(|index| (index * 9 % 256) as u8).collect();
    let chroma = mask_chroma_for(layout).expect("4:2:0 carries chroma");
    let mask_chroma = downscaled_chroma(&mask, format.width, format.height, chroma);

    let expected: Vec<Vec<u16>> = format
        .channels()
        .iter()
        .enumerate()
        .map(|(index, channel)| {
            let mask_plane = reference_mask_plane(&format, *channel, &mask, &mask_chroma);
            reference(&base[index], &overlay[index], mask_plane)
        })
        .collect();

    let backend = backend();
    let args = serde_json::json!({});
    let filter = built_blend(&backend, format, args);
    let base_bytes: Vec<Vec<u8>> = base.iter().map(|plane| widen(plane)).collect();
    let overlay_bytes: Vec<Vec<u8>> = overlay.iter().map(|plane| widen(plane)).collect();
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base: base_bytes,
        overlay: overlay_bytes,
        mask,
        mask_chroma: Some(mask_chroma),
    });

    let blended: Vec<Vec<u16>> = planes.iter().map(|plane| unwiden(plane)).collect();
    assert_eq!(blended, expected);
}

#[test]
fn planes_leaves_the_other_planes_as_base() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let format = frame(8, 4, layout, SampleFormat::U8);
    let base = planes_of(&format, |plane, sample| ((plane * 10 + sample) % 256) as u8);
    let overlay = planes_of(&format, |plane, sample| (255 - (plane * 10 + sample) % 256) as u8);

    let mask: Vec<u8> = (0..32).map(|index| (index * 9 % 256) as u8).collect();
    let chroma = mask_chroma_for(layout).expect("4:2:0 carries chroma");
    let mask_chroma = downscaled_chroma(&mask, format.width, format.height, chroma);
    let args = serde_json::json!({ "planes": [1, 2] });

    let expected_u = reference_u8(&base[1], &overlay[1], &mask_chroma);
    let expected_v = reference_u8(&base[2], &overlay[2], &mask_chroma);

    let backend = backend();
    let base_for_blend = base.clone();
    let filter = built_blend(&backend, format, args);
    let planes = backend.build(BlendOnce {
        filter,
        format,
        base: base_for_blend,
        overlay,
        mask,
        mask_chroma: Some(mask_chroma),
    });

    assert_eq!(planes[0], base[0]);
    assert_eq!(planes[1], expected_u);
    assert_eq!(planes[2], expected_v);
}
