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
use av_toolbox_cubek::resize::{Kernel, Taps, taps};
use av_toolbox_cubek::testing::{reference_resize, reference_resize_wide};
use av_toolbox_gpu::{
    Backend,
    BuildContext,
    BuildTask,
    ChromaLocation,
    ColourInfo,
    Depth,
    Device,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    StreamItem,
    Subsampling,
};
use cubecl::prelude::Runtime;
use serde_json::json;

fn backend() -> Backend {
    Backend::open_default(Device::Default).expect("a backend should open")
}

fn frame(
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
fn built_resize(
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

/// Resizes one frame on the device and reads every output plane back as bytes.
struct ResizeOnce {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
    planes: Vec<Vec<u8>>,
}

impl BuildTask for ResizeOnce {
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
        let resized = self.filter.process(&mut context, inputs).expect("should resize");

        assert_eq!(resized.format(), &self.output);

        self.output
            .channels()
            .iter()
            .map(|channel| {
                let plane = resized.plane(*channel).expect("the output carries this channel");
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

fn resized_to(input: &FrameFormat, width: u32, height: u32) -> FrameFormat {
    FrameFormat {
        width,
        height,
        ..*input
    }
}

fn run(
    stage: &str,
    input: FrameFormat,
    output: FrameFormat,
    args: serde_json::Value,
    planes: Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    let backend = backend();
    let filter = built_resize(&backend, stage, input, output, args);

    backend.build(ResizeOnce {
        filter,
        input,
        output,
        planes,
    })
}

fn assert_within_one(found: &[u8], expected: &[u8]) {
    assert_eq!(found.len(), expected.len());

    for (index, (got, want)) in found.iter().zip(expected).enumerate() {
        assert!(
            got.abs_diff(*want) <= 1,
            "sample {index} was {got}, expected {want}"
        );
    }
}

/// Taps for one axis of a centre-aligned resize, for luma when `factor` is 1, or else for chroma
/// sited `siting` luma samples past `factor * i`.
fn axis_taps(start: f32, length: f32, output: u32, factor: u32, siting: f32, kernel: &Kernel) -> Taps {
    let scale = length / output as f32;
    let luma_position = start + (siting + 0.5) * scale - 0.5;
    let offset = (luma_position - siting) / factor as f32;
    let plane_output = output / factor;

    taps(plane_output, scale, offset, kernel)
}

/// Vertical stripes whose edges sit on the same luma columns in every plane of a 16x8 4:2:0 frame.
fn aligned_stripes(plane: usize, sample: usize) -> u8 {
    let bright = match plane {
        0 => sample % 8 >= 4,
        _ => sample % 4 >= 2,
    };

    if bright { 240 } else { 16 }
}

fn check_420_doubling(location: ChromaLocation, siting_x: f32) {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(16, 8, layout, SampleFormat::U8, location);
    let output = resized_to(&input, 32, 16);
    let planes = planes_of(&input, aligned_stripes);
    let args = json!({ "width": 32, "height": 16 });

    let found = run("resize/bilinear", input, output, args, planes.clone());

    let kernel = Kernel::Bilinear;
    let luma_across = axis_taps(0.0, 16.0, 32, 1, 0.0, &kernel);
    let luma_down = axis_taps(0.0, 8.0, 16, 1, 0.0, &kernel);
    let luma = reference_resize(&planes[0], 16, 8, &luma_across, &luma_down);
    let chroma_across = axis_taps(0.0, 16.0, 32, 2, siting_x, &kernel);
    let chroma_down = axis_taps(0.0, 8.0, 16, 2, 0.5, &kernel);
    let chroma = reference_resize(&planes[1], 8, 4, &chroma_across, &chroma_down);

    assert_within_one(&found[0], &luma);
    assert_within_one(&found[1], &chroma);
    assert_within_one(&found[2], &chroma);
}

#[test]
fn a_left_sited_420_frame_keeps_its_chroma_siting_through_a_doubling() {
    check_420_doubling(ChromaLocation::Left, 0.0);
}

#[test]
fn a_centre_sited_420_frame_keeps_its_chroma_siting_through_a_doubling() {
    check_420_doubling(ChromaLocation::Center, 0.5);
}

#[test]
fn a_crop_window_picks_out_its_region() {
    let input = frame(8, 8, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_to(&input, 4, 2);
    let planes = planes_of(&input, |_, sample| sample as u8);
    let args =
        json!({ "width": 4, "height": 2, "src_left": 2, "src_top": 4, "src_width": 4, "src_height": 2 });

    let found = run("resize/point", input, output, args, planes);

    let expected: Vec<u8> = vec![34, 35, 36, 37, 42, 43, 44, 45];
    assert_eq!(found[0], expected);
}

#[test]
fn a_window_left_of_the_frame_reflects() {
    let input = frame(8, 4, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_to(&input, 8, 4);
    let planes = planes_of(&input, |_, sample| (sample % 8 * 30) as u8);
    let args = json!({ "width": 8, "height": 4, "src_left": -2 });

    let found = run("resize/point", input, output, args, planes);

    let row: Vec<u8> = vec![60, 30, 0, 30, 60, 90, 120, 150];
    let expected: Vec<u8> = row.iter().copied().cycle().take(32).collect();
    assert_eq!(found[0], expected);
}

#[test]
fn an_identity_resize_hands_back_the_input() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(8, 4, layout, SampleFormat::U8, ChromaLocation::Left);
    let planes = planes_of(&input, |plane, sample| (plane * 50 + sample) as u8);
    let args = json!({ "width": 8, "height": 4 });

    let found = run("resize/spline36", input, input, args, planes.clone());

    assert_eq!(found, planes);
}

#[test]
fn an_odd_sized_420_frame_resizes_every_plane() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let input = frame(15, 9, layout, SampleFormat::U8, ChromaLocation::Left);
    let output = resized_to(&input, 16, 10);
    let planes = planes_of(&input, |plane, sample| (plane * 40 + sample * 3) as u8);
    let args = json!({ "width": 16, "height": 10 });

    let found = run("resize/spline16", input, output, args, planes.clone());

    let kernel = Kernel::Spline16;
    let luma_across = axis_taps(0.0, 15.0, 16, 1, 0.0, &kernel);
    let luma_down = axis_taps(0.0, 9.0, 10, 1, 0.0, &kernel);
    let luma = reference_resize(&planes[0], 15, 9, &luma_across, &luma_down);
    let chroma_across = axis_taps(0.0, 15.0, 16, 2, 0.0, &kernel);
    let chroma_down = axis_taps(0.0, 9.0, 10, 2, 0.5, &kernel);
    let chroma = reference_resize(&planes[1], 8, 5, &chroma_across, &chroma_down);

    assert_within_one(&found[0], &luma);
    assert_within_one(&found[1], &chroma);
}

#[test]
fn a_ten_bit_frame_resizes_at_its_depth() {
    let sample = SampleFormat::U16 { depth: Depth::TEN };
    let input = frame(8, 8, FrameLayout::Grey, sample, ChromaLocation::Left);
    let output = resized_to(&input, 20, 8);
    let luma: Vec<u16> = (0..64)
        .map(|index| if index % 8 >= 4 { 1023 } else { 0 })
        .collect();
    let bytes: Vec<u8> = luma.iter().flat_map(|value| value.to_le_bytes()).collect();
    let args = json!({ "width": 20, "height": 8, "taps": 4 });

    let found = run("resize/lanczos", input, output, args, vec![bytes]);

    let (pairs, _) = found[0].as_chunks::<2>();
    let samples: Vec<u16> = pairs.iter().map(|pair| u16::from_le_bytes(*pair)).collect();
    let kernel = Kernel::Lanczos { taps: 4 };
    let across = axis_taps(0.0, 8.0, 20, 1, 0.0, &kernel);
    let down = axis_taps(0.0, 8.0, 8, 1, 0.0, &kernel);
    let expected = reference_resize_wide(&luma, 8, 8, &across, &down, 1023);

    let brightest = samples.iter().max().copied().unwrap_or(0);
    assert!(brightest <= 1023);
    assert_eq!(samples.len(), expected.len());

    for (got, want) in samples.iter().zip(&expected) {
        assert!(got.abs_diff(*want) <= 1, "{got} was not within 1 of {want}");
    }
}
