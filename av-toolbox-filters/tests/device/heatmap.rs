use av_toolbox_core::{Filter, Inputs};
use av_toolbox_cubek::heatmap::{PaintLayout, Painting, Style};
use av_toolbox_cubek::matrix::{Coefficients, MatrixSamples, SampleRange};
use av_toolbox_cubek::testing::reference_heatmap;
use av_toolbox_gpu::{
    Attachments,
    BuildContext,
    BuildTask,
    ChromaLocation,
    ColourRange,
    Frame,
    FrameContext,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    HostPool,
    HostStreamItem,
    Matrix,
    MetaValue,
    SampleFormat,
    StreamItem,
    Subsampling,
};
use cubecl::prelude::Runtime;

use crate::support::{backend, built_multi_filter, frame, ten_bit};

const STAGE: &str = "blend/heatmap";
const SCALE: f32 = 0.003;
const WIDTH: u32 = 48;
const HEIGHT: u32 = 36;

/// The output planes and metadata of one painted frame.
struct Painted {
    planes: Vec<Vec<u8>>,
    meta: FrameMeta,
}

/// Paints one map frame over one clip frame through the built stage.
struct PaintOnce {
    filter: Box<dyn Filter>,
    clip: FrameFormat,
    clip_planes: Vec<Vec<u8>>,
    map: FrameFormat,
    map_values: Vec<f32>,
    map_meta: FrameMeta,
}

/// A host frame of `format` holding `planes`, loaded onto the device.
fn loaded(context: &FrameContext, format: &FrameFormat, planes: &[Vec<u8>], meta: FrameMeta) -> StreamItem {
    let host = HostPool::new();
    let mut host_frame = host.alloc_frame(format, meta);
    for (index, channel) in format.channels().iter().enumerate() {
        let plane = host_frame
            .plane_mut(*channel)
            .expect("the format carries this channel");
        plane.copy_from_slice(&planes[index]);
    }

    let frozen = host_frame.freeze();
    let item = HostStreamItem::Frame(frozen);

    context.load_host(&item).expect("the item should load")
}

fn read_planes<R: Runtime>(build: &BuildContext<R>, painted: &Frame, format: &FrameFormat) -> Vec<Vec<u8>> {
    format
        .channels()
        .iter()
        .map(|channel| {
            let plane = painted.plane(*channel).expect("the output carries this channel");
            let handle = plane.buffer().handle().clone();
            let bytes = build
                .client()
                .read_one(handle)
                .expect("the device should hand back the plane");
            bytes.to_vec()
        })
        .collect()
}

impl BuildTask for PaintOnce {
    type Output = Painted;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();

        let clip_meta = FrameMeta::at(0);
        let clip = loaded(&context, &self.clip, &self.clip_planes, clip_meta);
        let map_bytes: Vec<u8> = self
            .map_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let map_planes = vec![map_bytes];
        let map = loaded(&context, &self.map, &map_planes, self.map_meta);

        self.filter
            .start_of_stream(&mut context)
            .expect("should start the stream");

        let inputs = Inputs::new(vec![Some(clip), Some(map)]);
        let mut emitted = self.filter.process(&mut context, inputs).expect("should process");

        assert_eq!(emitted.len(), 1);

        let painted = emitted.remove(0);

        assert_eq!(painted.format(), &self.clip);

        let planes = read_planes(build, &painted, &self.clip);
        let meta = painted.meta().clone();

        Painted { planes, meta }
    }
}

fn clip_format() -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let mut format = frame(WIDTH, HEIGHT, layout, ten_bit(), ChromaLocation::Left);
    format.colour.matrix = Matrix::Bt709;
    format.colour.range = ColourRange::Limited;

    format
}

fn map_format() -> FrameFormat {
    frame(
        WIDTH,
        HEIGHT,
        FrameLayout::Grey,
        SampleFormat::F32,
        ChromaLocation::Left,
    )
}

/// Limited range 10-bit code values, luma between 64 and 940 and chroma between 64 and 960.
fn patterned_clip(format: &FrameFormat) -> Vec<Vec<f32>> {
    format
        .channels()
        .iter()
        .enumerate()
        .map(|(plane, channel)| {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            let span = if plane == 0 { 876 } else { 896 };
            (0..samples)
                .map(|index| (64 + (index * 37 + plane * 11) % (span + 1)) as f32)
                .collect()
        })
        .collect()
}

/// A map that peaks at twice the scale in the middle of the frame and fades to nothing.
fn blob_map() -> Vec<f32> {
    let centre_x = WIDTH as f32 / 2.0;
    let centre_y = HEIGHT as f32 / 2.0;
    let spread = WIDTH as f32 / 5.0;

    (0..WIDTH * HEIGHT)
        .map(|index| {
            let distance_x = (index % WIDTH) as f32 - centre_x;
            let distance_y = (index / WIDTH) as f32 - centre_y;
            let squared = distance_x * distance_x + distance_y * distance_y;
            2.0 * SCALE * (-squared / (2.0 * spread * spread)).exp()
        })
        .collect()
}

fn ten_bit_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| (*value as u16).to_le_bytes())
        .collect()
}

fn ten_bit_values(bytes: &[u8]) -> Vec<f32> {
    let (pairs, _) = bytes.as_chunks::<2>();

    pairs
        .iter()
        .map(|pair| f32::from(u16::from_le_bytes(*pair)))
        .collect()
}

fn painted(map_meta: FrameMeta, picture: &[Vec<f32>]) -> Painted {
    let clip_planes = picture.iter().map(|values| ten_bit_bytes(values)).collect();
    let args = serde_json::json!({ "style": "heat", "scale": SCALE });
    let clip = clip_format();
    let map = map_format();

    let backend = backend();
    let (filter, output) = built_multi_filter(&backend, STAGE, &[clip, map], args);

    assert_eq!(output, clip);

    let task = PaintOnce {
        filter,
        clip,
        clip_planes,
        map,
        map_values: blob_map(),
        map_meta,
    };

    backend.build(task)
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
fn the_overlay_matches_the_reference() {
    let format = clip_format();
    let picture = patterned_clip(&format);
    let map_meta = FrameMeta::at(0);
    let output = painted(map_meta, &picture);

    let painting = Painting {
        style: Style::Heat,
        scale: SCALE,
        layout: PaintLayout::Yuv {
            width_shift: 1,
            height_shift: 1,
        },
        samples: MatrixSamples::U16 { bits: 10 },
        range: SampleRange::Limited,
        coefficients: Coefficients::Bt709,
    };
    let picture_planes = [
        picture[0].as_slice(),
        picture[1].as_slice(),
        picture[2].as_slice(),
    ];
    let map = blob_map();
    let expected = reference_heatmap(picture_planes, &map, WIDTH, HEIGHT, &painting);

    for (plane, wanted) in expected.iter().enumerate() {
        let found = ten_bit_values(&output.planes[plane]);

        assert_eq!(found.len(), wanted.len(), "plane {plane}");

        for (index, (got, want)) in found.iter().zip(wanted).enumerate() {
            let stored = (want + 0.5).floor().clamp(0.0, 1023.0);
            assert!(
                (got - stored).abs() <= 1.0,
                "plane {plane} sample {index} was {got}, expected {stored}",
            );
        }
    }
}

#[test]
fn the_map_frames_metadata_is_kept() {
    let format = clip_format();
    let picture = patterned_clip(&format);
    let mut map_meta = FrameMeta::at(0);
    map_meta.attach("score", MetaValue::Float(9.5));

    let output = painted(map_meta, &picture);

    let mut attachments = output.meta.attachments.clone();
    attachments.stamp("vship/cvvdp");
    let score = float_named(&attachments, "score");

    assert_eq!(score, 9.5);
}
