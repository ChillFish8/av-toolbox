use std::time::Duration;

use av_toolbox_core::{Filter, Inputs};
use av_toolbox_cubek::deinterlace::BwdifField;
use av_toolbox_cubek::testing::{BwdifReferencePlanes, reference_bwdif};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    ChromaLocation,
    FieldOrder,
    Frame,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    HostPool,
    HostStreamItem,
    SampleFormat,
    StreamItem,
    Subsampling,
};
use cubecl::prelude::Runtime;
use serde_json::json;

use crate::support::{assert_within_one, backend, built_filter, frame};

const STAGE: &str = "deinterlace/bwdif";

/// One frame handed to the filter, its metadata and every plane's bytes.
#[derive(Clone)]
struct Input {
    meta: FrameMeta,
    planes: Vec<Vec<u8>>,
}

/// One frame the filter emitted, its metadata and every plane's bytes.
#[derive(Debug, Clone, PartialEq)]
struct Output {
    meta: FrameMeta,
    planes: Vec<Vec<u8>>,
}

/// Runs every chunk of `chunks` through one filter instance in turn, starting and ending the stream around each.
struct Stream {
    filter: Box<dyn Filter>,
    input: FrameFormat,
    output: FrameFormat,
    chunks: Vec<Vec<Input>>,
}

fn read_back<R: Runtime>(build: &BuildContext<R>, format: &FrameFormat, emitted: Frame) -> Output {
    assert_eq!(emitted.format(), format);

    let mut planes = Vec::new();
    for channel in format.channels() {
        let plane = emitted.plane(*channel).expect("the output carries this channel");
        let handle = plane.buffer().handle().clone();
        let bytes = build
            .client()
            .read_one(handle)
            .expect("the device should hand back the plane");
        planes.push(bytes.to_vec());
    }

    Output {
        meta: emitted.meta().clone(),
        planes,
    }
}

impl BuildTask for Stream {
    type Output = Vec<Output>;

    fn run<R: Runtime>(mut self, build: &BuildContext<R>) -> Self::Output {
        let mut context = build.frame_context();
        let host = HostPool::new();
        let mut outputs = Vec::new();

        for chunk in self.chunks {
            self.filter
                .start_of_stream(&mut context)
                .expect("should start the stream");

            for input in chunk {
                let mut frame = host.alloc_frame(&self.input, input.meta);
                for (index, channel) in self.input.channels().iter().enumerate() {
                    let plane = frame
                        .plane_mut(*channel)
                        .expect("the format carries this channel");
                    plane.copy_from_slice(&input.planes[index]);
                }

                let frozen = frame.freeze();
                let item = HostStreamItem::Frame(frozen);
                let loaded: StreamItem = context.load_host(&item).expect("the item should load");
                let inputs = Inputs::new(vec![Some(loaded)]);

                let emitted = self.filter.process(&mut context, inputs).expect("should process");
                for frame in emitted {
                    let output = read_back(build, &self.output, frame);
                    outputs.push(output);
                }
            }

            let flushed = self
                .filter
                .end_of_stream(&mut context)
                .expect("should end the stream");
            for frame in flushed {
                let output = read_back(build, &self.output, frame);
                outputs.push(output);
            }
        }

        outputs
    }
}

fn interlaced(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    FrameFormat {
        frame_rate: FrameRate::new(25, 1),
        field_order: Some(FieldOrder::TopFirst),
        ..frame(width, height, layout, sample, ChromaLocation::Left)
    }
}

/// The format `deinterlace/bwdif` emits for `input`, at twice its rate when `double` is set.
fn deinterlaced(input: FrameFormat, double: bool) -> FrameFormat {
    let frame_rate = match (double, input.frame_rate) {
        (true, Some(rate)) => FrameRate::new(rate.numerator() * 2, rate.denominator()),
        (_, unchanged) => unchanged,
    };

    FrameFormat {
        frame_rate,
        field_order: Some(FieldOrder::Progressive),
        ..input
    }
}

fn run(
    args: serde_json::Value,
    input: FrameFormat,
    output: FrameFormat,
    chunks: Vec<Vec<Input>>,
) -> Vec<Output> {
    let backend = backend();
    let filter = built_filter(&backend, STAGE, input, output, args);
    let stream = Stream {
        filter,
        input,
        output,
        chunks,
    };

    backend.build(stream)
}

/// A deterministic hash of a sample's position, for noise.
fn scramble(column: u32, row: u32, frame: u32) -> u32 {
    let mut state =
        column.wrapping_mul(0x9e37_79b1) ^ row.wrapping_mul(0x85eb_ca77) ^ frame.wrapping_mul(0xc2b2_ae3d);
    state ^= state >> 15;
    state = state.wrapping_mul(0x2c1b_3c6d);
    state ^= state >> 12;

    state
}

/// An 8-bit plane of a bar that moves `frame` samples right across combed noise.
fn moving_plane(width: u32, height: u32, frame: u32) -> Vec<u8> {
    let shift = (frame * 2) % 32;
    let mut plane = Vec::with_capacity((width * height) as usize);
    for row in 0..height {
        for column in 0..width {
            let inside = (column + 32 - shift) % 32 < 6;
            let base = if inside && row % 2 == 1 { 220 } else { 40 };
            let noise = scramble(column, row, frame) % 17;
            plane.push((base + noise) as u8);
        }
    }

    plane
}

/// Frame `index` of a moving 4:2:0 clip, `width` by `height`, at 25 frames per second.
fn moving_frame(format: &FrameFormat, index: u64) -> Input {
    let frame = index as u32;
    let luma = moving_plane(format.width, format.height, frame);
    let chroma = moving_plane(format.width / 2, format.height / 2, frame + 7);
    let timestamp = Duration::from_millis(40 * index);
    let meta = FrameMeta {
        timestamp: Some(timestamp),
        ..FrameMeta::at(index)
    };

    Input {
        meta,
        planes: vec![luma, chroma.clone(), chroma],
    }
}

fn moving_clip(format: &FrameFormat, indices: std::ops::Range<u64>) -> Vec<Input> {
    indices.map(|index| moving_frame(format, index)).collect()
}

fn widened(bytes: &[u8]) -> Vec<f32> {
    bytes.iter().map(|byte| f32::from(*byte)).collect()
}

/// The 8-bit luma bwdif's reference gives for `field` of `current`, rounded and clamped as the stage narrows.
fn reference_luma(
    format: &FrameFormat,
    previous: &Input,
    current: &Input,
    next: &Input,
    field: BwdifField,
) -> Vec<u8> {
    let previous_values = widened(&previous.planes[0]);
    let current_values = widened(&current.planes[0]);
    let next_values = widened(&next.planes[0]);
    let planes = BwdifReferencePlanes {
        previous: &previous_values,
        current: &current_values,
        next: &next_values,
        spatial: None,
    };

    let values = reference_bwdif(&planes, format.width, format.height, field);

    values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[test]
fn a_static_clip_comes_out_unchanged_at_double_rate() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let still = moving_frame(&input, 0);
    let clip: Vec<Input> = (0..3)
        .map(|index| Input {
            meta: FrameMeta::at(index),
            ..still.clone()
        })
        .collect();

    let outputs = run(json!({}), input, output, vec![clip]);

    assert_eq!(outputs.len(), 6);
    for (index, emitted) in outputs.iter().enumerate() {
        assert_eq!(emitted.meta.index, index as u64);
        assert_eq!(emitted.planes, still.planes, "output {index}");
    }
}

#[test]
fn double_rate_numbers_fields_and_steps_timestamps_by_half_a_frame() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..4);

    let outputs = run(json!({}), input, output, vec![clip]);

    let indices: Vec<u64> = outputs.iter().map(|emitted| emitted.meta.index).collect();
    let timestamps: Vec<Option<Duration>> = outputs.iter().map(|emitted| emitted.meta.timestamp).collect();
    let expected_timestamps: Vec<Option<Duration>> = (0..8)
        .map(|field| Some(Duration::from_millis(20 * field)))
        .collect();

    assert_eq!(indices, (0..8).collect::<Vec<u64>>());
    assert_eq!(timestamps, expected_timestamps);
}

#[test]
fn double_rate_without_a_frame_rate_leaves_the_second_field_without_a_timestamp() {
    let input = FrameFormat {
        frame_rate: None,
        ..interlaced(16, 8, SampleFormat::U8)
    };
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..2);

    let outputs = run(json!({}), input, output, vec![clip]);

    assert_eq!(outputs[0].meta.timestamp, Some(Duration::ZERO));
    assert_eq!(outputs[1].meta.timestamp, None);
    assert_eq!(outputs[2].meta.timestamp, Some(Duration::from_millis(40)));
    assert_eq!(outputs[3].meta.timestamp, None);
}

#[test]
fn single_rate_keeps_every_frame_and_its_metadata() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, false);
    let mut clip = moving_clip(&input, 10..14);
    for input in &mut clip {
        input.meta.chunk = 3;
    }

    let outputs = run(json!({ "rate": "single" }), input, output, vec![clip.clone()]);

    let found: Vec<&FrameMeta> = outputs.iter().map(|emitted| &emitted.meta).collect();
    let expected: Vec<&FrameMeta> = clip.iter().map(|input| &input.meta).collect();
    assert_eq!(found, expected);
}

#[test]
fn every_field_matches_the_reference_on_luma() {
    let input = interlaced(32, 16, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..3);

    let outputs = run(json!({}), input, output, vec![clip.clone()]);

    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let second = BwdifField {
        rebuilt: 0,
        first: false,
    };
    let windows = [(0, 0, 1), (0, 1, 2), (1, 2, 2)];
    for (frame, (previous, current, next)) in windows.into_iter().enumerate() {
        for (offset, field) in [first, second].into_iter().enumerate() {
            let expected = reference_luma(&input, &clip[previous], &clip[current], &clip[next], field);
            let found = &outputs[frame * 2 + offset].planes[0];

            let label = format!("frame {frame}, field {offset}");
            assert_within_one(found, &expected, &label);
        }
    }
}

#[test]
fn bottom_first_keeps_the_odd_rows_first() {
    let input = FrameFormat {
        field_order: Some(FieldOrder::BottomFirst),
        ..interlaced(32, 16, SampleFormat::U8)
    };
    let output = deinterlaced(input, true);
    let clip = moving_clip(&input, 0..2);

    let outputs = run(json!({}), input, output, vec![clip.clone()]);

    let width = input.width as usize;
    let kept_rows = outputs[0].planes[0]
        .chunks(width)
        .zip(clip[0].planes[0].chunks(width));
    for (row, (found, original)) in kept_rows.enumerate() {
        if row % 2 == 1 {
            assert_eq!(found, original, "row {row}");
        }
    }
}

#[test]
fn each_chunk_is_deinterlaced_on_its_own() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let opening = moving_clip(&input, 0..3);
    let closing = moving_clip(&input, 3..5);

    let together = run(json!({}), input, output, vec![opening, closing.clone()]);
    let alone = run(json!({}), input, output, vec![closing]);

    assert_eq!(together.len(), 10);
    assert_eq!(&together[6..], &alone[..]);
}

#[test]
fn a_one_frame_chunk_uses_its_frame_for_every_neighbour() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, true);
    let only = moving_frame(&input, 0);

    let outputs = run(json!({}), input, output, vec![vec![only.clone()]]);

    assert_eq!(outputs.len(), 2);
    let first = BwdifField {
        rebuilt: 1,
        first: true,
    };
    let expected = reference_luma(&input, &only, &only, &only, first);
    assert_within_one(&outputs[0].planes[0], &expected, "the first field");
}

#[test]
fn an_unselected_plane_is_the_input() {
    let input = interlaced(16, 8, SampleFormat::U8);
    let output = deinterlaced(input, false);
    let clip = moving_clip(&input, 0..3);

    let outputs = run(
        json!({ "rate": "single", "planes": ["y"] }),
        input,
        output,
        vec![clip.clone()],
    );

    for (emitted, original) in outputs.iter().zip(&clip) {
        assert_eq!(emitted.planes[1], original.planes[1]);
        assert_eq!(emitted.planes[2], original.planes[2]);
    }
}
