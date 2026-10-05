use std::time::Duration;

use av_toolbox_cubek::deinterlace::qtgmc::Parity;
use av_toolbox_cubek::testing::{
    FieldClip,
    QtgmcReferenceSettings,
    field_clips,
    reference_bob,
    reference_qtgmc,
};
use av_toolbox_gpu::{
    ChromaLocation,
    FieldOrder,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    FrameRate,
    SampleFormat,
    Subsampling,
};
use serde_json::json;

use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter, frame};

const STAGE: &str = "deinterlace/qtgmc";

/// How far an output sample of the cubic run may sit from the CPU reference, in code values.
const REFERENCE_TOLERANCE: u8 = 1;

fn clip(name: &str) -> FieldClip {
    let clips = field_clips();

    clips
        .into_iter()
        .find(|clip| clip.name == name)
        .expect("the clip exists")
}

/// The 8-bit, top field first, 25 frames per second format of `clip`.
fn interlaced(clip: &FieldClip) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let base = frame(
        clip.width as u32,
        clip.height as u32,
        layout,
        SampleFormat::U8,
        ChromaLocation::Left,
    );

    FrameFormat {
        frame_rate: FrameRate::new(25, 1),
        field_order: Some(FieldOrder::TopFirst),
        ..base
    }
}

fn deinterlaced(input: FrameFormat) -> FrameFormat {
    FrameFormat {
        frame_rate: FrameRate::new(50, 1),
        field_order: Some(FieldOrder::Progressive),
        ..input
    }
}

fn inputs(clip: &FieldClip) -> Vec<Input> {
    clip.frames
        .iter()
        .enumerate()
        .map(|(index, planes)| {
            let timestamp = Duration::from_millis(40 * index as u64);
            let meta = FrameMeta {
                timestamp: Some(timestamp),
                ..FrameMeta::at(index as u64)
            };
            let bytes = planes
                .iter()
                .map(|plane| plane.iter().map(|sample| *sample as u8).collect())
                .collect();

            Input { meta, planes: bytes }
        })
        .collect()
}

fn run(args: serde_json::Value, clip: &FieldClip) -> Vec<Output> {
    let backend = backend();
    let input = interlaced(clip);
    let output = deinterlaced(input);
    let filter = built_filter(&backend, STAGE, input, output, args);
    let stream = Stream {
        filter,
        input,
        output,
        chunks: vec![inputs(clip)],
    };

    backend.build(stream)
}

fn bob_interpolate(plane: &[f32], width: usize, height: usize, parity: Parity) -> Vec<f32> {
    reference_bob(plane, (width, height), parity)
}

fn code_frames(clip: &FieldClip) -> Vec<[Vec<f32>; 3]> {
    clip.frames
        .iter()
        .map(|planes| {
            planes
                .each_ref()
                .map(|plane| plane.iter().map(|sample| *sample as f32).collect())
        })
        .collect()
}

fn narrowed(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect()
}

#[test]
fn the_stage_runs_a_clip_end_to_end() {
    let clip = clip("field_pan_8");
    let outputs = run(json!({}), &clip);

    let luma = clip.width * clip.height;
    let chroma = luma / 4;
    assert_eq!(outputs.len(), 2 * clip.frames.len(), "one output per field");

    for (index, output) in outputs.iter().enumerate() {
        assert_eq!(
            output.meta.index, index as u64,
            "output {index} is numbered in order"
        );

        let sizes: Vec<usize> = output.planes.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![luma, chroma, chroma], "output {index} plane sizes");

        let lowest = output.planes[0].iter().min().copied();
        let highest = output.planes[0].iter().max().copied();
        assert!(lowest < highest, "output {index} luma is flat");
    }

    let half_frame = Duration::from_millis(20);
    assert_eq!(outputs[1].meta.timestamp, Some(half_frame));
}

#[test]
fn a_cubic_run_lands_near_the_reference() {
    let clip = clip("field_pan_8");
    let outputs = run(json!({ "spatial": { "kind": "cubic" } }), &clip);

    let frames = code_frames(&clip);
    let luma = (clip.width, clip.height);
    let settings = QtgmcReferenceSettings::default();
    let expected = reference_qtgmc(&frames, luma, clip.bits, &settings, &bob_interpolate);
    assert_eq!(outputs.len(), expected.len());

    for (index, (output, planes)) in outputs.iter().zip(&expected).enumerate() {
        for (plane_index, plane) in planes.iter().enumerate() {
            let wanted = narrowed(plane);
            let found = &output.planes[plane_index];
            let largest = found
                .iter()
                .zip(&wanted)
                .map(|(got, want)| got.abs_diff(*want))
                .max()
                .unwrap_or(0);

            assert!(
                largest <= REFERENCE_TOLERANCE,
                "field {index} plane {plane_index} is {largest} codes from the reference"
            );
        }
    }
}
