use std::error::Error;

use av_toolbox_core::{InputArgument, PipelineRegistry, PreparedStage, StageHandle, StageKind, StreamType};
use av_toolbox_gpu::{
    Channel,
    ChromaLocation,
    ColourRange,
    FrameLayout,
    FrameMeta,
    ResolvedFormats,
    SampleFormat,
    StreamFormat,
    Subsampling,
};
use serde_json::json;

use super::{float_clip, noisy_clip, output_codes, run, yuv420};
use crate::stream::Input;
use crate::support::{assert_codes_within, backend, float_planes, float_samples, frame, widened};

fn error_chain(error: &dyn Error) -> String {
    let mut message = error.to_string();
    let mut source = error.source();
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }

    message
}

/// Whether any code of `found` is more than half a code from the 8-bit `input`.
fn differs(found: &[f32], input: &[u8]) -> bool {
    let codes: Vec<u16> = input.iter().map(|code| u16::from(*code)).collect();
    let expected = widened(&codes);

    found
        .iter()
        .zip(&expected)
        .any(|(got, want)| (got - want).abs() > 0.5)
}

#[test]
fn every_stage_keeps_frame_count_order_and_metadata() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);

    for stage in ["av-denoise/nlmeans", "av-denoise/nlmeans-hq", "av-denoise/nl4d"] {
        let outputs = run(stage, json!({}), format, vec![clip.clone()]);

        assert_eq!(outputs.len(), clip.len(), "{stage}");
        for (output, input) in outputs.iter().zip(&clip) {
            assert_eq!(output.meta, input.meta, "{stage}");
        }
    }
}

#[test]
fn every_stage_changes_the_selected_planes() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 5);

    for stage in ["av-denoise/nlmeans", "av-denoise/nlmeans-hq", "av-denoise/nl4d"] {
        let outputs = run(stage, json!({}), format, vec![clip.clone()]);

        let codes: Vec<Vec<Vec<f32>>> = outputs
            .iter()
            .map(|output| output_codes(&format, output))
            .collect();

        let changed = codes
            .iter()
            .zip(&clip)
            .any(|(planes, input)| differs(&planes[0], &input.planes[0]));
        assert!(changed, "{stage} left luma untouched");

        let chroma_changed = codes
            .iter()
            .zip(&clip)
            .any(|(planes, input)| differs(&planes[1], &input.planes[1]));
        assert!(chroma_changed, "{stage} left chroma untouched");
    }
}

#[test]
fn unselected_planes_are_the_input_as_f32() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);

    let outputs = run(
        "av-denoise/nlmeans",
        json!({ "planes": ["y"] }),
        format,
        vec![clip.clone()],
    );

    for (output, input) in outputs.iter().zip(&clip) {
        let codes: Vec<Vec<u16>> = input
            .planes
            .iter()
            .map(|plane| plane.iter().map(|code| u16::from(*code)).collect())
            .collect();
        let expected = float_planes(&format, &codes, 8);

        for plane in [1, 2] {
            let found = float_samples(&output.planes[plane]);
            let wanted = float_samples(&expected[plane]);
            let label = format!("plane {plane}");
            assert_codes_within(&found, &wanted, 1e-6, &label);
        }
    }
}

/// How far a flat plane may drift, about 0.02 of an 8-bit code. NL4D's transforms move it by about 4e-5.
const FLAT_TOLERANCE: f32 = 1e-4;

#[test]
fn an_eight_bit_limited_clip_lands_on_full_range() {
    let mut format = yuv420(64, 32, SampleFormat::U8);
    format.colour.range = ColourRange::Limited;
    let luma_samples = format.channel_samples(Channel::Y).expect("a listed channel");
    let chroma_samples = format.channel_samples(Channel::U).expect("a listed channel");
    let clip: Vec<Input> = (0..4)
        .map(|index| Input {
            meta: FrameMeta::at(index),
            planes: vec![
                vec![235; luma_samples],
                vec![128; chroma_samples],
                vec![128; chroma_samples],
            ],
        })
        .collect();

    for stage in ["av-denoise/nlmeans", "av-denoise/nlmeans-hq", "av-denoise/nl4d"] {
        let outputs = run(stage, json!({}), format, vec![clip.clone()]);

        for output in &outputs {
            let luma = float_samples(&output.planes[0]);
            let chroma = float_samples(&output.planes[1]);
            let white = vec![1.0; luma.len()];
            let neutral = vec![0.0; chroma.len()];

            assert_codes_within(&luma, &white, FLAT_TOLERANCE, stage);
            assert_codes_within(&chroma, &neutral, FLAT_TOLERANCE, stage);
        }
    }
}

#[test]
fn a_chunk_shorter_than_the_window_emits_every_frame() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 2);

    let nlmeans = run(
        "av-denoise/nlmeans",
        json!({ "temporal_radius": 4 }),
        format,
        vec![clip.clone()],
    );
    let nl4d = run(
        "av-denoise/nl4d",
        json!({ "temporal_radius": 4 }),
        format,
        vec![clip.clone()],
    );

    assert_eq!(nlmeans.len(), 2);
    assert_eq!(nl4d.len(), 2);
}

#[test]
fn each_chunk_is_denoised_on_its_own() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let opening = clip[..3].to_vec();
    let closing = clip[3..].to_vec();

    for stage in ["av-denoise/nlmeans", "av-denoise/nlmeans-hq", "av-denoise/nl4d"] {
        let together = run(stage, json!({}), format, vec![opening.clone(), closing.clone()]);
        let alone = run(stage, json!({}), format, vec![closing.clone()]);

        assert_eq!(together.len(), 6, "{stage}");
        assert_eq!(&together[3..], &alone[..], "{stage}");
    }
}

#[test]
fn nl4d_refuses_chroma_below_its_patch_size() {
    let format = yuv420(12, 12, SampleFormat::U8);
    let backend = backend();
    let mut registry = PipelineRegistry::new();
    av_toolbox_filters::register(&mut registry).expect("the filters register");

    let handle = StageHandle::new(1);
    let arguments = [InputArgument::positional(handle, StreamType::Frames)];
    let prepared = registry
        .prepare(StageKind::Filter, "av-denoise/nl4d", &json!({}), &arguments)
        .expect("should prepare");
    let PreparedStage::Filter(prepared) = prepared else {
        panic!("expected a filter");
    };

    let stream_format = StreamFormat::Frames(format);
    let formats = ResolvedFormats::new(vec![Some(stream_format)], Some(stream_format));
    let Err(error) = prepared.build(&backend, &formats) else {
        panic!("a 6x6 chroma plane should be refused");
    };

    let message = error_chain(&error);
    assert!(message.contains("av-denoise/nl4d"), "{message}");
    assert!(message.contains("must be at least"), "{message}");
}

#[test]
fn unselected_float_chroma_is_the_input() {
    let format = yuv420(64, 32, SampleFormat::F32);
    let clip = float_clip(&format, 4);

    let outputs = run(
        "av-denoise/nlmeans",
        json!({ "planes": ["y"] }),
        format,
        vec![clip.clone()],
    );

    for (output, input) in outputs.iter().zip(&clip) {
        assert_eq!(output.planes[1], input.planes[1]);
        assert_eq!(output.planes[2], input.planes[2]);
    }
}

/// The stream runner checks every emitted frame's format, so a lost chroma location fails there.
#[test]
fn every_stage_keeps_the_chroma_location() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };
    let format = frame(64, 32, layout, SampleFormat::U8, ChromaLocation::TopLeft);
    let clip = noisy_clip(&format, 4);

    for stage in ["av-denoise/nlmeans", "av-denoise/nlmeans-hq", "av-denoise/nl4d"] {
        let outputs = run(stage, json!({}), format, vec![clip.clone()]);

        assert_eq!(outputs.len(), clip.len(), "{stage}");
    }
}
