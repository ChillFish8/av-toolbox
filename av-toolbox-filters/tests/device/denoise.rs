use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, FrameMeta, SampleFormat, Subsampling};
use serde_json::json;

use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter, frame};

fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

/// A deterministic noisy ramp, different on every frame and plane.
fn noisy_clip(format: &FrameFormat, frames: u64) -> Vec<Input> {
    (0..frames)
        .map(|index| {
            let planes = format
                .channels()
                .iter()
                .enumerate()
                .map(|(plane, channel)| {
                    let samples = format.channel_samples(*channel).expect("a listed channel");
                    (0..samples)
                        .map(|sample| {
                            let seed = (sample as u64 * 2_654_435_761 + index * 97 + plane as u64 * 31) % 251;
                            let ramp = (sample % 200) as u64;
                            ((ramp + seed / 8) % 256) as u8
                        })
                        .collect()
                })
                .collect();

            Input {
                meta: FrameMeta::at(index),
                planes,
            }
        })
        .collect()
}

fn run(stage: &str, args: serde_json::Value, format: FrameFormat, chunks: Vec<Vec<Input>>) -> Vec<Output> {
    let backend = backend();
    let filter = built_filter(&backend, stage, format, format, args);
    let stream = Stream {
        filter,
        input: format,
        output: format,
        chunks,
    };

    backend.build(stream)
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

        let changed = outputs
            .iter()
            .zip(&clip)
            .any(|(output, input)| output.planes[0] != input.planes[0]);
        assert!(changed, "{stage} left luma untouched");
    }
}

#[test]
fn unselected_planes_are_the_input() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);

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

    for stage in ["av-denoise/nlmeans", "av-denoise/nl4d"] {
        let together = run(stage, json!({}), format, vec![opening.clone(), closing.clone()]);
        let alone = run(stage, json!({}), format, vec![closing.clone()]);

        assert_eq!(together.len(), 6, "{stage}");
        assert_eq!(&together[3..], &alone[..], "{stage}");
    }
}

#[test]
fn nl4d_refuses_chroma_below_its_patch_size() {
    use av_toolbox_core::{
        InputArgument,
        PipelineRegistry,
        PreparedStage,
        StageHandle,
        StageKind,
        StreamType,
    };
    use av_toolbox_gpu::{ResolvedFormats, StreamFormat};

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

    let message = error.to_string();
    assert!(message.contains("av-denoise/nl4d"), "{message}");
}
