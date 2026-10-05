mod parity;
mod stream;

use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, FrameMeta, SampleFormat, Subsampling};

use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter, frame};

pub(super) fn yuv420(width: u32, height: u32, sample: SampleFormat) -> FrameFormat {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv420,
    };

    frame(width, height, layout, sample, ChromaLocation::Left)
}

/// A deterministic noisy ramp, different on every frame and plane.
pub(super) fn noisy_clip(format: &FrameFormat, frames: u64) -> Vec<Input> {
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

pub(super) fn run(
    stage: &str,
    args: serde_json::Value,
    format: FrameFormat,
    chunks: Vec<Vec<Input>>,
) -> Vec<Output> {
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
