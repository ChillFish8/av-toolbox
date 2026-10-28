mod grain;
mod parity;
mod quality;
mod stream;

use av_toolbox_gpu::{ChromaLocation, FrameFormat, FrameLayout, FrameMeta, SampleFormat, Subsampling};

use crate::stream::{Input, Output, Stream};
use crate::support::{backend, built_filter, code_bits, frame, planes_as_codes};

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

/// `noisy_clip`'s samples as normalised floats, with chroma centred on 0 the way av-toolbox stores it.
pub(super) fn float_clip(format: &FrameFormat, frames: u64) -> Vec<Input> {
    let mut clip = noisy_clip(format, frames);

    for input in &mut clip {
        for (index, plane) in input.planes.iter_mut().enumerate() {
            let centre = if index == 0 { 0.0 } else { 0.5 };

            let mut bytes = Vec::with_capacity(plane.len() * 4);
            for code in plane.iter() {
                let normalised = f32::from(*code) / 255.0;
                let value = normalised - centre;
                bytes.extend_from_slice(&value.to_le_bytes());
            }

            *plane = bytes;
        }
    }

    clip
}

/// `format` with F32 samples, which every av-denoise stage emits.
pub(super) fn float_of(format: &FrameFormat) -> FrameFormat {
    FrameFormat {
        sample: SampleFormat::F32,
        ..*format
    }
}

/// The F32 planes of `output`, from a run on `format`, as codes of `format`'s depth.
pub(super) fn output_codes(format: &FrameFormat, output: &Output) -> Vec<Vec<f32>> {
    let float = float_of(format);
    let bits = code_bits(format.sample);

    planes_as_codes(&float, &output.planes, bits)
}

pub(super) fn run(
    stage: &str,
    args: serde_json::Value,
    format: FrameFormat,
    chunks: Vec<Vec<Input>>,
) -> Vec<Output> {
    let backend = backend();
    let output = float_of(&format);
    let filter = built_filter(&backend, stage, format, output, args);
    let stream = Stream {
        filter,
        input: format,
        output,
        chunks,
    };

    backend.build(stream)
}
