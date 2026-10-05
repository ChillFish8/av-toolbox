use std::collections::VecDeque;

use av_denoise_core::{
    ChannelMode,
    DenoisingMode,
    DevicePlane,
    Engine,
    Geometry,
    HqParams,
    Nl4d,
    Nl4dOptions,
    NlmTuning,
    Nlmeans,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    nl4d_spatial_radius_for,
};
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    ComputeClient,
    Depth,
    FrameFormat,
    FrameLayout,
    SampleFormat,
    Subsampling,
};
use cubecl::prelude::Runtime;
use cubecl::server::Handle;
use serde_json::json;

use super::{noisy_clip, run, yuv420};
use crate::stream::{Input, Output};
use crate::support::{backend, frame};

/// Every plane of every output frame, in channel order.
type ClipBytes = Vec<Vec<Vec<u8>>>;

/// One engine of the reference, its settings and the planes of each frame it takes.
#[derive(Clone)]
struct ReferenceEngine {
    options: ReferenceOptions,
    channels: ChannelMode,
    planes: Vec<usize>,
    width: u32,
    height: u32,
}

impl ReferenceEngine {
    fn plane_bytes(&self, sample_bytes: usize) -> usize {
        self.width as usize * self.height as usize * sample_bytes
    }
}

#[derive(Clone, Copy)]
enum ReferenceOptions {
    Nlmeans(NlmeansAlgorithm),
    Nl4d(Nl4dOptions),
}

/// Drives av-denoise-core directly over a clip, one engine per plane group.
///
/// Each frame's planes are uploaded into word-rounded handles, and planes no engine covers are
/// copied from the input.
struct Reference {
    engines: Vec<ReferenceEngine>,
    sample: av_denoise_core::SampleFormat,
    sample_bytes: usize,
    clip: Vec<Input>,
}

fn rounded(bytes: usize) -> usize {
    bytes.next_multiple_of(4)
}

fn built_engine<R: Runtime>(
    client: &ComputeClient<R>,
    spec: &ReferenceEngine,
    sample: av_denoise_core::SampleFormat,
) -> Box<dyn Engine> {
    let geometry = Geometry {
        width: spec.width,
        height: spec.height,
        channels: spec.channels,
        input: sample,
        output: sample,
    };

    match spec.options {
        ReferenceOptions::Nlmeans(algorithm) => {
            let engine = Nlmeans::new(client, algorithm, geometry).expect("the reference should build");
            Box::new(engine)
        },
        ReferenceOptions::Nl4d(options) => {
            let engine = Nl4d::new(client, options, geometry).expect("the reference should build");
            Box::new(engine)
        },
    }
}

/// Uploads one plane, zeroing the tail up to a whole word.
fn uploaded<R: Runtime>(client: &ComputeClient<R>, plane: &[u8]) -> Handle {
    let mut bytes = plane.to_vec();
    let padded = rounded(bytes.len());
    bytes.resize(padded, 0);

    client.create_from_slice(&bytes)
}

/// Pushes `input` into every engine and returns how many frames the last one has ready.
fn pushed<R: Runtime>(
    client: &ComputeClient<R>,
    built: &mut [Box<dyn Engine>],
    specs: &[ReferenceEngine],
    input: &Input,
) -> usize {
    let mut ready = 0;

    for (engine, spec) in built.iter_mut().zip(specs) {
        let handles: Vec<Handle> = spec
            .planes
            .iter()
            .map(|plane_index| uploaded(client, &input.planes[*plane_index]))
            .collect();
        let device_planes: Vec<DevicePlane<'_>> = handles
            .iter()
            .map(|handle| DevicePlane::new(handle, spec.width, spec.height))
            .collect();

        ready = engine.push(&device_planes).expect("the reference should push");
    }

    ready
}

/// Emits the oldest ready frame of every engine over the planes of `held`.
fn emitted<R: Runtime>(
    client: &ComputeClient<R>,
    built: &mut [Box<dyn Engine>],
    specs: &[ReferenceEngine],
    sample_bytes: usize,
    held: &Input,
) -> Vec<Vec<u8>> {
    let mut planes = held.planes.clone();

    for (engine, spec) in built.iter_mut().zip(specs) {
        let bytes = spec.plane_bytes(sample_bytes);
        let handles: Vec<Handle> = spec.planes.iter().map(|_| client.empty(rounded(bytes))).collect();
        let device_planes: Vec<DevicePlane<'_>> = handles
            .iter()
            .map(|handle| DevicePlane::new(handle, spec.width, spec.height))
            .collect();
        engine
            .emit_into(&device_planes)
            .expect("the reference should emit");

        for (plane_index, handle) in spec.planes.iter().zip(&handles) {
            let read = client
                .read_one(handle.clone())
                .expect("the device should hand back the plane");
            planes[*plane_index] = read[..bytes].to_vec();
        }
    }

    planes
}

impl BuildTask for Reference {
    type Output = ClipBytes;

    fn run<R: Runtime>(self, build: &BuildContext<R>) -> Self::Output {
        let client = build.client();
        let mut built: Vec<Box<dyn Engine>> = self
            .engines
            .iter()
            .map(|spec| built_engine(client, spec, self.sample))
            .collect();

        let mut outputs = Vec::with_capacity(self.clip.len());
        let mut held = VecDeque::new();

        for input in &self.clip {
            let ready = pushed(client, &mut built, &self.engines, input);
            held.push_back(input);

            for _ in 0..ready {
                let held_frame = held.pop_front().expect("a held frame");
                let planes = emitted(client, &mut built, &self.engines, self.sample_bytes, held_frame);
                outputs.push(planes);
            }
        }

        let mut tail = 0;
        for engine in built.iter_mut() {
            tail = engine.finish().expect("the reference should finish");
        }

        for _ in 0..tail {
            let held_frame = held.pop_front().expect("a held frame");
            let planes = emitted(client, &mut built, &self.engines, self.sample_bytes, held_frame);
            outputs.push(planes);
        }

        assert_eq!(outputs.len(), self.clip.len());

        outputs
    }
}

fn reference(engines: Vec<ReferenceEngine>, format: &FrameFormat, clip: Vec<Input>) -> ClipBytes {
    let sample = match format.sample {
        SampleFormat::U8 => av_denoise_core::SampleFormat::U8,
        SampleFormat::U16 { depth } => av_denoise_core::SampleFormat::U16 { depth: depth.bits() },
        SampleFormat::F32 => av_denoise_core::SampleFormat::F32,
        SampleFormat::F16 => panic!("av-denoise cannot read f16"),
    };
    let task = Reference {
        engines,
        sample,
        sample_bytes: format.sample.bytes(),
        clip,
    };

    backend().build(task)
}

/// The luma and chroma engines of a 4:2:0 or 4:2:2 clip.
fn split_engines(
    format: &FrameFormat,
    luma: ReferenceOptions,
    chroma: ReferenceOptions,
) -> Vec<ReferenceEngine> {
    let (chroma_width, chroma_height) = format
        .channel_dimensions(Channel::U)
        .expect("a yuv clip carries U");

    vec![
        ReferenceEngine {
            options: luma,
            channels: ChannelMode::Luma,
            planes: vec![0],
            width: format.width,
            height: format.height,
        },
        ReferenceEngine {
            options: chroma,
            channels: ChannelMode::Chroma,
            planes: vec![1, 2],
            width: chroma_width,
            height: chroma_height,
        },
    ]
}

/// What `{}` resolves to for `nlmeans` at the base preset.
fn base_nlmeans() -> NlmeansOptions {
    NlmeansOptions {
        mode: DenoisingMode::Temporal { radius: 2 },
        tuning: NlmTuning {
            search_radius: Some(2),
            ..NlmTuning::default()
        },
        ..NlmeansOptions::default()
    }
}

/// What `{}` resolves to for `nl4d` at the base preset.
fn base_nl4d() -> Nl4dOptions {
    Nl4dOptions {
        temporal_radius: 2,
        spatial_radius: nl4d_spatial_radius_for(av_denoise_core::Preset::Base),
        ..Nl4dOptions::default()
    }
}

/// `noisy_clip`'s samples spread over 10 bits, stored as little-endian words.
fn ten_bit_clip(format: &FrameFormat, frames: u64) -> Vec<Input> {
    let mut clip = noisy_clip(format, frames);

    for input in &mut clip {
        for (plane, channel) in input.planes.iter_mut().zip(format.channels()) {
            let samples = format.channel_samples(*channel).expect("a listed channel");
            let mut bytes = Vec::with_capacity(samples * 2);
            for sample in 0..samples {
                let base = plane[sample % plane.len()] as u16;
                let value = (base * 4 + (sample as u16 % 4)) & 0x3ff;
                bytes.extend_from_slice(&value.to_le_bytes());
            }

            *plane = bytes;
        }
    }

    clip
}

fn assert_same(stage: &str, found: &[Output], expected: &[Vec<Vec<u8>>]) {
    assert_eq!(found.len(), expected.len(), "{stage}");

    for (index, (output, planes)) in found.iter().zip(expected).enumerate() {
        assert_eq!(output.planes.len(), planes.len(), "{stage} frame {index}");

        for (plane, (found_plane, expected_plane)) in output.planes.iter().zip(planes).enumerate() {
            let same = found_plane == expected_plane;
            assert!(
                same,
                "{stage} frame {index} plane {plane} differs from av-denoise"
            );
        }
    }
}

#[test]
fn nlmeans_matches_the_engine_at_8_bit_420() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans()));
    let engines = split_engines(&format, options, options);

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nlmeans", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans", &found, &expected);
}

#[test]
fn nlmeans_hq_matches_the_engine_at_8_bit_420() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let hq = NlmeansHqOptions {
        nlm: base_nlmeans(),
        hq: HqParams::default(),
    };
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Hq(hq));
    let engines = split_engines(&format, options, options);

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nlmeans-hq", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans-hq", &found, &expected);
}

#[test]
fn nl4d_matches_the_engine_at_8_bit_420() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let options = ReferenceOptions::Nl4d(base_nl4d());
    let engines = split_engines(&format, options, options);

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nl4d", json!({}), format, vec![clip]);

    assert_same("av-denoise/nl4d", &found, &expected);
}

#[test]
fn nlmeans_hq_matches_the_engine_at_10_bit_420() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let format = yuv420(64, 32, ten_bit);
    let clip = ten_bit_clip(&format, 6);
    let hq = NlmeansHqOptions {
        nlm: base_nlmeans(),
        hq: HqParams::default(),
    };
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Hq(hq));
    let engines = split_engines(&format, options, options);

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nlmeans-hq", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans-hq", &found, &expected);
}

#[test]
fn nl4d_matches_the_engine_on_grey() {
    let format = frame(64, 32, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let clip = noisy_clip(&format, 6);
    let engines = vec![ReferenceEngine {
        options: ReferenceOptions::Nl4d(base_nl4d()),
        channels: ChannelMode::Luma,
        planes: vec![0],
        width: 64,
        height: 32,
    }];

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nl4d", json!({}), format, vec![clip]);

    assert_same("av-denoise/nl4d", &found, &expected);
}

#[test]
fn fused_nlmeans_matches_the_engine_at_444() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let format = frame(32, 32, layout, SampleFormat::U8, ChromaLocation::Left);
    let clip = noisy_clip(&format, 5);
    let engines = vec![ReferenceEngine {
        options: ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans())),
        channels: ChannelMode::Yuv,
        planes: vec![0, 1, 2],
        width: 32,
        height: 32,
    }];

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nlmeans", json!({ "fused": true }), format, vec![clip]);

    assert_same("av-denoise/nlmeans", &found, &expected);
}

#[test]
fn an_odd_sized_clip_matches_the_engine() {
    let format = yuv420(33, 17, SampleFormat::U8);
    let clip = noisy_clip(&format, 5);
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans()));
    let engines = split_engines(&format, options, options);

    let expected = reference(engines, &format, clip.clone());
    let found = run("av-denoise/nlmeans", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans", &found, &expected);
}

#[test]
fn per_plane_strength_reaches_each_engine() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 4);
    let luma = NlmeansOptions {
        tuning: NlmTuning {
            strength: Some(2.0),
            ..base_nlmeans().tuning
        },
        ..base_nlmeans()
    };
    let chroma = NlmeansOptions {
        tuning: NlmTuning {
            strength: Some(0.5),
            ..base_nlmeans().tuning
        },
        ..base_nlmeans()
    };
    let engines = split_engines(
        &format,
        ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(luma)),
        ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(chroma)),
    );

    let expected = reference(engines, &format, clip.clone());
    let args = json!({ "luma_strength": 2.0, "chroma_strength": 0.5 });
    let found = run("av-denoise/nlmeans", args, format, vec![clip]);

    assert_same("av-denoise/nlmeans", &found, &expected);
}

#[test]
fn a_wide_search_radius_builds() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 1);

    let outputs = run(
        "av-denoise/nlmeans",
        json!({ "search_radius": 5 }),
        format,
        vec![clip],
    );

    assert_eq!(outputs.len(), 1);
}
