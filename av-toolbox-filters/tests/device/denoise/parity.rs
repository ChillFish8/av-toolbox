use std::collections::VecDeque;

use av_denoise_core::{
    ChromaLayout,
    ChromaSiting,
    DenoisingMode,
    DevicePlane,
    Engine,
    EngineSpec,
    Geometry,
    HqParams,
    Nl4dOptions,
    NlmTuning,
    NlmeansAlgorithm,
    NlmeansHqOptions,
    NlmeansOptions,
    TuningConfig,
    TuningMode,
    build_engine,
};
use av_toolbox_cubek::values::FloatCodes;
use av_toolbox_gpu::{
    BuildContext,
    BuildTask,
    Channel,
    ChromaLocation,
    Client,
    Depth,
    FrameFormat,
    FrameLayout,
    FrameMeta,
    SampleFormat,
    Subsampling,
    Tuning,
};
use cubecl::server::Handle;
use serde_json::json;

use super::{float_clip, noisy_clip, run, yuv420};
use crate::stream::{Input, Output};
use crate::support::{assert_codes_within, backend, code_bits, float_planes, float_samples, frame, range_of};

/// Every F32 plane of every output frame, in channel order.
type ClipSamples = Vec<Vec<Vec<f32>>>;

/// The settings one plane group of the reference is denoised with.
#[derive(Clone, Copy)]
enum ReferenceOptions {
    Nlmeans(NlmeansAlgorithm),
    Nl4d(Nl4dOptions),
}

/// Drives one av-denoise-core engine directly over a clip into F32.
///
/// Each frame's planes are uploaded into word-rounded handles.
struct Reference {
    spec: EngineSpec,
    /// How each plane's `code / max` output maps onto full range samples, for integer input.
    units: Option<Vec<FloatCodes>>,
    clip: Vec<Input>,
}

fn rounded(bytes: usize) -> usize {
    bytes.next_multiple_of(4)
}

/// The tuning the stages build their engines with on `build`.
fn tuning_of(build: &BuildContext) -> TuningConfig {
    let mode = match build.tuning() {
        Tuning::Auto => TuningMode::Auto,
        Tuning::Cached => TuningMode::Cached,
        Tuning::Off => TuningMode::Off,
    };

    TuningConfig {
        mode,
        ..TuningConfig::default()
    }
}

/// Uploads one plane, zeroing the tail up to a whole word.
fn uploaded(client: &Client, plane: &[u8]) -> Handle {
    let mut bytes = plane.to_vec();
    let padded = rounded(bytes.len());
    bytes.resize(padded, 0);

    client.create_from_slice(&bytes)
}

/// `handles` as the planes of one frame of `geometry`, luma first.
fn frame_planes<'a>(geometry: &Geometry, handles: &'a [Handle]) -> Vec<DevicePlane<'a>> {
    handles
        .iter()
        .enumerate()
        .map(|(index, handle)| {
            let (width, height) = geometry.plane_dims(index);
            DevicePlane::new(handle, width, height)
        })
        .collect()
}

/// Pushes every plane of `input` into the engine and returns how many frames are ready.
fn pushed(client: &Client, engine: &mut Box<dyn Engine>, geometry: &Geometry, input: &Input) -> usize {
    let handles: Vec<Handle> = input.planes.iter().map(|plane| uploaded(client, plane)).collect();
    let planes = frame_planes(geometry, &handles);

    engine.push(&planes).expect("the reference should push")
}

/// Maps an engine's `code / max` float onto the full range sample of the same code.
fn unit_to_float(unit: f32, codes: FloatCodes) -> f32 {
    let code = unit * codes.peak;

    (code - codes.offset) / codes.scale
}

/// Emits the engine's oldest ready frame as F32 planes.
fn emitted(
    client: &Client,
    engine: &mut Box<dyn Engine>,
    geometry: &Geometry,
    units: Option<&[FloatCodes]>,
) -> Vec<Vec<f32>> {
    let plane_count = geometry.chroma.plane_count();
    let sizes: Vec<usize> = (0..plane_count)
        .map(|index| {
            let (width, height) = geometry.plane_dims(index);
            width as usize * height as usize * size_of::<f32>()
        })
        .collect();
    let handles: Vec<Handle> = sizes.iter().map(|bytes| client.empty(*bytes)).collect();
    let planes = frame_planes(geometry, &handles);
    engine.emit_into(&planes).expect("the reference should emit");

    let mut samples = Vec::with_capacity(plane_count);
    for (index, (handle, bytes)) in handles.iter().zip(&sizes).enumerate() {
        let read = client
            .read_one(handle.clone())
            .expect("the device should hand back the plane");
        let floats = float_samples(&read[..*bytes]);

        let plane = match units {
            Some(units) => floats
                .iter()
                .map(|unit| unit_to_float(*unit, units[index]))
                .collect(),
            None => floats,
        };
        samples.push(plane);
    }

    samples
}

impl BuildTask for Reference {
    type Output = ClipSamples;

    fn run(self, build: &BuildContext) -> Self::Output {
        let client = build.client();
        let tuning = tuning_of(build);
        let mut engine = build_engine(build.cubecl_device(), client, self.spec, &tuning)
            .expect("the reference should build");
        let geometry = self.spec.geometry();
        let units = self.units.as_deref();

        let mut outputs = Vec::with_capacity(self.clip.len());
        let mut held = VecDeque::new();

        for input in &self.clip {
            let ready = pushed(client, &mut engine, &geometry, input);
            held.push_back(input);

            for _ in 0..ready {
                held.pop_front().expect("a held frame");
                let planes = emitted(client, &mut engine, &geometry, units);
                outputs.push(planes);
            }
        }

        let tail = engine.finish().expect("the reference should finish");
        for _ in 0..tail {
            held.pop_front().expect("a held frame");
            let planes = emitted(client, &mut engine, &geometry, units);
            outputs.push(planes);
        }

        assert_eq!(outputs.len(), self.clip.len());

        outputs
    }
}

/// How each plane of `format` maps from `code / max` onto full range samples, for integer input.
fn units_of(format: &FrameFormat) -> Option<Vec<FloatCodes>> {
    if format.sample == SampleFormat::F32 {
        return None;
    }

    let bits = code_bits(format.sample);
    let range = range_of(format);
    let units = format
        .channels()
        .iter()
        .map(|channel| {
            let chroma = matches!(channel, Channel::U | Channel::V);
            FloatCodes::new(bits, range, chroma)
        })
        .collect();

    Some(units)
}

fn reference(spec: EngineSpec, format: &FrameFormat, clip: Vec<Input>) -> ClipSamples {
    let units = units_of(format);
    let task = Reference { spec, units, clip };

    backend().build(task)
}

/// The frame of `format` an engine is built for, writing F32.
fn geometry_of(format: &FrameFormat) -> Geometry {
    let chroma = match format.layout {
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv420,
        } => ChromaLayout::Yuv420,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv422,
        } => ChromaLayout::Yuv422,
        FrameLayout::Yuv {
            subsampling: Subsampling::Yuv444,
        } => ChromaLayout::Yuv444,
        _ => ChromaLayout::Gray,
    };
    let input = match format.sample {
        SampleFormat::U8 => av_denoise_core::SampleFormat::U8,
        SampleFormat::U16 { depth } => av_denoise_core::SampleFormat::U16 { depth: depth.bits() },
        SampleFormat::F32 => av_denoise_core::SampleFormat::F32,
    };

    Geometry {
        width: format.width,
        height: format.height,
        chroma,
        input,
        output: av_denoise_core::SampleFormat::F32,
        chroma_siting: ChromaSiting::Unknown,
    }
}

/// The engine denoising each plane group of `format` with its own options, `None` passing it through.
fn spec_of(
    format: &FrameFormat,
    luma: Option<ReferenceOptions>,
    chroma: Option<ReferenceOptions>,
) -> EngineSpec {
    let geometry = geometry_of(format);

    let first = luma.or(chroma).expect("a plane group to denoise");

    match first {
        ReferenceOptions::Nlmeans(_) => EngineSpec::Nlmeans {
            luma: luma.map(nlmeans_algorithm),
            chroma: chroma.map(nlmeans_algorithm),
            geometry,
        },
        ReferenceOptions::Nl4d(_) => EngineSpec::Nl4d {
            luma: luma.map(nl4d_options),
            chroma: chroma.map(nl4d_options),
            geometry,
        },
    }
}

fn nlmeans_algorithm(options: ReferenceOptions) -> NlmeansAlgorithm {
    let ReferenceOptions::Nlmeans(algorithm) = options else {
        panic!("expected nlmeans options");
    };

    algorithm
}

fn nl4d_options(options: ReferenceOptions) -> Nl4dOptions {
    let ReferenceOptions::Nl4d(options) = options else {
        panic!("expected nl4d options");
    };

    options
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

/// What `{}` resolves to for `nl4d`.
fn base_nl4d() -> Nl4dOptions {
    Nl4dOptions::default()
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

/// Adds `amount` to every chroma sample of a frame's float planes.
fn shift_chroma(planes: &mut [Vec<u8>], amount: f32) {
    for plane in planes.iter_mut().skip(1) {
        let (words, _) = plane.as_chunks::<4>();

        let mut bytes = Vec::with_capacity(plane.len());
        for word in words {
            let value = f32::from_le_bytes(*word);
            let shifted = value + amount;
            bytes.extend_from_slice(&shifted.to_le_bytes());
        }

        *plane = bytes;
    }
}

/// Moves every chroma sample of a frame's F32 planes by `amount`.
fn shift_float_chroma(planes: &mut [Vec<f32>], amount: f32) {
    for plane in planes.iter_mut().skip(1) {
        for sample in plane.iter_mut() {
            *sample += amount;
        }
    }
}

fn assert_same(stage: &str, found: &[Output], expected: &[Vec<Vec<f32>>]) {
    assert_eq!(found.len(), expected.len(), "{stage}");

    for (index, (output, planes)) in found.iter().zip(expected).enumerate() {
        assert_eq!(output.planes.len(), planes.len(), "{stage} frame {index}");

        for (plane, (found_plane, expected_plane)) in output.planes.iter().zip(planes).enumerate() {
            let samples = float_samples(found_plane);
            let label = format!("{stage} frame {index} plane {plane}");
            assert_codes_within(&samples, expected_plane, 1e-5, &label);
        }
    }
}

#[test]
fn nlmeans_matches_the_engine_at_8_bit_420() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans()));
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
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
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
    let found = run("av-denoise/nlmeans-hq", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans-hq", &found, &expected);
}

#[test]
fn nl4d_matches_the_engine_at_8_bit_420() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 6);
    let options = ReferenceOptions::Nl4d(base_nl4d());
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
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
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
    let found = run("av-denoise/nlmeans-hq", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans-hq", &found, &expected);
}

#[test]
fn an_odd_sized_ten_bit_clip_matches_the_engine() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let format = yuv420(33, 17, ten_bit);
    let clip = ten_bit_clip(&format, 5);
    let hq = NlmeansHqOptions {
        nlm: base_nlmeans(),
        hq: HqParams::default(),
    };
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Hq(hq));
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
    let found = run("av-denoise/nlmeans-hq", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans-hq", &found, &expected);
}

#[test]
fn a_ten_bit_luma_selection_matches_the_full_selection_luma() {
    let ten_bit = SampleFormat::U16 { depth: Depth::TEN };
    let format = yuv420(64, 32, ten_bit);
    let clip = ten_bit_clip(&format, 5);

    let full = run("av-denoise/nlmeans", json!({}), format, vec![clip.clone()]);
    let luma_only = run(
        "av-denoise/nlmeans",
        json!({ "planes": ["y"] }),
        format,
        vec![clip.clone()],
    );

    assert_eq!(luma_only.len(), clip.len());
    for (index, ((partial, whole), input)) in luma_only.iter().zip(&full).zip(&clip).enumerate() {
        let partial_luma = float_samples(&partial.planes[0]);
        let whole_luma = float_samples(&whole.planes[0]);
        let label = format!("frame {index} luma");
        assert_codes_within(&partial_luma, &whole_luma, 1e-5, &label);

        let codes: Vec<Vec<u16>> = input
            .planes
            .iter()
            .map(|plane| {
                let (words, _) = plane.as_chunks::<2>();
                words.iter().map(|word| u16::from_le_bytes(*word)).collect()
            })
            .collect();
        let expected = float_planes(&format, &codes, 10);

        for plane in [1, 2] {
            let found = float_samples(&partial.planes[plane]);
            let wanted = float_samples(&expected[plane]);
            let label = format!("frame {index} plane {plane}");
            assert_codes_within(&found, &wanted, 1e-6, &label);
        }
    }
}

/// Compares plane `plane` of every frame of two runs within the parity tolerance.
fn assert_plane_matches(found: &[Output], expected: &[Output], plane: usize) {
    assert_eq!(found.len(), expected.len());

    for (index, (partial, whole)) in found.iter().zip(expected).enumerate() {
        let partial_plane = float_samples(&partial.planes[plane]);
        let whole_plane = float_samples(&whole.planes[plane]);
        let label = format!("frame {index} plane {plane}");
        assert_codes_within(&partial_plane, &whole_plane, 1e-5, &label);
    }
}

/// The chroma of one patch, cycling through neutral and saturated values.
fn patch_chroma(patch: usize) -> (f32, f32) {
    const LEVELS: [f32; 3] = [0.0, 0.3, -0.3];

    (LEVELS[patch % 3], LEVELS[(patch / 3) % 3])
}

/// Float frames of 16x16 chroma patches, with luma grain that grows with each patch's saturation.
fn colour_grain_clip(format: &FrameFormat, frames: u64) -> Vec<Input> {
    let width = format.width as usize;
    let height = format.height as usize;
    let chroma_width = width.div_ceil(2);
    let patches_across = chroma_width.div_ceil(16);
    let patch_of = |x: usize, y: usize| (y / 16) * patches_across + x / 16;

    (0..frames)
        .map(|index| {
            let mut state = (index as u32 + 1).wrapping_mul(2_654_435_761);
            let mut luma = Vec::with_capacity(width * height * 4);
            for y in 0..height {
                for x in 0..width {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let grain = (state % 1000) as f32 / 1000.0 - 0.5;
                    let (u, v) = patch_chroma(patch_of(x / 2, y / 2));
                    let amount = 0.02 + 0.2 * (u.abs() + v.abs());
                    let value = 0.5 + grain * amount;
                    luma.extend_from_slice(&value.to_le_bytes());
                }
            }

            let mut planes = vec![luma];
            for channel in [0, 1] {
                let chroma_height = height.div_ceil(2);
                let mut bytes = Vec::with_capacity(chroma_width * chroma_height * 4);
                for y in 0..chroma_height {
                    for x in 0..chroma_width {
                        let (u, v) = patch_chroma(patch_of(x, y));
                        let value = if channel == 0 { u } else { v };
                        bytes.extend_from_slice(&value.to_le_bytes());
                    }
                }
                planes.push(bytes);
            }

            Input {
                meta: FrameMeta::at(index),
                planes,
            }
        })
        .collect()
}

/// nl4d measures colour on luma from the chroma planes, so chroma it passes through must still be centred.
#[test]
fn a_float_nl4d_luma_selection_matches_the_full_selection_luma() {
    let format = yuv420(256, 128, SampleFormat::F32);
    let clip = colour_grain_clip(&format, 6);

    let full = run("av-denoise/nl4d", json!({}), format, vec![clip.clone()]);
    let luma_only = run(
        "av-denoise/nl4d",
        json!({ "planes": ["y"] }),
        format,
        vec![clip.clone()],
    );

    assert_plane_matches(&luma_only, &full, 0);
    for (output, input) in luma_only.iter().zip(&clip) {
        assert_eq!(output.planes[1], input.planes[1]);
        assert_eq!(output.planes[2], input.planes[2]);
    }
}

#[test]
fn an_eight_bit_chroma_selection_matches_the_full_selection_chroma() {
    let format = yuv420(64, 32, SampleFormat::U8);
    let clip = noisy_clip(&format, 5);

    let full = run("av-denoise/nlmeans", json!({}), format, vec![clip.clone()]);
    let chroma_only = run(
        "av-denoise/nlmeans",
        json!({ "planes": ["u", "v"] }),
        format,
        vec![clip],
    );

    assert_plane_matches(&chroma_only, &full, 1);
    assert_plane_matches(&chroma_only, &full, 2);
}

#[test]
fn nl4d_matches_the_engine_on_grey() {
    let format = frame(64, 32, FrameLayout::Grey, SampleFormat::U8, ChromaLocation::Left);
    let clip = noisy_clip(&format, 6);
    let options = ReferenceOptions::Nl4d(base_nl4d());
    let spec = spec_of(&format, Some(options), None);

    let expected = reference(spec, &format, clip.clone());
    let found = run("av-denoise/nl4d", json!({}), format, vec![clip]);

    assert_same("av-denoise/nl4d", &found, &expected);
}

#[test]
fn nlmeans_matches_the_engine_at_444() {
    let layout = FrameLayout::Yuv {
        subsampling: Subsampling::Yuv444,
    };
    let format = frame(32, 32, layout, SampleFormat::U8, ChromaLocation::Left);
    let clip = noisy_clip(&format, 5);
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans()));
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
    let found = run("av-denoise/nlmeans", json!({}), format, vec![clip]);

    assert_same("av-denoise/nlmeans", &found, &expected);
}

#[test]
fn an_odd_sized_clip_matches_the_engine() {
    let format = yuv420(33, 17, SampleFormat::U8);
    let clip = noisy_clip(&format, 5);
    let options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(base_nlmeans()));
    let spec = spec_of(&format, Some(options), Some(options));

    let expected = reference(spec, &format, clip.clone());
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
    let luma_options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(luma));
    let chroma_options = ReferenceOptions::Nlmeans(NlmeansAlgorithm::Fast(chroma));
    let spec = spec_of(&format, Some(luma_options), Some(chroma_options));

    let expected = reference(spec, &format, clip.clone());
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

#[test]
fn centred_float_chroma_matches_the_engine() {
    let format = yuv420(64, 32, SampleFormat::F32);
    let clip = float_clip(&format, 6);

    let mut uncentred = clip.clone();
    for input in &mut uncentred {
        shift_chroma(&mut input.planes, 0.5);
    }

    let hq = NlmeansHqOptions {
        nlm: base_nlmeans(),
        hq: HqParams::default(),
    };
    let stages = [
        (
            "av-denoise/nlmeans-hq",
            ReferenceOptions::Nlmeans(NlmeansAlgorithm::Hq(hq)),
        ),
        ("av-denoise/nl4d", ReferenceOptions::Nl4d(base_nl4d())),
    ];

    for (stage, options) in stages {
        let spec = spec_of(&format, Some(options), Some(options));

        let mut expected = reference(spec, &format, uncentred.clone());
        for planes in &mut expected {
            shift_float_chroma(planes, -0.5);
        }

        let found = run(stage, json!({}), format, vec![clip.clone()]);

        assert_same(stage, &found, &expected);
    }
}
